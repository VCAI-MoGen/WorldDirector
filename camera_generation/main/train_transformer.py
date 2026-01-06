import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime

from config import cfg
from base import Trainer
from torch.utils.data import DataLoader
from pc_encoder import PCEncoder
import clip
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

# -----------------------------
# Transformer Feedforward 모델
# -----------------------------
import torch
import torch.nn as nn

class CameraTransformerModel(nn.Module):
    def __init__(self, 
                 input_dim=9,
                 text_dim=512,
                 point_dim=768,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 traj_len=61):   # N개의 learnable token
        super().__init__()
        self.traj_len = traj_len

        # N개의 learnable tokens (N, d_model)
        self.tokens = nn.Parameter(torch.randn(traj_len, d_model))

        # project embeddings to d_model
        self.text_proj = nn.Linear(text_dim, d_model)
        self.point_proj = nn.Linear(point_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=0,
            batch_first=False   # default: (S, B, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.ln_token = nn.LayerNorm(d_model)
        self.ln_text = nn.LayerNorm(d_model)
        self.ln_point = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, text_emb, point_emb):
        """
        text_emb:  (B, 512)
        point_emb: (B, Np, 768)
        """
        B = text_emb.size(0)
        N = self.traj_len

        # 1) Prepare learnable tokens for batch
        # tokens: (N, d_model) -> (B, N, d_model)
        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1)

        # 2) Prepare conditioning features
        text_feat = self.text_proj(text_emb).unsqueeze(1)      # (B, 1, d_model)
        point_feat = self.point_proj(point_emb.mean(1)).unsqueeze(1)  # (B, 1, d_model)

        # 3) Add conditioning to tokens
        h = torch.cat([self.ln_token(tokens), self.ln_text(text_feat), self.ln_point(point_feat)], dim=1)   # (B, N + 2, d_model)

        # 4) Transformer expects (N, B, d_model)
        h = h.permute(1, 0, 2)

        # 5) Transformer
        h = self.transformer(h)

        # 6) Back to (B, N, d_model)
        h = h.permute(1, 0, 2)
        h = h[:, :-2, :] # (B, N, 9)

        # 7) Output projection
        out = self.output_proj(h)   # (B, N, 9)
        return out


# -----------------------------
# Training
# -----------------------------
def train():
    # hyperparams
    traj_len = 61
    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('../results', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project="transformer-trajectory", name=f"exp_{exp_name}", config={
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs
    })

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()

    # 모델
    model = CameraTransformerModel(traj_len=traj_len).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()

    pc_encoder = PCEncoder()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        for data in pbar:
            point = data['pc']
            text = data['text_prompt']
            traj = data['cam_param'].to(device)
            
            with torch.no_grad():
                point_embed = pc_encoder(point) # (B, N_point, 768)
                text_tokens = clip.tokenize(text).to(device)
                text_embed = clip_model.encode_text(text_tokens).float()
                text_embed /= torch.norm(text_embed, dim=-1, keepdim=True) # (B, 512)
            
            # forward
            out = model(text_embed, point_embed)

            # MSE loss
            # mse_loss = F.mse_loss(out, traj)

            # MAE loss
            mae_loss = loss_fn(out, traj)

            # optional temporal smoothness
            # smooth_loss = ((out[:, 1:, :] - out[:, :-1, :])**2).mean()
            loss = mae_loss#  + 0.1 * smooth_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({
                "train/loss": loss.item(),
                # "train/mse_loss": mse_loss.item(),
                "train/mse_loss": mae_loss.item(),
                # "train/smooth_loss": smooth_loss.item()
            })

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # validation / save
        if epoch % 10 == 0:
            with torch.no_grad():
                out_val = model(text_embed, point_embed)
                val_loss = loss_fn(out_val, traj)
                wandb.log({"val/loss": val_loss})

            matrix_trajectory = torch.eye(4, device=device)[None].repeat(out_val.shape[1], 1, 1)
            raw_trans = out_val[0, :, 6:]
            matrix_trajectory[:, :3, 3] = raw_trans
            rot6d = out_val[0, :, :6]
            raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
            matrix_trajectory[:, :3, :3] = raw_rot
            np.save(os.path.join(output_dir, f'{epoch}.npy'), matrix_trajectory.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))

    print("Training done.")

if __name__ == "__main__":
    train()

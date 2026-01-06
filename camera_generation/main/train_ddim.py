import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime

from config import cfg
from base import Trainer
from model import CameraDiffusionModel
from torch.utils.data import DataLoader
from pc_encoder import PCEncoder
import clip
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

def make_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def compute_alpha(beta):
    return 1.0 - beta

def compute_alpha_cumprod(alpha):
    return torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample_ddim(model, T, alphas, alphas_cumprod, text_emb, point_emb, traj_len, eta=0.0):
    """
    DDIM sampling for camera trajectory
    """
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, 9).to(device)

    for t in reversed(range(T)):
        ts = torch.full((B,), t, device=device, dtype=torch.long)

        a_t = alphas_cumprod[t].view(B, 1, 1)
        a_prev = alphas_cumprod[t-1].view(B, 1, 1) if t > 0 else torch.tensor(1.0, device=device)

        # predict x0
        x0_pred = model(x_t, ts.float(), text_emb, point_emb)

        # epsilon estimate
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_a_prev = torch.sqrt(a_prev)
        sqrt_one_minus_a_t = torch.sqrt(1 - a_t)
        sqrt_one_minus_a_prev = torch.sqrt(1 - a_prev)

        eps = (x_t - sqrt_a_t * x0_pred) / sqrt_one_minus_a_t

        # DDIM update
        x_t = sqrt_a_prev * x0_pred + torch.sqrt(
            1 - a_prev - eta**2 * (1 - a_prev) / (1 - a_t)
        ) * eps

        if eta > 0:
            x_t += eta * torch.randn_like(x_t)

    return x_t

def diffusion_loss(model, x0, ts, alphas_cumprod, text_emb, point_emb):
    B = x0.shape[0]

    noise = torch.randn_like(x0)
    a = alphas_cumprod[ts].view(B, 1, 1)

    x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
    x0_pred = model(x_t, ts.float(), text_emb, point_emb)

    return F.mse_loss(x0_pred, x0)

def train():
    # hyperparams
    T = 100                         # diffusion steps
    traj_len = 61

    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('../results', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project="my-diffusion-training", name=f"exp_{exp_name}", config={
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs
    })

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # diffusion schedule
    beta = make_beta_schedule(T)
    alpha = compute_alpha(beta)
    a_cum = compute_alpha_cumprod(alpha).cuda()

    # model
    model = CameraDiffusionModel().cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    pc_encoder = PCEncoder()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        for data in pbar:
            point = data['pc']
            text = data['text_prompt']
            traj = data['cam_param'].cuda()
            
            with torch.no_grad():
                point_embed = pc_encoder(point) # (B, N, 768)
                text_tokens = clip.tokenize(text).to(device)
                text_embed = clip_model.encode_text(text_tokens).float()
                text_embed /= torch.norm(text_embed, dim=-1, keepdim=True) # (B, 512)

            B = traj.size(0)
            ts = torch.randint(0, T, (B,), device=traj.device)

            loss = diffusion_loss(model, traj, ts, a_cum, text_embed, point_embed)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({
                "train/loss": loss.item()
            })

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # sampling each epoch using DDIM
        if epoch % 100 == 0:
            with torch.no_grad():
                out = sample_ddim(model, T, alpha.cuda(), a_cum, text_embed, point_embed, traj_len, eta=0.0)
                val_loss = F.mse_loss(out, traj)
                wandb.log({
                    "val/loss": val_loss
                })

            matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)

            raw_trans = out[0, :, 6:]
            matrix_trajectory[:, :3, 3] = raw_trans

            rot6d = out[0, :, :6]
            raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
            matrix_trajectory[:, :3, :3] = raw_rot
            np.save(os.path.join(output_dir, f'{epoch}.npy'), matrix_trajectory.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))

    print("Training done.")

if __name__ == "__main__":
    train()

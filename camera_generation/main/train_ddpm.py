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
from vae import CameraVAE
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

def make_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def compute_alpha(beta):
    return 1.0 - beta

def compute_alpha_cumprod(alpha):
    return torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample(model, T, alphas, alphas_cumprod, text_emb, point_emb, traj_len):
    B = text_emb.size(0)

    x_t = torch.randn(B, traj_len, 32).to(text_emb.device)

    for t in reversed(range(T)):
        ts = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        a = alphas_cumprod[ts].view(B, 1, 1)
        a_prev = alphas_cumprod[ts-1].view(B, 1, 1) if t > 0 else a

        x0_pred = model(x_t, ts.float(), text_emb, point_emb)

        if t == 0:
            x_t = x0_pred
        else:
            noise = torch.randn_like(x_t)
            x_t = (
                torch.sqrt(a_prev) * x0_pred +
                torch.sqrt(1 - a_prev) * noise
            )

    return x_t

def diffusion_loss(model, x0, ts, alphas_cumprod, text_emb, point_emb, point_len):
    B = x0.shape[0]

    noise = torch.randn_like(x0)
    a = alphas_cumprod[ts].view(B, 1, 1)

    x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    x0_pred = model(x_t, ts.float(), text_emb, point_emb, point_len)

    return F.mse_loss(x0_pred, x0)
    # return torch.abs(x0_pred - x0).mean()

def train():
    # hyperparams
    T = 100                         # diffusion steps
    traj_len = 13

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
    a_cum = compute_alpha_cumprod(alpha)
    a_cum = a_cum.cuda()

    # model
    model = CameraDiffusionModel().cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    pc_encoder = PCEncoder()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    camera_vae = CameraVAE().cuda()
    camera_vae.load_state_dict(torch.load("/home/ckd248/data/SCVideo/camera_generation/results_vae/20251231_055137/ckpts/9900.pth", map_location="cuda"))

    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        for data in pbar:
            point = data['pc']
            point_len = data['pc_len']
            text = data['text_prompt']
            traj = data['cam_param'].cuda()
            
            with torch.no_grad():
                point_embed = pc_encoder(point, point_len) # (N, 768) (N=n1+n2+...)
                text_tokens = clip.tokenize(text).to(device)
                text_embed = clip_model.encode_text(text_tokens).float()
                text_embed /= torch.norm(text_embed, dim=-1, keepdim=True) # (B, 512)
                traj_latents = camera_vae.encode(traj) # (B, 1 + (T - 1) // 4, 32)

            B = traj_latents.size(0)
            ts = torch.randint(0, T, (B,), device=traj_latents.device)

            loss = diffusion_loss(model, traj_latents, ts, a_cum, text_embed, point_embed, point_len)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({
                "train/loss": loss.item()
            })

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # sampling each epoch
        if epoch % 100 == 0:
            with torch.no_grad():
                out = torch.cat([sample(model, T, alpha.cuda(), a_cum, text_embed[0:1], point_embed[0:1], traj_len), \
                                 sample(model, T, alpha.cuda(), a_cum, text_embed[1:2], point_embed[1:2], traj_len)], dim=0)
                loss = F.mse_loss(out, traj_latents)
                # loss = torch.abs(out - traj).mean()
                wandb.log({
                    "val/loss": loss
                })
                traj_pred = camera_vae.decode(out)
            matrix_trajectory = torch.eye(4, device=device)[None].repeat(traj_pred.shape[1], 1, 1)[None].repeat(traj_pred.shape[0], 1, 1, 1)

            raw_trans = traj_pred[:, :, 6:]
            print(traj[:, :2, 6:])
            print(raw_trans[:, :2])
            matrix_trajectory[:, :, :3, 3] = raw_trans

            rot6d = traj_pred[:, :, :6]
            for idx, rot in enumerate(rot6d):
                raw_rot = compute_rotation_matrix_from_ortho6d(rot)
                matrix_trajectory[idx, :, :3, :3] = raw_rot
            np.save(os.path.join(output_dir, f'{epoch}.npy'), matrix_trajectory.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))


    print("Training done.")

if __name__ == "__main__":
    train()

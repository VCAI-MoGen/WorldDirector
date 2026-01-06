import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime

from config import cfg
from base import Trainer
from vae import CameraVAE
from torch.utils.data import DataLoader
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

def camera_vae_loss(x, x_hat, mu, logvar, beta=1e-3):
    """
    x, x_hat: (B, 9)
    """
    recon_loss = torch.abs(x - x_hat).mean()

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train():
    # hyperparams
    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('../results_vae', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project="my-vae-training", name=f"exp_{exp_name}", config={
        "lr": cfg.lr_vae,
        "batch_size": cfg.batch_size_vae,
        "epochs": cfg.epochs_vae
    })

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # model
    model = CameraVAE().cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        for data in pbar:
            traj = data['cam_param'].cuda()
            traj_pred, mu, logvar = model(traj)
            loss, recon_loss, kl_loss = camera_vae_loss(traj, traj_pred, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/kl_loss": kl_loss.item()
            })

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # sampling each epoch
        if epoch % 100 == 0:
            with torch.no_grad():
                out, _, _ = model(traj)
            matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)[None].repeat(out.shape[0], 1, 1, 1)

            raw_trans = out[:, :, 6:]
            matrix_trajectory[:, :, :3, 3] = raw_trans

            rot6d = out[:, :, :6]
            for idx, rot in enumerate(rot6d):
                raw_rot = compute_rotation_matrix_from_ortho6d(rot)
                matrix_trajectory[idx, :, :3, :3] = raw_rot
            np.save(os.path.join(output_dir, f'{epoch}_vae.npy'), matrix_trajectory.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))
    print("Training done.")

if __name__ == "__main__":
    train()

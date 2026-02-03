import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime

from config import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.common.utils.rotation_utils import compute_rotation_matrix_from_ortho6d
from core_pkg.models.vae_intr import CameraVAE
from torch.utils.data import DataLoader

def camera_vae_loss(x, x_hat, mu, logvar, beta=cfg.vae_beta):
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
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project="my-vae-training", name=f"exp_{exp_name}", config=cfg_dict)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # model
    model = CameraVAE().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        loss_info = {
            "loss": [],
            "recon_loss": [],
            "kl_loss": [],
        }
        for data in pbar:
            data_name = data['data_name']
            traj = data['cam_param'].to(device)
            traj_pred, mu, logvar = model(traj)
            loss, recon_loss, kl_loss = camera_vae_loss(traj, traj_pred, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({
                "train/loss_iter": loss.item(),
                "train/recon_loss_iter": recon_loss.item(),
                "train/kl_loss_iter": kl_loss.item()
            })

            loss_info["loss"].append(loss.item())
            loss_info["recon_loss"].append(recon_loss.item())
            loss_info["kl_loss"].append(kl_loss.item())

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        wandb.log({'train/' + k: sum(v) / len(v) for k, v in loss_info.items()})

        # sampling each epoch
        if epoch % cfg.save_epoch == 0:
            with torch.no_grad():
                z = model.encode(traj)
                out = model.decode(z)
                _, recon_loss_val, _ = camera_vae_loss(traj, out, mu, logvar)
            wandb.log({
                "val/recon_loss": recon_loss_val.item(),
            })
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))
            
    print("Training done.")

if __name__ == "__main__":
    train()

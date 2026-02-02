import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime

from config import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.camera_diffusion_model_director import CameraDiffusionModel
from core_pkg.models.pc_encoder import PCEncoder
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.rotation_utils import compute_rotation_matrix_from_ortho6d
import clip

def make_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def compute_alpha(beta):
    return 1.0 - beta

def compute_alpha_cumprod(alpha):
    return torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample(model, T, alphas, alphas_cumprod, text_emb, text_masks, point_emb, point_mask, traj_len):
    B = text_emb.size(0)

    x_t = torch.randn(B, traj_len, 32).to(text_emb.device)

    for t in reversed(range(T)):
        ts = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        a = alphas_cumprod[ts].view(B, 1, 1)
        a_prev = alphas_cumprod[ts-1].view(B, 1, 1) if t > 0 else a

        x0_pred = model(x_t, ts.float(), text_emb, text_masks, point_emb, point_mask)

        if t == 0:
            x_t = x0_pred
        else:
            noise = torch.randn_like(x_t)
            x_t = (
                torch.sqrt(a_prev) * x0_pred +
                torch.sqrt(1 - a_prev) * noise
            )

    return x_t

def diffusion_loss(model, x0, ts, alphas_cumprod, text_emb, text_masks, point_emb, point_masks):
    B = x0.shape[0]

    noise = torch.randn_like(x0)
    a = alphas_cumprod[ts].view(B, 1, 1)

    x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    x0_pred = model(x_t, ts.float(), text_emb, text_masks, point_emb, point_masks)

    assert torch.isfinite(x0_pred).all()
    assert torch.isfinite(x0).all()

    return F.mse_loss(x0_pred, x0)
    # return torch.abs(x0_pred - x0).mean()

def train():
    # hyperparams
    T = cfg.diffusion_max_step
    traj_len = (cfg.num_frames - 1) // 4 + 1

    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('../results', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(
        project="my-diffusion-training-director",
        name=f"exp_{exp_name}",
        config=cfg_dict
    )

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
    opt.zero_grad(set_to_none=True)

    pc_encoder = PCEncoder()
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
        shard_fn=None)

    camera_vae = CameraVAE().cuda()
    camera_vae.load_state_dict(
        torch.load(
            cfg.vae_ckpt_path,
            map_location="cuda"
        )
    )
    camera_vae.eval()

    global_step = 0

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(wandb.config.epochs):
            pbar = tqdm(trainer.batch_generator)
            loss_info = {"loss": []}

            for step, data in enumerate(pbar):
                data_name = data['data_name']
                point = data['pc']
                text_prompt = data['text_prompt']
                traj = data['cam_param'].cuda()

                with torch.no_grad():
                    pc_embeds, pc_masks, _ = pc_encoder(point)
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                    text_embeds = text_embeds.float()
                    text_masks = text_masks.bool()
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale

                B = traj_latents.size(0)
                ts = torch.randint(0, T, (B,), device=traj_latents.device)

                loss = diffusion_loss(
                    model,
                    traj_latents,
                    ts,
                    a_cum,
                    text_embeds,
                    text_masks,
                    pc_embeds,
                    pc_masks,
                )

                # -------- gradient accumulation --------
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                
                wandb.log({"train/loss": loss.item()}, step=global_step)
                loss_info["loss"].append(loss.item())

                pbar.set_description(
                    f"Epoch {epoch} | Loss {loss.item():.4f}"
                )

            wandb.log(
                {"train/epoch_loss": sum(loss_info["loss"]) / len(loss_info["loss"])},
                step=global_step
            )

            # -------- sampling --------
            if epoch % cfg.save_epoch == 0:
                with torch.no_grad():
                    out = sample(
                        model,
                        T,
                        alpha.cuda(),
                        a_cum,
                        text_embeds,
                        text_masks,
                        pc_embeds,
                        pc_masks,
                        traj_len
                    )

                    val_loss = F.mse_loss(out, traj_latents)
                    wandb.log({"val/loss": val_loss.item()}, step=global_step)

                    traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)

                raw_trans = traj_pred[:, :, 6:]
                print(traj[0, :2, 6:])
                print(raw_trans[0, :2])

                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, f"{epoch}.pth")
                )

    print("Training done.")

if __name__ == "__main__":
    train()

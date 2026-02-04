import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime
import math
import time

from config import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
# from core_pkg.models.camera_diffusion_model_director import CameraDiffusionModel
from core_pkg.models.pc_encoder import PCEncoder
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from diffusers import DDPMScheduler, DDIMScheduler

from torch.optim.lr_scheduler import LambdaLR

@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb=None, point_mask=None):
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, 32).to(device)

    for t in scheduler.timesteps:
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        if point_emb is not None:
            noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        else:
            noise_pred = model(x_t, timesteps.float(), text_emb, text_masks)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        # 1) warmup
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))

        # 2) cosine decay
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train():
    T = cfg.diffusion_max_step 
    traj_len = (cfg.num_frames - 1) // 4 + 1
    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('../results', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project="my-diffusion-training-noise-pred-ddpm", name=f"exp_{exp_name}", config=cfg_dict)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    if cfg.sampling_type == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=T,
        )
    elif cfg.sampling_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=T,
        )    
    noise_scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)

    model = CameraDiffusionModel().cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.lr,
                            # weight_decay=cfg.weight_decay,
                            )
    if cfg.use_lr_scheduler:
        scheduler = get_lr_scheduler(
            optimizer=opt,
            warmup_epochs=cfg.warmup_epoch,
            total_epochs=cfg.epochs
        )

    if cfg.load_points:
        pc_encoder = PCEncoder()
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
        shard_fn=None)

    camera_vae = CameraVAE().cuda()
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location="cuda"))
    camera_vae.eval()

    start_time = time.time()
    global_step = 0
    for epoch in range(wandb.config.epochs):
        pbar = tqdm(trainer.batch_generator)
        loss_info = {"loss": []}
        for step, data in enumerate(pbar):
            if cfg.load_points:
                point = data['pc']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].cuda()

            with torch.no_grad():
                if cfg.load_points:
                    pc_embeds, pc_masks, _ = pc_encoder(point)
                text_embeds, text_masks = text_encoder(text_prompt, device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale

            noise = torch.randn_like(traj_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (traj_latents.shape[0],),
                device=device,
            ).long()
            noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
            if cfg.load_points:
                noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
            else:
                noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            
            wandb.log({"train/loss": loss.item()}, step=global_step)
            loss_info["loss"].append(loss.item())
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

        if cfg.use_lr_scheduler:
            scheduler.step()
        
        wandb.log(
            {"train/epoch_loss": sum(loss_info["loss"]) / len(loss_info["loss"]),
             "train/epoch_lr": opt.param_groups[0]["lr"],
             "epoch": epoch},
            step=global_step
        )

        if epoch % cfg.save_epoch == 0:
            model.eval()
            with torch.no_grad():
                if cfg.load_points:
                    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)
                else:
                    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks)
                
                val_loss = F.mse_loss(out, traj_latents)
                wandb.log({"val/loss": val_loss.item()}, step=global_step)

                traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)

            raw_trans = traj_pred[:, :, 6:]
            print(traj[0, :2, 6:])
            print(raw_trans[0, :2])
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))
            model.train()
    
    duration = time.time() - start_time
    print(f"It took {int(duration)}sec == {duration / 3600}hour")

if __name__ == "__main__":
    train()
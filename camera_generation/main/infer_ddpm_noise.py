import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import datetime

from config import cfg
from core_pkg.common.base import Trainer
from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
from core_pkg.models.pc_encoder import PCEncoder
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.vis_utils import attention_bar, attn_to_color
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics

from diffusers import DDPMScheduler, DDIMScheduler

@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask, from_t=None, traj_latents=None):
    B = text_emb.size(0)
    device = text_emb.device

    x_t = torch.randn(B, traj_len, 32).to(device)
    noise = x_t.clone()
    if from_t is not None and traj_latents is not None:
        from_t = torch.tensor(from_t).to(device)
        x_t = scheduler.add_noise(traj_latents, x_t, from_t)

    for t in scheduler.timesteps:
        if from_t is not None and t > from_t:
            continue
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        print(timesteps)
        noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        print(t, F.mse_loss(noise_pred, noise))
        output = scheduler.step(noise_pred, t, x_t)
        x_t = output.prev_sample
        clean_x = output.pred_original_sample
        print(t, F.mse_loss(clean_x, traj_latents))
    return x_t

def train():
    # hyperparams
    T = cfg.diffusion_max_step 
    traj_len = (cfg.num_frames - 1) // 4 + 1
    cfg.batch_size = 1

    device = 'cuda:0'
    ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260205_043924/ckpts/80.pth'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.abspath(os.path.join('../infer_ddpm', exp_name))
    os.makedirs(result_dir, exist_ok=True)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    if cfg.sampling_type == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
    elif cfg.sampling_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )    
    noise_scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)

    # model
    model = CameraDiffusionModel().cuda()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

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
    pbar = tqdm(trainer.batch_generator)
    with torch.no_grad():
        for step, data in enumerate(pbar):
            data_name = data['data_name']
            text = data['text_prompt']
            traj = data['cam_param'].cuda()
            E0 = data['first_extrinsic'].cuda()
            scale = data['avg_scale'].cuda()
            width = data['width'].cuda()
            height = data['height'].cuda()

            if cfg.load_points:
                point = data['pc']
                pc_embeds, pc_masks, point_obj = pc_encoder(point)
            else:
                pc_embeds, pc_masks = None, None
                
            text_embeds, text_masks = text_encoder(text, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()

            traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale

            # while True:
            #     noise = torch.randn_like(traj_latents)
                
            #     timesteps = torch.randint(
            #         0,
            #         noise_scheduler.config.num_train_timesteps,
            #         (traj_latents.shape[0],),
            #         device=device,
            #     ).long()
            #     # timesteps[0] = 0
            #     noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
            #     noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
            #     loss = F.mse_loss(noise_pred, noise)
            #     print(timesteps)
            #     print(loss)
            #     input()
            # out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)
            out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks, from_t=999, traj_latents=traj_latents)

            val_loss = F.mse_loss(out, traj_latents)
            print("Loss (Latent):", val_loss.item())

            out = camera_vae.decode(out * cfg.vae_latent_scale)
            traj_loss = F.mse_loss(out, traj)
            print("Loss (Traj):", traj_loss.item())

            print("Intrinsics (pred, gt)")
            pred_intrinsics = make_intrinsics(out[:, :, -2:], width, height)
            gt_intrinsics = make_intrinsics(traj[:, :, -2:], width, height)
            print(pred_intrinsics[0][0])
            print(gt_intrinsics[0][0])

            pred_traj = out_to_trajectory(out, scale, E0, device)
            gt_traj = out_to_trajectory(traj, scale, E0, device)
            print(F.mse_loss(camera_vae.decode(traj_latents * cfg.vae_latent_scale), traj))
            print("after denormalize")
            print(pred_traj[0][0])
            print(gt_traj[0][0])

            pred_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_pred.npz')
            np.savez(pred_path, poses=pred_traj[0].cpu().numpy(), intrinsics=pred_intrinsics[0].cpu().numpy())
            gt_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_gt.npz')
            np.savez(gt_path, poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intrinsics[0].cpu().numpy())

            print("Data_name")
            print(data_name)
            print("Prediction, GT path")
            print(pred_path)
            print(gt_path)
            print("Text Prompt")
            print(text)

            # print("Scene save")
            # scene_point = point[0][:, :3].cuda()
            # scene_point = scene_point * scale[0]
            # point[0][:, :3] = scene_point @ E0[0, :3, :3].T + E0[0, :3, 3]
            # torch.save(point[0].cpu(), 'scene.pt')

            if hasattr(model, "cross_attn_weight"):
                cross_attn_weight = model.cross_attn_weight
                

                print('Text max attention')
                text_weight = cross_attn_weight[0, :, :512]
                text_max, text_max_idx = text_weight.max(dim=-1)
                print(text_max, text_max_idx)

                text_tokens = text_encoder.tokenizer.tokenizer.tokenize(text[0])
                attention_path = os.path.join(result_dir, "text_attention.png")
                attention_bar(tokens=text_tokens, attn=text_weight.mean(dim=0)[:len(text_tokens)], save_path=attention_path)

                print("Text attention saved in : ", attention_path)
                print(text_tokens)

                
            #     print('Point max attention')
            #     point_weight = cross_attn_weight[0, :, 512:]
            #     point_max, point_max_idx = point_weight.max(dim=-1)
            #     print(point_max, point_max_idx)

            #     _, _, point_attention = pc_encoder.decode_with_attention(point_obj[0], point_weight.mean(dim=0)[:, None], recover_origin_scale=True) # (N_points, 1)
            #     point_attn_vis = attn_to_color(point_attention[:, 0].detach().cpu().numpy())
            #     point_attention_path = os.path.join(result_dir, "point_attention.pt")
            #     point[0][:, 3:] = torch.tensor(point_attn_vis)
            #     torch.save(point[0].cpu(), point_attention_path)

            #     print('Point attention saved in : ', point_attention_path)
            input()

if __name__ == "__main__":
    train()

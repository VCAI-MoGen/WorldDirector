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
from core_pkg.common.utils.rotation_utils import compute_rotation_matrix_from_ortho6d
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.vis_utils import attention_bar, attn_to_color

def out_to_trajectory(out, scale, e0, device):
    """
    out: (B, N, 9)

    matrix_trajectory: (B, N, 4, 4)
    """
    if out.shape[-1] > 9:
        out = out[:, :, :9]
    matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)[None].repeat(out.shape[0], 1, 1, 1)

    raw_trans = out[:, :, 6:] * scale
    matrix_trajectory[:, :, :3, 3] = raw_trans

    rot6d = out[:, :, :6]
    for idx, rot in enumerate(rot6d):
        raw_rot = compute_rotation_matrix_from_ortho6d(rot)
        matrix_trajectory[idx, :, :3, :3] = raw_rot
    print(f'before denormalize\n{matrix_trajectory[0][0]}')
    matrix_trajectory = e0 @ matrix_trajectory

    return matrix_trajectory

def make_intrinsics(fx_fy, width, height):
    """
    fx_fy: (N, 2)
    width, height: scalars
    returns: (N, 3, 3)
    """
    N = fx_fy.shape[0]
    device = fx_fy.device
    dtype = fx_fy.dtype

    fx = fx_fy[:, 0] * width
    fy = fx_fy[:, 1] * height

    cx = width * 0.5
    cy = height * 0.5

    K = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    return K

def make_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def compute_alpha(beta):
    return 1.0 - beta

def compute_alpha_cumprod(alpha):
    return torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample(model, T, beta, alphas, alphas_cumprod, text_emb, text_masks, point_emb, point_mask, traj_len):
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, 32).to(device)

    for t in reversed(range(T)):
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, ts.float(), text_emb, text_masks, point_emb, point_mask)
        
        alpha_t = alphas[t]
        alpha_cum_t = alphas_cumprod[t]
        beta_t = beta[t]
        sqrt_one_minus_alpha_cum = torch.sqrt(1 - alpha_cum_t)
        
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cum) * eps_pred)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * noise
        else:
            x_t = mean
    return x_t

def diffusion_loss(model, x0, ts, alphas_cumprod, text_emb, point_emb):
    B = x0.shape[0]

    noise = torch.randn_like(x0)
    a = alphas_cumprod[ts].view(B, 1, 1)

    x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    x0_pred = model(x_t, ts.float(), text_emb, point_emb)

    assert torch.isfinite(x0_pred).all()
    assert torch.isfinite(x0).all()

    return F.mse_loss(x0_pred, x0)
    # return torch.abs(x0_pred - x0).mean()

def train():
    # hyperparams
    T = cfg.diffusion_max_step 
    traj_len = (cfg.num_frames - 1) // 4 + 1
    cfg.batch_size = 1

    device = 'cuda:0'
    ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260131_073303/ckpts/160.pth'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.abspath(os.path.join('../infer_ddpm', exp_name))
    os.makedirs(result_dir, exist_ok=True)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # diffusion schedule
    beta = make_beta_schedule(T)
    alpha = compute_alpha(beta)
    a_cum = compute_alpha_cumprod(alpha).cuda()

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
            point = data['pc']
            text = data['text_prompt']
            traj = data['cam_param'].cuda()
            E0 = data['first_extrinsic'].cuda()
            scale = data['avg_scale'].cuda()
            width = data['width']
            height = data['height']

            pc_embeds, pc_masks, point_obj = pc_encoder(point)
            text_embeds, text_masks = text_encoder(text, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()

            traj_latents = camera_vae.encode(traj / cfg.vae_latent_scale)
            out = sample(model, T, beta, alpha.cuda(), a_cum, text_embeds.float(), text_masks.bool(), pc_embeds, pc_masks, traj_len)

            val_loss = F.mse_loss(out, traj_latents)
            print("Loss:", val_loss.item())

            out = camera_vae.decode(out * cfg.vae_latent_scale)

            print("Intrinsics (pred, gt)")
            pred_intrinsics = make_intrinsics(out[0, :, -2:], width[0], height[0])
            gt_intrinsics = make_intrinsics(traj[0, :, -2:], width[0], height[0])
            print(pred_intrinsics[0])
            print(gt_intrinsics[0])

            pred_traj = out_to_trajectory(out, scale, E0, device)
            gt_traj = out_to_trajectory(traj, scale, E0, device)
            print("after denormalize")
            print(pred_traj[0][0])
            print(gt_traj[0][0])

            pred_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_pred.npz')
            np.savez(pred_path, poses=pred_traj[0].cpu().numpy(), intrinsics=pred_intrinsics.cpu().numpy())
            gt_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_gt.npz')
            np.savez(gt_path, poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intrinsics.cpu().numpy())

            print("Data_name")
            print(data_name)
            print("Prediction, GT path")
            print(pred_path)
            print(gt_path)
            print("Text Prompt")
            print(text)

            print("Scene save")
            scene_point = point[0][:, :3].cuda()
            scene_point = scene_point * scale[0]
            point[0][:, :3] = scene_point @ E0[:3, :3].T + E0[:3, 3]
            # torch.save(point[0].cpu(), 'scene.pt')

            if hasattr(model, "cross_attn_weight"):
                cross_attn_weight = model.cross_attn_weight
                text_weight = cross_attn_weight[0, :, :512]
                point_weight = cross_attn_weight[0, :, 512:]

                print('Text max attention')
                text_max, text_max_idx = text_weight.max(dim=-1)
                print(text_max, text_max_idx)

                print('Point max attention')
                point_max, point_max_idx = point_weight.max(dim=-1)
                print(point_max, point_max_idx)

                text_tokens = text_encoder.tokenizer.tokenizer.tokenize(text[0])
                attention_path = os.path.join(result_dir, "text_attention.png")
                attention_bar(tokens=text_tokens, attn=text_weight.mean(dim=0)[:len(text_tokens)], save_path=attention_path)

                print("Text attention saved in : ", attention_path)
                print(text_tokens)

                _, _, point_attention = pc_encoder.decode_with_attention(point_obj[0], point_weight.mean(dim=0)[:, None], recover_origin_scale=True) # (N_points, 1)
                point_attn_vis = attn_to_color(point_attention[:, 0].detach().cpu().numpy())
                point_attention_path = os.path.join(result_dir, "point_attention.pt")
                point[0][:, 3:] = torch.tensor(point_attn_vis)
                torch.save(point[0].cpu(), point_attention_path)

                print('Point attention saved in : ', point_attention_path)
            input()

if __name__ == "__main__":
    train()

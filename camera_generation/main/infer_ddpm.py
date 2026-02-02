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
        print(x0_pred.shape)

        if t == 0:
            x_t = x0_pred
        else:
            noise = torch.randn_like(x_t)
            x_t = (
                torch.sqrt(a_prev) * x0_pred +
                torch.sqrt(1 - a_prev) * noise
            )

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
    ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260125_064948/ckpts/40.pth'
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

            pc_embeds, pc_masks, _ = pc_encoder(point)
            text_embeds, text_masks = text_encoder(text, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()
            traj_latents = camera_vae.encode(traj / cfg.vae_latent_scale)
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

            out = camera_vae.decode(out * cfg.vae_latent_scale)

            pred_traj = out_to_trajectory(out, scale, E0, device)
            gt_traj = out_to_trajectory(traj, scale, E0, device)
            print("after denormalize")
            print(pred_traj[0][0])
            print(gt_traj[0][0])

            pred_npy_file_path = os.path.join(result_dir, f'{data_name[0]}_pred.npy')
            np.save(pred_npy_file_path, pred_traj[0].cpu().numpy())
            gt_npy_file_path = os.path.join(result_dir, f'{data_name[0]}_gt.npy')
            np.save(gt_npy_file_path, gt_traj[0].cpu().numpy())
            print(scale)
            scene_point = point[0][:, :3].cuda()
            scene_point = scene_point * scale[0]
            point[0][:, :3] = scene_point @ E0[:3, :3].T + E0[:3, 3]
            torch.save(point[0].cpu(), 'scene.pt')

            print("Loss:", val_loss.item())
            print(data_name)
            print(pred_npy_file_path)
            print(gt_npy_file_path)
            input()

if __name__ == "__main__":
    train()

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import datetime

from config import cfg
from core_pkg.common.base import Trainer
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.common.utils.rotation_utils import compute_rotation_matrix_from_ortho6d
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def out_to_trajectory(out, scale, e0, width, height, device):
    """
    out: (B, N, 9)

    matrix_trajectory: (B, N, 4, 4)
    """
    if out.shape[-1] != 9:
        out = out[:, :, :9]
    matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)[None].repeat(out.shape[0], 1, 1, 1)

    raw_trans = out[:, :, 6:] * scale
    matrix_trajectory[:, :, :3, 3] = raw_trans

    rot6d = out[:, :, :6]
    for idx, rot in enumerate(rot6d):
        raw_rot = compute_rotation_matrix_from_ortho6d(rot)
        matrix_trajectory[idx, :, :3, :3] = raw_rot
    # print(f'before denormalize\n{matrix_trajectory[0][0]}')
    matrix_trajectory = e0 @ matrix_trajectory

    return matrix_trajectory

def infer():
    # hyperparams
    device = 'cuda:0'
    ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260123_074547/ckpts/900.pth'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.abspath(os.path.join('../infer_vae', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cfg.batch_size = 1
    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # model
    model = CameraVAE().cuda()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    zs = []
    pbar = tqdm(trainer.batch_generator)
    with torch.no_grad():
        for data in pbar:
            data_name = data['data_name']
            traj = data['cam_param'].cuda()
            E0 = data['first_extrinsic'].cuda()
            scale = data['avg_scale'].cuda()
            width = data['width'].cuda()
            height = data['height'].cuda()
            
            # out, _, _ = model(traj)
            z = model.encode(traj)
            traj_hat, mu, log_var = model(traj)
            # print(mu.mean())
            # print(mu.std())
            # print(z.mean())
            # print(z.std())
            # print(z)
            zs.append(z.reshape(-1))
            
            # Random sampling
            # z = torch.randn_like(z).to(z)
            print(z.shape)
            # z = model.reparameterize(mu, log_var)
            # z = z.transpose(-1, -2)
            out = model.decode(z)

            val_loss = torch.abs(traj - out).mean()

            pred_traj = out_to_trajectory(out, scale, E0, width, height, device)
            gt_traj = out_to_trajectory(traj, scale, E0, width, height, device)
            print(pred_traj.shape)
            print(gt_traj.shape)
            print(pred_traj[0][0])
            print(gt_traj[0][0])

            pred_npy_file_path = os.path.join(result_dir, f'{data_name[0]}_pred.npy')
            np.save(pred_npy_file_path, pred_traj[0].cpu().numpy())
            gt_npy_file_path = os.path.join(result_dir, f'{data_name[0]}_gt.npy')
            np.save(gt_npy_file_path, gt_traj[0].cpu().numpy())

            print("Loss:", val_loss.item())
            print(data_name)
            print(pred_npy_file_path)
            print(gt_npy_file_path)

            # # Feature Interpolation
            # if z.shape[0] == 2:
            #     feature_a, feature_b = z[0], z[1]
            #     print(data_name[0], data_name[1])
            #     for i in range(0, 11):
            #         t = i / 10
            #         interpolated_feat = feature_a * t + feature_b * (1 - t)
            #         out = model.decode(interpolated_feat[None])
            #         # scale = torch.ones((1, 1)).to(device)
            #         # E0 = torch.eye(4)[None].to(device)
            #         interpolated_traj = out_to_trajectory(out, scale[0:1], E0[0:1], device)
            #         interpolated_traj_path = os.path.join(result_dir, f'{data_name[0]}_{data_name[1]}_{i}_{10-i}.npy')
            #         np.save(interpolated_traj_path, interpolated_traj[0].cpu().numpy())

            input()
    z_all = torch.stack(zs)
    print(z_all.shape)
    s = z_all.reshape(-1).std()
    print(s.item())
    z_np = z_all.detach().cpu().numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)
    print(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(z_pca[:, 0], label="PC1")
    ax.plot(z_pca[:, 1], label="PC2")

    ax.set_title("PCA Components over Time")
    ax.legend()

    fig.savefig("pca_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=30,      # 10~50 보통
    #     learning_rate='auto',
    #     init='pca',
    #     max_iter=1000,
    #     random_state=42,
    # )

    # z_2d = tsne.fit_transform(z_np)
    # plt.figure(figsize=(6,6))
    # plt.scatter(z_2d[:,0], z_2d[:,1], s=5, alpha=0.6)
    # plt.title("VAE Latent t-SNE")
    # plt.savefig("tsne.png", dpi=300, bbox_inches="tight")
    # plt.close()



if __name__ == "__main__":
    infer()

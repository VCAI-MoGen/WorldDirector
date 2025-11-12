import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import open3d as o3d
import json
from tqdm import tqdm
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--step_size', type=int, default=1000)
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
lr = args.learning_rate
step_size = args.step_size
depth_path = os.path.join(root_path, 'depths_vggt')
mask_path = os.path.join(root_path, 'masks')
scene_depth_path = os.path.join(root_path, 'depths_scene')

device = "cuda:0"

masks = torch.from_numpy(
    np.stack([
    cv2.imread(os.path.join(mask_path, f'{i}.png')) / 255 for i in range(len(os.listdir(depth_path)))
])).bool()[:, :, :, 0].to(device) # (N, H, W)

depths = torch.from_numpy(
    np.stack([
    cv2.resize(np.load(os.path.join(depth_path, f'{i}.npy')), (masks.shape[2], masks.shape[1])) for i in range(len(os.listdir(depth_path)))
])).to(device) # (N, H, W)

scene_depths = torch.from_numpy(
    np.stack([
    cv2.resize(np.load(os.path.join(scene_depth_path, f'{i}.npy'))[:, :, 0], (masks.shape[2], masks.shape[1])) for i in range(len(os.listdir(scene_depth_path)))
])).to(device) # (N, H, W)

print("Max scene depth:", scene_depths.max())
print("Min scene depth:", scene_depths.min())

def depth_to_point_cloud(depth_map, intrinsics):
    H, W = depth_map.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)

def apply_c2w(points, R, t):
    return (R @ points.T).T + t

def depth_to_pc_world(depth_map, cam_param, c2w=True):
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    
    if c2w:
        R_c2w = np.array(cam_param["R"])
        t_c2w = np.array(cam_param["t"])
    else:
        R_c2w = np.array(cam_param["R"]).T
        t_c2w = -R_c2w @ np.array(cam_param["t"])

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    pc_cam = depth_to_point_cloud(depth_map, intrinsics)
    pc_world = apply_c2w(pc_cam, R_c2w, t_c2w)
    return pc_world


import torch

def pc_world_to_depth(points_world, cam_param, w2c=True, device='cuda:0'):
    """
    Args:
        points_world: (N,3) torch tensor
        cam_param: dict with keys:
            'focal': (fx, fy)
            'princpt': (cx, cy)
            'R': (3,3)
            't': (3,)
        w2c: True -> world->camera, False -> camera->world
        device: 'cpu' or 'cuda'
    Returns:
        depth_map: (H,W) torch tensor
    """
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    H, W = int(cy*2), int(cx*2)

    if w2c:
        R_w2c = torch.tensor(cam_param["R"], dtype=torch.float32, device=device)
        t_w2c = torch.tensor(cam_param["t"], dtype=torch.float32, device=device)
    else:
        R_w2c = torch.tensor(cam_param["R"], dtype=torch.float32, device=device).T
        t_w2c = -R_w2c @ torch.tensor(cam_param["t"], dtype=torch.float32, device=device)

    K = torch.tensor([[fx,0,cx],
                      [0,fy,cy],
                      [0,0,1]], dtype=torch.float32, device=device)

    points_world = points_world.to(device)

    # world -> camera
    points_cam = (R_w2c @ points_world.T + t_w2c.reshape(3,1)).T  # (N,3)
    x, y, z = points_cam[:,0], points_cam[:,1], points_cam[:,2]

    # z>0
    mask = z > 0
    x, y, z = x[mask], y[mask], z[mask]

    # projection
    pts = torch.stack([x/z, y/z, torch.ones_like(z)], dim=0)  # (3,M)
    uv = K @ pts  # (3,M)
    u = torch.clamp(torch.round(uv[0,:]).long(), 0, W-1)
    v = torch.clamp(torch.round(uv[1,:]).long(), 0, H-1)

    # flatten indices
    depth_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    indices = v * W + u  # flatten

    # if multiple points map to same pixel, keep closest (z-buffer)
    z_buffer = torch.full((H*W,), float('inf'), device=device)
    z_buffer[indices] = torch.minimum(z_buffer[indices], z)
    depth_map = z_buffer.view(H, W)
    depth_map[depth_map == float('inf')] = 0  # reset empty pixels

    return depth_map

mean_in = depths.mean()
std_in = depths.std()
mean_tg = scene_depths.mean()
std_tg = scene_depths.std()
s_init = std_tg / (std_in + 1e-8)
t_init = mean_tg - s_init * mean_in

s = torch.tensor(s_init, requires_grad=True)
t = torch.tensor(t_init, requires_grad=True)

optimizer = optim.Adam([s, t], lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

pbar = trange(step_size, desc="Training", dynamic_ncols=True)
for step in pbar:
    optimizer.zero_grad()

    pred = depths * torch.abs(s) + t
    pred = torch.clamp(pred, min=1e-6)
    # loss = (((pred - scene_depths)**2)[~masks]).mean()
    loss = ((torch.abs(pred - scene_depths))[~masks]).mean()

    loss.backward()
    optimizer.step()
    scheduler.step()
    if step == 0:
        print("Initial Loss:", loss.item())

    pbar.set_postfix(loss=f"{loss.item():.6f}", scale=f"{s.item():.6f}", shift=f"{t.item():.6f}")

print('prediction (max/min):', pred.max(), pred.min())
print('pseudo depth (max/min):', depths.max(), depths.min())
print('gt depth (max/min):', scene_depths.max(), scene_depths.min())

# Visualization
save_path = os.path.join(root_path, 'depth_optimization_vis.png')
target_depth = scene_depths.detach().cpu().numpy()
optimized_depths = pred.detach().cpu().numpy()

num_to_show = min(4, optimized_depths.shape[0])
indices_to_show = np.linspace(0, optimized_depths.shape[0] - 1, num_to_show).astype(int).tolist()
print(indices_to_show)

fig, axes = plt.subplots(num_to_show, 3, figsize=(12, 3 * num_to_show))
vmin_depth = min(optimized_depths.min(), target_depth.min())
vmax_depth = max(optimized_depths.max(), target_depth.max())

for i, idx in enumerate(indices_to_show):
    pred = optimized_depths[idx]
    target = target_depth[idx]
    diff = np.abs(pred - target)

    ax0, ax1, ax2 = axes[i] if num_to_show > 1 else axes

    im0 = ax0.imshow(pred, cmap='jet', vmin=vmin_depth, vmax=vmax_depth)
    ax0.set_title(f'Predicted Depth {i}')
    ax0.axis('off')
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(target, cmap='jet', vmin=vmin_depth, vmax=vmax_depth)
    ax1.set_title('Target Depth')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    
    im2 = ax2.imshow(diff, cmap='inferno', vmin=0, vmax=diff.max())
    ax2.set_title('Abs Difference (Max: diff max)')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Save visualization in {save_path}")

optimized_depth_dir = os.path.join(root_path, 'depths_vggt_optimized')
os.makedirs(optimized_depth_dir, exist_ok=True)
optimized_depth_vis_dir = os.path.join(root_path, 'depths_vggt_optimized_vis')
os.makedirs(optimized_depth_vis_dir, exist_ok=True)

H, W = np.load(os.path.join(depth_path, '0.npy')).shape
for i in range(optimized_depths.shape[0]):
    # depth = cv2.resize(optimized_depths[i], (W, H))
    np.save(os.path.join(optimized_depth_dir, f'{i}.npy'), optimized_depths[i])
    optimized_depths_norm = (optimized_depths[i] - optimized_depths[i].min()) / (optimized_depths[i].max() - optimized_depths[i].min())
    optimized_depths_norm = optimized_depths_norm * 255
    imageio.imwrite(os.path.join(optimized_depth_vis_dir, f'{i}.png'), optimized_depths_norm.astype(np.uint8))

print(f"Save optimized_depth in {optimized_depth_dir}")
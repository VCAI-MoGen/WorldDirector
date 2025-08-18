import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt

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
mask_path = os.path.join(root_path, 'depths')
scene_info_path = os.path.join(root_path, 'scene_gs')
scene_depth_path = os.path.join(scene_info_path, 'depths_npy')

device = "cuda:0"

masks = torch.from_numpy(
    np.stack([
    np.load(os.path.join(mask_path, f'{i}_mask.npy')) for i in range(len(os.listdir(depth_path)))
])).to(device) # (N, H, W)

depths = torch.from_numpy(
    np.stack([
    cv2.resize(np.load(os.path.join(depth_path, f'{i}.npy')), (masks.shape[2], masks.shape[1])) for i in range(len(os.listdir(depth_path)))
])).to(device) # (N, H, W)

scene_depth = torch.from_numpy(
    np.stack([
    cv2.resize(np.load(os.path.join(scene_depth_path, f'{i}.npy'))[:, :, 0], (masks.shape[2], masks.shape[1])) for i in range(len(os.listdir(scene_depth_path)))
])).to(device) # (N, H, W)

print("Max scene depth:", scene_depth.max())
print("Min scene depth:", scene_depth.min())

s = torch.tensor(1.0, requires_grad=True)
t = torch.tensor(0.0, requires_grad=True)

optimizer = optim.Adam([s, t], lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

pbar = trange(step_size, desc="Training", dynamic_ncols=True)
for step in pbar:
    optimizer.zero_grad()

    pred = depths * s + t
    pred = torch.clamp(pred, min=1e-3)
    loss = (((pred - scene_depth)**2)[~masks]).mean()

    loss.backward()
    optimizer.step()
    scheduler.step()
    if step == 0:
        print("Initial Loss:", loss.item())

    pbar.set_postfix(loss=f"{loss.item():.6f}", scale=f"{s.item():.6f}", shift=f"{t.item():.6f}")


print('prediction (max/min):', pred.max(), pred.min())
print('pseudo depth (max/min):', depths.max(), depths.min())
print('gt depth (max/min):', scene_depth.max(), scene_depth.min())
pred_path = os.path.join(scene_info_path, 'optimized_depth.npy')
np.save(pred_path, pred.detach().cpu().numpy())
print(f"Save optimized depth in {pred_path}")


# Visualization
save_path = os.path.join(scene_info_path, 'visualization.png')
target_depth = scene_depth.detach().cpu().numpy()
optimized_depths = pred.detach().cpu().numpy()

if target_depth.shape[0] == 1:
    target_depth = target_depth[0]

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

H, W = np.load(os.path.join(depth_path, '0.npy')).shape
for i in range(optimized_depths.shape[0]):
    # depth = cv2.resize(optimized_depths[i], (W, H))
    np.save(os.path.join(optimized_depth_dir, f'{i}.npy'), optimized_depths[i])

print(f"Save optimized_depth in {optimized_depth_dir}")
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
import imageio
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--video_dir_path', type=str, dest='video_dir_path')
    parser.add_argument('--step', type=int, dest='step')
    parser.add_argument('--text_idx', type=str, dest='text_idx')
    parser.add_argument('--mask_threshold', type=float, dest='mask_threshold')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
video_dir_path = args.video_dir_path
mask_threshold = args.mask_threshold

# 예시 attention map (head, Q_d, K_d)
T, H, W = 81, 480, 720
step_idx = args.step
text_idices = eval(args.text_idx)
fps=16

root_path = os.path.join(args.root_path, f'{step_idx}')

attention_map_path = os.path.join(root_path, 'control_avg_attn_map.npy')
attention_map = np.array(attention_map_path)
video_path = os.path.join(video_dir_path, "src_video.mp4" )
mask_video_path = os.path.join(video_dir_path, "src_mask.mp4")
attn_vis_path = os.path.join(root_path, f'vis_with_frame_vace_{text_idices}')
os.makedirs(attn_vis_path, exist_ok=True)
attn_vis_video_path = os.path.join(root_path, f'vis_with_frame_vace_{text_idices}_norm.mp4')
mask_from_attn_vis_video_path = os.path.join(root_path, f'vis_mask_vace_{text_idices}_norm.mp4')

print(attention_map.shape)
"""
for i in tqdm(range(num_heads)):
    plt.figure(figsize=(5, 5))
    plt.imshow(attention_map[i], cmap='viridis', aspect='auto')
    plt.title(f'Head {i}')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.colorbar()
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'attention_head_{i}.png')
    plt.savefig(filename)
    plt.close()  # 메모리 절약 위해 figure 닫기
"""
mean_attn = attention_map.mean(axis=0)
print(mean_attn.shape)
plt.figure(figsize=(5, 5))
plt.imshow(mean_attn, cmap='viridis', aspect='auto')
plt.title(f'Mean')
plt.xlabel('Key positions')
plt.ylabel('Query positions')
plt.colorbar()
plt.tight_layout()

filename = os.path.join(root_path, f'mean_attention_vace.png')
plt.savefig(filename)
plt.close()  # 메모리 절약 위해 figure 닫기

print(np.sum(mean_attn))
text_aligned_attn = torch.tensor(mean_attn[:, text_idices]).mean(dim=1)

reshaped_attn = text_aligned_attn.reshape(1 + T // 4, H // 16, W // 16)
attn_up = F.interpolate(
    reshaped_attn.unsqueeze(0).unsqueeze(0),
    size=(T, H, W),
    mode='trilinear',  # 3D linear interpolation
    align_corners=False
)[0, 0].cpu().numpy()

cap = cv2.VideoCapture(video_path)
mask_cap = cv2.VideoCapture(mask_video_path)
frame_idx = 0
writer = imageio.get_writer(attn_vis_video_path, fps=fps)
mask_writer = imageio.get_writer(mask_from_attn_vis_video_path, fps=fps)

while True:
    ret, frame = cap.read()
    ret_mask, frame_mask = mask_cap.read()
    if not ret or frame_idx >= attn_up.shape[0]:
        break

    # 원본 프레임 크기 맞추기
    frame_resized = cv2.resize(frame, (W, H))

    # 해당 프레임의 attention map
    attn_frame = attn_up[frame_idx]

    attn_frame[frame_mask[:, :, 0] < 127] = 0

    # Normalize (0~255)
    attn_norm = cv2.normalize(attn_frame, None, 0, 255, cv2.NORM_MINMAX)
    attn_norm = attn_norm.astype(np.uint8)

    # Heatmap 만들기
    heatmap = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(frame_resized, 0.6, heatmap, 0.4, 0)

    mask_from_attn = (attn_norm > int(mask_threshold * 255)).astype(np.uint8) * 255

    # 저장
    save_path = os.path.join(attn_vis_path, f"{frame_idx:03d}.jpg")
    cv2.imwrite(save_path, overlay)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    writer.append_data(overlay_rgb)
    mask_writer.append_data(mask_from_attn)
    frame_idx += 1

writer.close()
mask_writer.close()
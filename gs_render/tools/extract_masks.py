import argparse
import os
import os.path as osp
from PIL import Image
from lang_sam import LangSAM
import cv2
import numpy as np
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
batch_size = args.batch_size

sam_model = LangSAM()

def mask_to_bbox_mask(mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 3 and mask.shape[2] == 1

    mask_2d = mask[:, :, 0]
    
    if np.sum(mask_2d) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    y_indices, x_indices = np.where(mask_2d > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    bbox_mask_2d = np.zeros_like(mask_2d, dtype=np.uint8)
    bbox_mask_2d[y_min:y_max+1, x_min:x_max+1] = 1

    bbox_mask = bbox_mask_2d[:, :, np.newaxis]
    return bbox_mask

def expand_bbox_mask_width(mask, pad_ratio=0.1):
    """
    mask: (H, W, 3) ndarray of 0 and 1
    pad: number of pixels to expand on left and right
    """
    H, W = mask.shape
    expanded_mask = np.zeros_like(mask)

    # 각 row별로 1이 있는 좌우 인덱스를 찾아서 확장
    for y in range(H):
        indices = np.where(mask[y] == 1)[0]
        if len(indices) == 0:
            continue
        mask_width = indices[-1] - indices[0]
        pad = int(mask_width * pad_ratio)
        left = max(indices[0] - pad, 0)
        right = min(indices[-1] + pad + 1, W)
        if left >= right:
            continue
        expanded_mask[y, left:right] = 1

    return expanded_mask

frame_path = osp.join(root_path, 'frames')
frame_list = sorted(os.listdir(frame_path))
image_names = [osp.join(frame_path, frame) for frame in frame_list]


image_pil_list = [Image.open(image_name).convert("RGB") for image_name in image_names]
W, H = image_pil_list[0].size
human_masks = []
bb_masks = []
with torch.no_grad():
    for i in range(0, len(image_pil_list) + 1, batch_size):
        start = i
        end = i + batch_size
        if end >= len(image_pil_list):
            end = len(image_pil_list)
        target_image_list = image_pil_list[start:end]
        results = sam_model.predict(target_image_list, ['human.'] * len(target_image_list))
        human_masks += [result['masks'].transpose(1, 2, 0)[:, :, 0:1] if 'masks' in result else np.zeros((H, W, 1)) for result in results] # (B, H, W, 1)
bb_masks = [mask_to_bbox_mask(mask) for mask in human_masks] # (B, H, W, 1)

mask_path = osp.join(root_path, 'masks')
os.makedirs(mask_path, exist_ok=True)
bb_mask_path = osp.join(root_path, 'bb_masks')
os.makedirs(bb_mask_path, exist_ok=True)

fps = 30
h, w = human_masks[0].shape[:2]

writer = cv2.VideoWriter(f"{osp.join(root_path, 'mask.mp4')}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i, human_mask in enumerate(human_masks):
    human_mask = np.any(human_mask != 0, axis=2).astype(np.uint8)[:, :, None] * 255
    human_mask = human_mask.repeat(3, axis=2)
    cv2.imwrite(osp.join(mask_path, str(frame_list[i])), human_mask)
    writer.write(human_mask)
writer.release()

writer = cv2.VideoWriter(f"{osp.join(root_path, 'bb_mask.mp4')}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i, bb_mask in enumerate(bb_masks):

    # Expanding mask in width axis
    expanded_mask = expand_bbox_mask_width(bb_mask[:, :, 0], pad_ratio=0)

    bb_mask = np.stack([expanded_mask] * 3, axis=-1)
    bb_mask = np.any(bb_mask != 0, axis=2).astype(np.uint8)[:, :, None] * 255
    bb_mask = bb_mask.repeat(3, axis=2)
    
    cv2.imwrite(osp.join(bb_mask_path, str(frame_list[i])), bb_mask)
    writer.write(bb_mask)
writer.release()

torch.cuda.empty_cache()
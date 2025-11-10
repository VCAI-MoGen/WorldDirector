import argparse
import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import torch
import imageio.v3 as iio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--conf_thresh', type=float, dest='conf_thresh', default=0.01)
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
conf_thresh = args.conf_thresh
scene_render_dir_path = os.path.join(root_path, 'scene_renders')
# if os.path.exists(os.path.join(root_path, 'scene_renders_enhanced')):
#     scene_render_dir_path = os.path.join(root_path, 'scene_renders_enhanced')
confidence_dir_path = os.path.join(root_path, 'scene_confidence')

scene_render_path_list = [os.path.join(scene_render_dir_path, f'{i}.png') for i in range(len(os.listdir(scene_render_dir_path)))]
confidence_path_list = [os.path.join(confidence_dir_path, f'{i}.png.npy') for i in range(len(os.listdir(confidence_dir_path)))]

frames_list = []
masks_list = []
for i in tqdm(range(len(scene_render_path_list))):
    scene_render_path = scene_render_path_list[i]
    scene_render = iio.imread(scene_render_path)
    confidence_path = confidence_path_list[i]
    confidence = np.load(confidence_path)
    
    mask = (confidence < conf_thresh).repeat(3, axis=-1)
    scene_render[mask] = 127
    mask = mask.astype(np.uint8) * 255
    frames_list.append(scene_render)
    masks_list.append(mask)

iio.imwrite(os.path.join(root_path, 'confidence_masked_video.mp4'), frames_list, fps=16)
iio.imwrite(os.path.join(root_path, 'confidence_mask_video.mp4'), masks_list, fps=16)
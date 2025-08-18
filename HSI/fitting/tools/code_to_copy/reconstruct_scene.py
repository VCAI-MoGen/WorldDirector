import os
import os.path as osp
from glob import glob
import sys
import argparse
import cv2
import matplotlib
import numpy as np
import trimesh
from visual_util import apply_scene_alignment, integrate_camera_into_scene
from scipy.ndimage import binary_dilation
from tqdm import tqdm

def voxel_grid_filter(points, voxel_size):
    """
    Voxel grid 필터링: 너무 가까운 포인트를 병합하고 하나만 남긴다.
    
    Args:
        points (np.ndarray): (N, 3) 포인트 클라우드
        voxel_size (float): 포인트 간 최소 거리 기준 (예: 0.01)

    Returns:
        np.ndarray: 필터링된 (M, 3) 포인트
    """
    # 정규화된 voxel 좌표 (정수로 바뀜)
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # 고유 voxel만 남기고, 해당 인덱스만 취함 (첫 번째 포인트 유지)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    
    # 해당 포인트만 반환
    return unique_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--remove_human', dest='remove_human', action='store_true')
    parser.add_argument('--mask_object', dest='mask_object', action='store_true')
    parser.add_argument('--get_object', dest='get_object', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
remove_human = args.remove_human
mask_object = args.mask_object
get_object = args.get_object
frame_list = sorted([int(x.split('/')[-1][:-4]) for x in os.listdir(osp.join(root_path, 'frames'))])

scene_info_path = osp.join(root_path, 'scene_info')

# Get images from predictions
images = np.load(osp.join(scene_info_path, 'images.npy'))

N, C, H, W = images.shape
assert N == len(frame_list)

# Use extrinsic matrices instead of pred_extrinsic_list
camera_matrices = np.load(osp.join(scene_info_path, 'extrinsic.npy'))

pred_world_points = np.load(osp.join(scene_info_path, 'world_points.npy'))
pred_world_points_conf = np.load(osp.join(scene_info_path, 'world_points_conf.npy'))

conf = pred_world_points_conf.reshape(-1)
# Convert percentage threshold to actual confidence value
conf_thres = 1e-4
if conf_thres == 0.0:
    conf_threshold = 0.0
else:
    conf_threshold = np.percentile(conf, conf_thres)

conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

# Remove human point clouds
radius = 10
kernel = np.ones((2*radius + 1, 2*radius + 1), dtype=np.uint8)

# Remove Human point clouds
if remove_human:
    for i, frame_idx in enumerate(frame_list):
        mask_path = osp.join(root_path, 'depths')
        human_mask = cv2.resize(np.load(osp.join(mask_path, f'{frame_idx}_mask.npy')).astype(int), (W, H), interpolation=cv2.INTER_NEAREST)
        human_mask = binary_dilation(human_mask, structure=kernel).astype(bool)
        pred_world_points[i][human_mask] = 0

if get_object:
    for i, frame_idx in enumerate(frame_list):
        mask_path = osp.join(root_path, 'object_segments', 'masks')
        object_mask = cv2.resize(np.load(osp.join(mask_path, f'{frame_idx}.npy')).astype(int), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred_world_points[i][~object_mask] = 0

vertices_3d = pred_world_points.reshape(-1, 3)
# Handle different image formats - check if images need transposing
if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
    colors_rgb = np.transpose(images, (0, 2, 3, 1))
else:  # Assume already in NHWC format
    colors_rgb = images

if mask_object:
    for i, frame_idx in enumerate(frame_list):
        mask_path = osp.join(root_path, 'object_segments', 'masks')
        object_mask = cv2.resize(np.load(osp.join(mask_path, f'{frame_idx}.npy')).astype(int), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        colors_rgb[i][object_mask] = (0.5, 0.25, 0.5)
       
colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

vertices_3d = vertices_3d[conf_mask]
colors_rgb = colors_rgb[conf_mask]

if remove_human or get_object:
    zero_mask = np.all(vertices_3d == 0, axis=1)
    vertices_3d = vertices_3d[~zero_mask]
    colors_rgb = colors_rgb[~zero_mask]

norm_mask = voxel_grid_filter(vertices_3d, 0.005)
vertices_3d = vertices_3d[norm_mask]
colors_rgb = colors_rgb[norm_mask]

if vertices_3d is None or np.asarray(vertices_3d).size == 0:
    vertices_3d = np.array([[1, 0, 0]])
    colors_rgb = np.array([[255, 255, 255]])

scene = trimesh.points.PointCloud(vertices=vertices_3d, colors=colors_rgb)

if mask_object:
    scene_file_name = 'scene_mask_object.ply'
elif get_object:
    scene_file_name = 'object.ply'
else:
    scene_file_name = 'scene.ply'

scene.export(os.path.join(root_path, 'scene_info', scene_file_name))
print(f"Scene saved in {os.path.join(root_path, 'scene_info', scene_file_name)}")
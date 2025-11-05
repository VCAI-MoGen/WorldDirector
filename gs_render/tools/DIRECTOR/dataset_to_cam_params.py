import os
import shutil
import argparse
from evo.tools.file_interface import read_kitti_poses_file
import torch
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--npy_path', type=str, dest='npy_path')
    args = parser.parse_args()
    return args

def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            orth = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            v = np.cross(a, orth)
            v /= np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            return np.eye(3) + 2 * K @ K
    else:
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
        return R

trajectory_dir_path = "/home/ckd248/data/cam/SCVideo/DIRECTOR/et-data/traj"
intrinsics_dir = "/home/ckd248/data/SCVideo/gs_render/tools/DIRECTOR/et-data/intrinsics"
character_dir = "/home/ckd248/data/SCVideo/gs_render/tools/DIRECTOR/et-data/char_raw"

args = parse_args()
root_path = args.root_path
data_name = args.data_name
npy_path = args.npy_path
print(npy_path)
cam_params_path = os.path.join(root_path, 'detailed', 'cam_params_scene')
shutil.rmtree(cam_params_path)
os.mkdir(cam_params_path)

if npy_path is None:
    trajectory_path = os.path.join(trajectory_dir_path, data_name + '.txt')
    trajectory = read_kitti_poses_file(trajectory_path)
    matrix_trajectory = np.array(trajectory.poses_se3)
else:
    matrix_trajectory = np.load(npy_path)

intrinsics_filename = data_name + ".npy"
intrinsics_path = os.path.join(intrinsics_dir, intrinsics_filename)
intrinsics = np.load(intrinsics_path)
R_scene_up_to_director_up = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))

character_filename = data_name + ".npy"
character_path = os.path.join(character_dir, character_filename)
char_traj = np.load(character_path)
print(char_traj.shape)
print(matrix_trajectory.shape)

for i, traj in enumerate(matrix_trajectory):
    R_c2w = R_scene_up_to_director_up.T @ traj[:3, :3]
    t_c2w = R_scene_up_to_director_up.T @ traj[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    cam_param = {
        "R": R_w2c.tolist(),
        "t": t_w2c.tolist(),
        "focal": [312.7740969872587, 312.7740969872587], 
        "princpt": [360, 240]
    }
    with open(os.path.join(cam_params_path, f'{i}.json'), 'w') as f:
        json.dump(cam_param, f)
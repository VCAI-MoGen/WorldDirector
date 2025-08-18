import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
smplx_params_dir = os.path.join(root_path, 'smplx_optimized', 'traj')
smplx_params_path_list = [os.path.join(smplx_params_dir, f'{i}.json') for i in range(len(os.listdir(smplx_params_dir)))]
scene_gs_path = os.path.join(root_path, 'scene_gs')
scene_cam_param_path = os.path.join(scene_gs_path, 'cam_param.json')
with open(scene_cam_param_path, 'r') as f:
    scene_cam_param = json.load(f)
R = np.array(scene_cam_param['R'])#  @ np.diag((1, 1, -1))#  @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
t = np.array(scene_cam_param['t'])

traj_cam = []
for smplx_params_path in smplx_params_path_list:
    with open(smplx_params_path, 'r') as f:
        smplx_params = json.load(f)
    traj_cam.append(smplx_params['traj'])

traj_cam = np.array(traj_cam)

B = traj_cam.shape[0]
traj_world = traj_cam @ R.T + t # (N, 3)
traj_world = traj_world

save_path = os.path.join(scene_gs_path, 'traj.npy')
np.save(save_path, traj_world)
print(f"Save human trajectory in {save_path}")

def visualize_trajectory(trajectory: np.ndarray, save_path: str = 'trajectory.png'):
    """
    시점이 camera coordinate인 3D trajectory를 이미지로 저장합니다.

    Args:
        trajectory (np.ndarray): (N, 3) shape의 trajectory. 각 행은 (x, y, z)
        save_path (str): 저장할 이미지 경로
    """
    assert trajectory.shape[1] == 3, "trajectory shape must be (N, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # trajectory 선 그리기
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue', linewidth=2, label='Trajectory')

    # 시작점, 끝점 강조
    ax.scatter(*trajectory[0], color='green', s=50, label='Start')
    ax.scatter(*trajectory[-1], color='red', s=50, label='End')

    # 카메라 좌표계 축 표시
    axis_len = 0.2
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', label='Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Human Trajectory in Camera Coordinates')
    ax.legend()

    # 보기 좋게 설정
    ax.view_init(elev=30, azim=120)
    ax.grid(True)
    ax.set_box_aspect([1,1,1])  # equal aspect ratio

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved trajectory image to {save_path}")

def plot_xyz_axis_changes(trajectory: np.ndarray, save_path='xyz_change.png'):
    """
    x, y, z 값의 시간에 따른 변화를 한 그래프에 시각화합니다.
    """
    assert trajectory.shape[1] == 3, "trajectory must be of shape (N, 3)"

    frames = np.arange(trajectory.shape[0])
    x_values = trajectory[:, 0]
    y_values = trajectory[:, 1]
    z_values = trajectory[:, 2]

    plt.figure(figsize=(8, 4))
    plt.plot(frames, x_values, label='X (left-right)', color='red')
    plt.plot(frames, y_values, label='Y (up-down)', color='green')
    plt.plot(frames, z_values, label='Z (forward-back)', color='blue')

    plt.xlabel('Frame Index')
    plt.ylabel('Coordinate Value')
    plt.title('X/Y/Z Axis Change Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved xyz-axis plot to {save_path}")

visualize_trajectory(traj_world)
plot_xyz_axis_changes(traj_world)
print(traj_cam)
print(traj_world)
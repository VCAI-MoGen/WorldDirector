import viser
import argparse
import numpy as np
import open3d as o3d
from cameras import Camera, Cameras
from scipy.spatial.transform import Rotation
import torch
import imageio
import os
import time
from tqdm import tqdm
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="0000", help="Name of the video.")
args = parser.parse_args()

data_name = args.data
dataset_path = '/home/ckd248/data/SCVideo/camera_generation/dataset'
camera_dataset_path = os.path.join(dataset_path, 'cam')
scene_dataset_path = os.path.join(dataset_path, 'scene')

scene_data_path = os.path.join(scene_dataset_path, f'{data_name}.ply')
camera_data_path = os.path.join(camera_dataset_path, f'{data_name}.npz')



# 1) viser 서버 실행
server = viser.ViserServer(port=8085)

# 2) PLY 로드 (trimesh로 만든 PLY도 완전하게 로드됨)
pcd = o3d.io.read_point_cloud(scene_data_path)
points = np.asarray(pcd.points)

# color 있으면 사용, 없으면 흰색
if pcd.colors:
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
else:
    colors = np.ones((len(points), 3), dtype=np.uint8) * 255

# 3) viser에 point cloud 추가
server.add_point_cloud(
    "/pointcloud",
    points=points,
    colors=colors,
    point_size=0.01,   # 점 크기
)

def build_cameras(extrinsics, intrinsics):
    """
    Input:
        extrinsics: [N, 4, 4] numpy array
        intrinsics: [N, 3, 3] numpy array
    """
    N = extrinsics.shape[0]
    R_w2c = extrinsics[:, :3, :3]
    T_w2c = extrinsics[:, :3, 3:4]

    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    T_c2w = (- R_c2w @ T_w2c).squeeze(-1)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    width = intrinsics[:, 0, 2] * 2
    height = intrinsics[:, 1, 2] * 2
    # width = np.array([432] * N)
    # height = np.array([240] * N)
    appearance_id = torch.zeros((N), dtype=torch.int)
    normalized_appearance_id = torch.zeros((N), dtype=torch.int)
    distortion_params = torch.zeros((N, 4), dtype=torch.int)
    camera_type = torch.zeros((N), dtype=torch.int)

    cameras = Cameras(
        R=torch.from_numpy(R_c2w).float(),
        T=torch.from_numpy(T_c2w).float(),
        fx=torch.from_numpy(fx).float(),
        fy=torch.from_numpy(fy).float(),
        cx=torch.from_numpy(cx).float(),
        cy=torch.from_numpy(cy).float(),
        width=torch.from_numpy(width).int(),
        height=torch.from_numpy(height).int(),
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=distortion_params,
        camera_type=camera_type,
    )
    return cameras

def render(client, cameras: Cameras):
    images = []
    print("Rendering...")
    for camera in tqdm(cameras):
        xyzw = Rotation.from_matrix(camera.R.numpy()).as_quat().tolist()
        img = client.get_render(
            height=camera.height.item(),
            width=camera.width.item(),
            wxyz=np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]),
            position=camera.T.numpy(),
            fov=camera.fov_y.item(),
        )
        images.append(img)

    save_path = f"{data_name}.mp4"
    imageio.mimsave(save_path, images, fps=12)
    print("Rendering done. Saved to", save_path)

with server.gui.add_folder("Viz"):
    render_by_cam_button = server.gui.add_button(
                "Render by Cam",
                color="purple",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
    )
    @render_by_cam_button.on_click
    def _(event: viser.GuiEvent) -> None:
        def bg():
            try:
                cam_params = np.load(camera_data_path, allow_pickle=True)
                w2c_ext = cam_params['extrinsics']
                intrinsics = cam_params['intrinsics']
                cameras = build_cameras(w2c_ext, intrinsics)
                render(event.client, cameras)
            except Exception as e:
                print("Error:", e)

        threading.Thread(target=bg, daemon=True).start()
    
    render_by_pred_cam_button = server.gui.add_button(
                "Render by Pred Cam",
                color="Blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
    )
    @render_by_pred_cam_button.on_click
    def _(event: viser.GuiEvent) -> None:
        def bg():
            try:
                cam_params = np.load(f"{data_name}.npz", allow_pickle=True)
                w2c_ext = cam_params['extrinsics']
                intrinsics = cam_params['intrinsics']
                cameras = build_cameras(w2c_ext, intrinsics)
                render(event.client, cameras)
            except Exception as e:
                print("Error:", e)

        threading.Thread(target=bg, daemon=True).start()

    
while True:
    time.sleep(1)
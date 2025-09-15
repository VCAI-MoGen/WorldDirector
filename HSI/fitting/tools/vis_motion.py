import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

# 기본 설정
point_cloud_path = os.path.join(root_path, "scene_info", "scene.ply")
mesh_dir = os.path.join(root_path, "smplx_optimized", "meshes_smoothed")
cam_params_dir = os.path.join(root_path, 'cam_params')
frame_list = sorted([int(x.split('/')[-1][:-4]) for x in os.listdir(mesh_dir)])

output_frames_dir = os.path.join(root_path, "smplx_optimized", "renders_scene")
video_output_path = os.path.join(root_path, "smplx_optimized_smoothed_3d.mp4")
fps = 30
width, height = 1280, 720

# 출력 프레임 저장 폴더 생성
os.makedirs(output_frames_dir, exist_ok=True)

# 포인트 클라우드 불러오기
pcd = o3d.io.read_point_cloud(point_cloud_path)
pcd.estimate_normals()
pcd_material = o3d.visualization.rendering.MaterialRecord()
pcd_material.shader = "defaultUnlit"
pcd_material.point_size = 3.0

mesh_material = o3d.visualization.rendering.MaterialRecord()
mesh_material.shader = "defaultLit"

# 렌더링 창 설정
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background([1, 1, 1, 1])

# 조명 설정
scene.scene.set_indirect_light_intensity(15000)
scene.scene.enable_sun_light(True)

scene.add_geometry("scene", pcd, pcd_material)

for i in tqdm(frame_list):
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, f'{i}.ply'))
    mesh.compute_vertex_normals()

    with open(os.path.join(cam_params_dir, f'{i}.json'), 'r') as f:
        cam_params = json.load(f)
    
    R = np.array(cam_params['R'])
    t = np.array(cam_params['t'])
    
    transform = np.eye(4)
    transform[:3, 3] = -t
    mesh.transform(transform)

    transform = np.eye(4)
    transform[:3, :3] = R.T
    mesh.transform(transform)

    # mesh 추가
    scene.add_geometry("human", mesh, mesh_material)

    # 카메라 설정
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    eye = center - [0, 0, extent * 0.9]  # 카메라 위치 조정
    up = [0, -1, 0]
    
    scene.camera.look_at(center, eye, up)

    # 프레임 저장
    img = renderer.render_to_image()
    image_path = os.path.join(output_frames_dir, f"{i}.png")
    o3d.io.write_image(image_path, img)

    # 이전 mesh 제거
    scene.remove_geometry("human")

# 시각화 종료
print(f"Frames are saved in {output_frames_dir}")

# 이미지 확장자에 따라 패턴 지정 (예: png, jpg 등)
image_paths = [os.path.join(output_frames_dir, f'{i}.png') for i in frame_list] # 또는 '*.jpg'

# 첫 번째 이미지로 프레임 사이즈 결정
frame = cv2.imread(image_paths[0])
height, width, _ = frame.shape
frame_size = (width, height)

# 비디오 코덱 설정 (MP4: 'mp4v', AVI: 'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi
fps = 30  # 초당 프레임 수

# 비디오 라이터 초기화
out = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)

# 이미지들을 순서대로 비디오에 추가
for image_path in image_paths:
    img = cv2.imread(image_path)
    out.write(img)

# 종료
out.release()
print(f"Video is saved in {video_output_path}")
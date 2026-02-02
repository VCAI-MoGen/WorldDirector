import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from glob import glob
import json
import math
import trimesh
from PIL import Image
from tqdm import tqdm
from core_pkg.common.utils.camera_utils import get_cameras_from_json
from core_pkg.common.utils.data_utils import normalize_camera_extrinsics_and_points


def load_ply_as_pointcloud(path):
    mesh = trimesh.load(path)
    points = mesh.vertices
    if hasattr(mesh, "vertex_colors"):
        colors = mesh.vertex_colors[:, :3]
    else:
        colors = np.ones_like(points) * 255
    pc = np.hstack([points, colors])
    return pc.astype(np.float32)

class CamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_path = self.cfg.dataset_dir

        self.data_names, self.extrinsics_list, self.intrinsics_list, self.scene_paths, self.text_prompts, self.frames_list = self.load_data()

    def load_data(self):
        if os.path.basename(self.dataset_path)[-1] == 'K':
            data_list = os.listdir(self.dataset_path)
        else:
            data_list = []
            for data_part in os.listdir(self.dataset_path):
                data_list += [data_part + '/' + data_name for data_name in os.listdir(os.path.join(self.dataset_path, data_part))]

        data_names = []
        scene_paths = []
        text_prompts = []
        frames_list = []
        extrinsics_list = []
        intrinsics_list = []

        print("Loading data...")
        for idx, data_name in tqdm(enumerate(sorted(data_list))):

            # Cut
            # if idx > 3:
            #     break
            # Experiments for linear interpolation
            # if data_name not in ['1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed', '006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed']:
            # # # if data_name not in ['2K/75fa1a43d70b7fbc2d58bff7122bf57b47af4d74855694aca46eb8a3fbf5d51b']:
            #     continue

            
            data_path = osp.join(self.dataset_path, data_name)

            # Camera & Scene processing
            camera_path = osp.join(data_path, 'cameras.json')
            cameras = get_cameras_from_json(camera_path) # w2c
            cameras._calculate_w2c()
            cameras._calculate_K()
            extrinsics = cameras.world_to_camera.transpose(-1, -2)[:self.cfg.num_frames] # (N_cam, 4, 4), transpose is needed
            intrinsics = cameras.K[:self.cfg.num_frames] # (N_cam, 3, 3)
            
            if extrinsics.shape[0] < self.cfg.num_frames:
                continue

            scene_path = osp.join(data_path, 'scene.pt')

            # Text processing
            text_path = osp.join(data_path, 'prompt.json')         
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    # text_prompt = json.load(f)['prompt']
                    text_prompt = json.load(f)['prompt_camera']
            else:
                text_prompt = " "

            # Video processing
            """
            frames_path = osp.join(colmap_path, 'images')
            frames = []
            for frame_name in sorted(os.listdir(frames_path)):
                frame_path = osp.join(frames_path, frame_name)
                img = Image.open(frame_path).convert('RGB')
                frames.append(img)
            frames_list.append(frames)
            """

            data_names.append(data_name)
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)
            scene_paths.append(scene_path)
            text_prompts.append(text_prompt)
        
        return data_names, extrinsics_list, intrinsics_list, scene_paths, text_prompts, frames_list
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        extrinsics = self.extrinsics_list[idx]
        intrinsics = self.intrinsics_list[idx]
        scene_path = self.scene_paths[idx]
        text_prompt = self.text_prompts[idx]

        scene_data = torch.load(scene_path) # (N, 6), coordinates (3) + color (3)
        normalized_extrinsics, normalized_scene_pc, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, scene_data[:, :3])

        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        width = intrinsics[:, 0, 2] * 2
        height = intrinsics[:, 1, 2] * 2
        normalized_fx = (fx / width)[:, None] # (N, 1)
        normalized_fy = (fy / height)[:, None] # (N, 1)

        normalized_intrinsics = np.concatenate([normalized_fx, normalized_fy], axis=-1).astype(np.float32)

        cam_param = np.concatenate([normalized_extrinsics[:, :3, 0], normalized_extrinsics[:, :3, 1], normalized_extrinsics[:, :3, 3], normalized_intrinsics], axis=-1).astype(np.float32) # (N_cam, 11)
        # cam_param = np.concatenate([normalized_extrinsics[:, :3, 0], normalized_extrinsics[:, :3, 1], normalized_extrinsics[:, :3, 3]], axis=-1).astype(np.float32) # (N_cam, 9)

        first_extrinsic = extrinsics[0]

        scene_data[:, :3] = normalized_scene_pc
        scene_data = scene_data[mask]

        data = {
            'data_name': data_name,
            'cam_param': cam_param,
            'first_extrinsic': first_extrinsic,
            'width': width,
            'height': height,
            'avg_scale': avg_scale,
            'pc': scene_data,
            'text_prompt': text_prompt
        }
        
        return data
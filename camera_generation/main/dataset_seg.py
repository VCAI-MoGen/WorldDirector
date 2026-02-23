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
    def __init__(self, cfg, type='train'):
        self.cfg = cfg
        self.type = type
        if self.type == 'train':
            self.dataset_path = self.cfg.train_dataset_dir
        if self.type == 'val':
            self.dataset_path = self.cfg.val_dataset_dir

        self.data_names, self.extrinsics_list, self.intrinsics_list, self.scene_paths, self.text_prompts, self.frames_list = self.load_data()

    def load_data(self):
        if os.path.basename(self.dataset_path)[-1] == 'K':
            data_list = os.listdir(self.dataset_path)
        else:
            data_list = []
            for data_part in os.listdir(self.dataset_path):
                if data_part[-1] != 'K':
                    continue
                data_list += [data_part + '/' + data_name for data_name in os.listdir(os.path.join(self.dataset_path, data_part))]

        data_names = []
        scene_paths = []
        text_prompts = []
        frames_list = []
        extrinsics_list = []
        intrinsics_list = []
        idx_list = []

        print("Loading data...")
        for idx, data_name in tqdm(enumerate(sorted(data_list))):

            if self.type == 'val':
                if idx > 250:
                    break

            # Cut
            # if idx > 127:
            #     break
            # # Experiments for linear interpolation
            # if data_name not in ['1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed', '006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed']:
            #     continue
            # if data_name not in ['006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed', '00f32fbf46f60a718942a417962d42d998317725fb59248ed728ee69ac3a3e8f']:
            #     continue
            
            data_path = osp.join(self.dataset_path, data_name)

            # Camera & Scene processing
            camera_path = osp.join(data_path, 'cameras.json')
            cameras = get_cameras_from_json(camera_path) # w2c
            cameras._calculate_w2c()
            cameras._calculate_K()
            extrinsics = cameras.world_to_camera.transpose(-1, -2) # (N_cam, 4, 4), transpose is needed
            intrinsics = cameras.K # (N_cam, 3, 3)
            
            if extrinsics.shape[0] < self.cfg.num_frames:
                continue

            scene_path = osp.join(data_path, 'scene.pt')

            # Text processing
            text_path = osp.join(data_path, 'prompts.json')         
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    # text_prompt = json.load(f)['prompt']
                    text_prompt = json.load(f)
            else:
                text_prompt = " "
            for k, v in text_prompt.items():
                data_names.append(data_name.replace('/', '_') + '_' + k)
                frame_idx = v['frame_idx']
                prompt = v['prompt_camera_with_scene_video']['concise']
                text_prompts.append(prompt)
                extrinsics_list.append(extrinsics[frame_idx[0]:frame_idx[1]])
                intrinsics_list.append(intrinsics[frame_idx[0]:frame_idx[1]])
                scene_paths.append(scene_path)
        
        return data_names, extrinsics_list, intrinsics_list, scene_paths, text_prompts, frames_list
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        extrinsics = self.extrinsics_list[idx]
        intrinsics = self.intrinsics_list[idx]
        text_prompt = self.text_prompts[idx]

        # if self.cfg.load_points:
        #     scene_data = torch.load(scene_path) # (N, 6), coordinates (3) + color (3)
        #     normalized_extrinsics, normalized_scene_pc, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, scene_data[:, :3])

        #     pc_embeds_path = os.path.join(os.path.dirname(scene_path), 'pc_embeds_decoded.pt')
        #     if self.cfg.load_saved_pc_embeds and os.path.exists(pc_embeds_path):
        #         scene_embeds = torch.load(pc_embeds_path, map_location='cpu')
        #         scene_data = None
        #     else:
        #         scene_data[:, :3] = normalized_scene_pc
        #         scene_data = scene_data[mask]
        # else:
        #     saved_avg_scale_path = os.path.join(os.path.dirname(scene_path), "point_avg_scale.json")
        #     saved_avg_scale = json.load(open(saved_avg_scale_path, "r"))
        #     normalized_extrinsics, _, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, avg_scale=saved_avg_scale)

        data = {
            'data_name': data_name,
            'cam_param': extrinsics,
            'intrinsics': intrinsics,
            'text_prompt': text_prompt
        }
        return data
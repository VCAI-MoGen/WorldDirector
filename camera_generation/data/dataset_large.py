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
import concerto
import random

def load_ply_as_pointcloud(path):
    mesh = trimesh.load(path)
    points = mesh.vertices
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0  # (N, 4) - RGBA uint8
    else:
        colors = np.ones_like(points)
    colors = np.clip(colors, 0.0, 1.0)
    pc = np.hstack([points, colors])
    return pc.astype(np.float32)

class CamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir_list = self.cfg.dataset_dir_list

        if self.cfg.point_encoder == 'concerto':
            pc_transform_config = [
                dict(type="CenterShift", apply_z=True),
                dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal"]}),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "inverse", "offset"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ]
        elif self.cfg.point_encoder == 'custom':
            pc_transform_config = [
                dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color"]}),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "inverse", "offset"),
                    feat_keys=("coord", "color"),
                ),
            ]
        self.pc_transform = concerto.transform.Compose(pc_transform_config)

        self.data_names, self.extrinsics_list, self.intrinsics_list, self.scene_paths, self.text_prompts, = self.load_data()
        self.sample()

    def load_data(self):
        data_names = []
        scene_paths = []
        text_prompts = []
        frames_list = []
        extrinsics_list = []
        intrinsics_list = []

        print("Loading data...")
        dataset_cnt_dist = []
        for dataset_dir_path in tqdm(sorted(self.dataset_dir_list)):
            
            data_name_list = os.listdir(dataset_dir_path)
            dataset_name = os.path.basename(dataset_dir_path)
            dataset_cnt = 0
            for data_name in data_name_list:
                data_path = os.path.join(dataset_dir_path, data_name)

                # Camera & Scene processing
                camera_path = osp.join(data_path, 'cameras.json')
                cameras = get_cameras_from_json(camera_path) # w2c
                cameras._calculate_w2c()
                cameras._calculate_K()
                extrinsics = cameras.world_to_camera.transpose(-1, -2) # (N_cam, 4, 4), transpose is needed
                intrinsics = cameras.K # (N_cam, 4, 4)
                
                if extrinsics.shape[0] < self.cfg.num_frames:
                    continue
                if torch.isnan(extrinsics).any() or torch.isnan(intrinsics).any():
                    print(data_name)

                scene_path = osp.join(data_path, 'scene.pt')
                if os.path.exists(scene_path) is False:
                    scene_path = osp.join(data_path, 'scene.ply')
                    if os.path.exists(scene_path) is False:
                        scene_path = None

                # Text processing
                text_path = osp.join(data_path, 'prompts.json')         
                if os.path.exists(text_path):
                    with open(text_path, 'r') as f:
                        text_prompt = json.load(f)
                else:
                    text_prompt = " "
                for k, v in text_prompt.items():
                    frame_idx = v['frame_idx']
                    if frame_idx[1] - frame_idx[0] < 49:
                        continue
                    if 'prompt_camera_with_scene_video_inpainted' in v:
                        prompt = v['prompt_camera_with_scene_video_inpainted']['concise']
                    else:
                        prompt = v['prompt_camera_with_scene_video']['concise']
                    # file_path = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds'
                    # if os.path.exists(os.path.join(file_path, f'{dataset_name}_{data_name}_{k}')):
                    #     continue
                    data_names.append(f'{dataset_name}_{data_name}_{k}')
                    text_prompts.append(prompt)
                    extrinsics_list.append(extrinsics[frame_idx[0]:frame_idx[1]])
                    intrinsics_list.append(intrinsics[frame_idx[0]:frame_idx[1]])
                    scene_paths.append(scene_path)
                    dataset_cnt += 1
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
            dataset_cnt_dist.append((dataset_name, dataset_cnt))
        # [print(name, cnt) for name, cnt in dataset_cnt_dist]
        # print(f"Total Data Count: {len(data_names)}")
        # print(f"Total scene Count: {len(list(set(scene_paths)))}")
        # input()
        return data_names, extrinsics_list, intrinsics_list, scene_paths, text_prompts

    def sample(self):
        sample_N = self.cfg.sample_data
        if sample_N is None:
            return
        idx = random.sample(range(len(self.data_names)), sample_N)

        def random_sample(target_list, target_idx):
            return [target_list[i] for i in target_idx]

        self.data_names = random_sample(self.data_names, idx)
        self.extrinsics_list = random_sample(self.extrinsics_list, idx)
        self.intrinsics_list = random_sample(self.intrinsics_list, idx)
        self.scene_paths = random_sample(self.scene_paths, idx)
        self.text_prompts = random_sample(self.text_prompts, idx)
        return

    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        extrinsics = self.extrinsics_list[idx]
        intrinsics = self.intrinsics_list[idx]
        scene_path = self.scene_paths[idx]
        text_prompt = self.text_prompts[idx]

        scene_data = None
        scene_embeds = None

        if scene_path == None:
            normalized_extrinsics = extrinsics
            avg_scale = torch.tensor([1.0])
            scene_data = [0.0]
            point_obj = {
                "coord": np.zeros((1, 3)),  # (N, 3)
                "color": np.zeros((1, 3)),  # (N, 3)
                # "batch": batch, # (N, )
                "offset": np.ones((1, )), # (B, )
                "normal": np.zeros((1, 3)),  # (N, 3)
                # "segment": numpy.array,  # (N,) optional
            }
            point_obj = self.pc_transform(point_obj)
        elif self.cfg.load_points:
            pc_embeds_path = os.path.join(self.cfg.embed_path, data_name)
            if self.cfg.load_saved_pc_embeds and os.path.exists(pc_embeds_path):
                scene_embeds_data = torch.load(pc_embeds_path, map_location='cpu')
                scene_embeds = scene_embeds_data['embed']
                scene_sim = scene_embeds_data['sim']

                scene_data = None
                saved_avg_scale_path = os.path.join(os.path.dirname(scene_path), "avg_scale", f"{data_name.split('_')[-1]}.json")
                saved_avg_scale = torch.tensor([json.load(open(saved_avg_scale_path, "r"))])
                normalized_extrinsics, _, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, avg_scale=saved_avg_scale)
            else:
                if scene_path.endswith(".pt"):
                    scene_data = torch.load(scene_path) # (N, 6), coordinates (3) + color (3, range:0~1), 
                elif scene_path.endswith(".ply"):
                    scene_data = torch.tensor(load_ply_as_pointcloud(scene_path))
                scene_data[:, 3:] *= 255.0
                normalized_extrinsics, normalized_scene_pc, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, scene_data[:, :3])
                # avg_scale_dir = os.path.join(os.path.dirname(scene_path), 'avg_scale')
                # os.makedirs(avg_scale_dir, exist_ok=True)
                # with open(os.path.join(avg_scale_dir, f'{data_name.split('_')[-1]}.json'), 'w') as f:
                #     json.dump(avg_scale[0].item(), f)
                
                scene_data[:, :3] = normalized_scene_pc * 1.8
                scene_data = scene_data[mask]
                normalized_scene_pc = normalized_scene_pc[mask]
                offset = torch.tensor([scene_data.shape[0]])
                R_y_down_to_z_up = torch.tensor([
                    [1.0,  0,  0],
                    [0,  0, -1.0],
                    [0,  1.0,  0],
                ])
                scene_data[:, :3]  = scene_data[:, :3] @ R_y_down_to_z_up
                scene_data = scene_data.numpy()
                offset = offset.numpy()
                point_obj = {
                    "coord": scene_data[:, :3],  # (N, 3)
                    "color": scene_data[:, 3:],  # (N, 3)
                    # "batch": batch, # (N, )
                    "offset": offset, # (B, )
                    "normal": np.zeros_like(scene_data[:, :3]),  # (N, 3)
                    # "segment": numpy.array,  # (N,) optional
                }
                point_obj = self.pc_transform(point_obj)
        else:
            saved_avg_scale_path = os.path.join(os.path.dirname(scene_path), "avg_scale", f"{data_name.split('_')[-1]}.json")
            saved_avg_scale = torch.tensor([json.load(open(saved_avg_scale_path, "r"))])
            normalized_extrinsics, _, avg_scale, mask = normalize_camera_extrinsics_and_points(extrinsics, avg_scale=saved_avg_scale)

        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        width = intrinsics[:, 0, 2] * 2
        height = intrinsics[:, 1, 2] * 2
        normalized_fx = (fx / width)[:, None] # (N, 1)
        normalized_fy = (fy / height)[:, None] # (N, 1)
        normalized_intrinsics = torch.cat([normalized_fx, normalized_fy], dim=-1).to(torch.float32)
        normalized_intrinsics = normalized_intrinsics / normalized_intrinsics[0:1, :]

        cam_param = torch.cat([normalized_extrinsics[:, :3, 0], normalized_extrinsics[:, :3, 1], normalized_extrinsics[:, :3, 3], normalized_intrinsics], dim=-1).to(torch.float32) # (N_cam, 11)
        first_extrinsic = extrinsics[0]

        data = {
            'data_name': data_name,
            'cam_param': cam_param,
            'first_extrinsic': first_extrinsic,
            'width': width,
            'height': height,
            'avg_scale': avg_scale,
            'text_prompt': text_prompt,
            # 'extrinsics': normalized_extrinsics,
            # 'intrinsics': intrinsics[:, :3, :3],
            # 'points': normalized_scene_pc,
            # 'inverse': point_obj['inverse'],
        }
        if scene_data is not None:
            data['pc'] = point_obj
        if scene_embeds is not None:
            data['pc_embeds'] = scene_embeds
            data['pc_sim'] = scene_sim
        
        return data
    
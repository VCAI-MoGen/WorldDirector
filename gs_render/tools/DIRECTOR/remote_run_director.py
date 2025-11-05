import os

import numpy as np
import torch
import torch.nn.functional as F
import json

from utils.random_utils import set_random_seed
from visualization.common_viz import to_device, encode_text, load_clip_model, get_batch
import argparse

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import scipy.ndimage

from typing import Any, Dict, List, Tuple

import clip
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
import torch.nn.functional as F

from src.training.diffuser import Diffuser
from src.datasets.multimodal_dataset import MultimodalDataset

from hydra.core.global_hydra import GlobalHydra

import sys

def init_remote(
    config_name: str, config_abs_dir: str
) -> Tuple[Diffuser, clip.model.CLIP, MultimodalDataset, torch.device]:
    
    abs_dir = os.path.join(os.path.dirname(__file__))
    prev_dir = os.getcwd()
    os.chdir(abs_dir)
    
    if GlobalHydra.instance().is_initialized():
        global_hydra = GlobalHydra.instance()
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=config_abs_dir):
        config = compose(config_name=config_name)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # Initialize model
    device = torch.device(config.compnode.device)
    diffuser = instantiate(config.diffuser)
    state_dict = torch.load(config.checkpoint_path, map_location=device, weights_only=False)["state_dict"]
    state_dict["ema.initted"] = diffuser.ema.initted
    state_dict["ema.step"] = diffuser.ema.step
    diffuser.load_state_dict(state_dict, strict=False)
    diffuser.to(device).eval()

    # Initialize CLIP model
    clip_model = load_clip_model("ViT-B/32", device)

    # Initialize dataset
    config.dataset.char.load_vertices = True
    config.batch_size = 1
    dataset = instantiate(config.dataset)
    dataset.set_split("test")
    diffuser.modalities = list(dataset.modality_datasets.keys())
    diffuser.get_matrix = dataset.get_matrix
    diffuser.v_get_matrix = dataset.get_matrix

    for key, val in config.guidance_control.custom_guidance_ctrl.items():
        if not val:
            config.guidance_control.custom_guidance_ctrl[key] = 0.0

    GlobalHydra.set_instance(global_hydra)
    os.chdir(prev_dir)

    return diffuser, clip_model, dataset, device, config.guidance_control

def get_batch_custom(char_feature, padding_mask, prompt, clip_model, seq_feat, device):
    # generate dummy batch for inference

    # Encode text
    caption_seq, caption_tokens = encode_text([prompt], clip_model, None, device)

    if seq_feat:
        caption_feat = caption_seq[0]
        caption_feat = F.pad(caption_feat, (0, 0, 0, 77 - caption_feat.shape[0]))
        caption_feat = caption_feat.unsqueeze(0).permute(0, 2, 1)
    else:
        caption_feat = caption_tokens
    
    dummy_batch = {}
    dummy_batch['traj_feat'] = torch.zeros((1, 9, 300)) # (1, 9, 300)
    dummy_batch['padding_mask'] = padding_mask[None] # (1, 300)
    dummy_batch['intrinsics'] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    dummy_batch['char_filename'] = ["dummy"]
    dummy_batch['char_feat'] = char_feature[None] # (1, 3, 300)
    dummy_batch['char_padding_mask'] = padding_mask[None] # (1, 300)
    dummy_batch['caption_feat'] = caption_feat
    dummy_batch['caption_raw'] = [prompt]
    dummy_batch['char_raw'] = {}
    dummy_batch['char_raw']['char_centers'] = torch.zeros((1, 3, 300))
    dummy_batch['char_raw']['char_vertices'] = torch.zeros((1, 300, 10475, 3))
    dummy_batch['char_raw']['char_vertices'] = torch.zeros((1, 300, 20908, 3))
    dummy_batch['char_raw']['char_joints'] = torch.zeros((1, 300, 127, 3))

    return dummy_batch

def encode_char_traj(char_dataset, char_traj):
    """
    char_dataset: CharacterDataset
    char_traj: (N, 3) numpy array
    """
    raw_char_feature = torch.from_numpy(char_traj).to(torch.float32)
    padding_size = char_dataset.num_cams - raw_char_feature.shape[0]

    char_feature = raw_char_feature.clone()
    if char_dataset.velocity:
        velocity = char_feature[1:].clone() - char_feature[:-1].clone()
        char_feature = torch.cat([raw_char_feature[0][None], velocity])

    if char_dataset.standardize:
        # Normalize the first frame (orgin) and the rest (velocity) separately
        if len(char_dataset.norm_mean) == 6:
            char_feature[0] -= char_dataset.norm_mean[:3, 0].to(raw_char_feature.device)
            char_feature[0] /= char_dataset.norm_std[:3, 0].to(raw_char_feature.device)
            char_feature[1:] -= char_dataset.norm_mean[3:, 0].to(raw_char_feature.device)
            char_feature[1:] /= char_dataset.norm_std[3:, 0].to(raw_char_feature.device)
        # Normalize all in one
        else:
            char_feature -= char_dataset.norm_mean[:, 0].to(raw_char_feature.device)
            char_feature /= char_dataset.norm_std[:, 0].to(raw_char_feature.device)
    padded_char_feature = F.pad(
        char_feature,
        (0, 0, 0, padding_size),
    )
    padding_mask = torch.ones((char_dataset.num_cams))
    padding_mask[char_feature.shape[0] :] = 0

    if char_dataset.sequential:
        padded_char_feature = padded_char_feature.permute(1, 0)
    else:
        padded_char_feature = padded_char_feature.reshape(-1)

    return padded_char_feature, padding_mask

def traj_dir_to_npy(traj_dir_path):
    json_list = [f'{i}.json' for i in range(len(os.listdir(traj_dir_path)))]
    traj_list = []
    for json_path in json_list:
        with open(os.path.join(traj_dir_path, json_path), 'r') as f:
            traj = json.load(f)["traj"]
        traj_list.append(traj)
    traj_list = np.array(traj_list) # (N, 3)

    return traj_list

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
    
def smooth_camera_parameters(positions, rotations, pos_sigma=2, rot_window=5):
    """
    positions: (N, 3)
    rotations: (N, 3, 3)
    pos_sigma: Gaussian smoothing strength for position
    rot_window: number of frames for rotation SLERP window
    """
    N = len(positions)

    smoothed_positions = scipy.ndimage.gaussian_filter1d(
        positions, sigma=pos_sigma, axis=0
    )

    return smoothed_positions

class RemoteDirector:
    def __init__(self):
        self.diffuser, self.clip_model, self.dataset, self.device, self.control_cfg = \
            init_remote("config_remote", os.path.abspath(os.path.join(os.path.dirname(__file__), "configs")))
        
        self.root_path = self.control_cfg.root_path
        self.PROMPT = self.control_cfg.prompt

        SEED = 33
        W_GUIDANCE = 1.4

        set_random_seed(SEED)
        self.diffuser.gen_seeds = np.array([SEED])
        self.diffuser.guidance_weight = W_GUIDANCE
        self.diffuser.guidance_ctrl = self.control_cfg

        print(self.diffuser.guidance_ctrl)

        self.focal = [312.7740969872587, 312.7740969872587]
        self.princpt = [360, 240]

    def remote_run_director(self, save_path=None, camera_start_pose=None):
        seq_feat = self.diffuser.net.model.clip_sequential
        char_dataset = self.dataset.modality_datasets['char']

        char_traj_dir = os.path.join(self.root_path, 'traj', 'smplx_optimized', 'traj_smoothed')
        source_cam_params_path = os.path.join(self.root_path, 'traj', 'cam_params_scene')

        target_cam_params_path = os.path.join(self.root_path, 'detailed', 'cam_params_scene')
        os.makedirs(target_cam_params_path, exist_ok=True)
        if save_path is not None:
            target_cam_params_path = os.path.join(save_path, 'detailed', 'cam_params_scene')
            os.makedirs(target_cam_params_path, exist_ok=True)

        up_vector_path = os.path.join(self.root_path, 'up_vector.npy')
        smplx_scale_path = os.path.join(self.root_path, 'traj', 'smplx_optimized', 'scale.json')
        with open(smplx_scale_path, 'r') as f:
            self.smplx_scale = json.load(f)[0]

        char_traj = traj_dir_to_npy(char_traj_dir)
        text = self.PROMPT

        self.raw_char_traj = torch.tensor(char_traj)

        # ===== Print Parameters =====
        print("Prompt :", text)
        print("Guidance control :", self.control_cfg)
        # ============================

        # c2w transformation
        with open(os.path.join(source_cam_params_path, '0.json'), 'r') as f:
            source_cam_param = json.load(f)

        # camera to world transformation for character trajectory
        R_w2c = np.array(source_cam_param["R"])
        t_w2c = np.array(source_cam_param["t"])
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        char_traj = (R_c2w @ char_traj.T).T + t_c2w[None]

        # scaling by smplx_scale
        char_traj = char_traj / self.smplx_scale

        # start point as origin
        self.char_start = char_traj[0:1].copy() # (1, 3)
        char_traj = char_traj - self.char_start

        # rotate scene coordinates --> director character trajectory coordinates
        scene_up_vector = np.load(up_vector_path)
        self.R_scene_up_to_director_up = rotation_matrix_from_vectors(scene_up_vector, np.array([0.0, -1.0, 0.0]))
        char_traj = (self.R_scene_up_to_director_up @ char_traj.T).T

        self.char_traj = torch.tensor(char_traj)

        char_feature, padding_mask = encode_char_traj(char_dataset, char_traj)
        batch = get_batch_custom(char_feature, padding_mask, text, self.clip_model, seq_feat, self.device)

        # with torch.no_grad():
        out = self.diffuser.predict_step(batch, 0, camera_start_pose)
        padding_mask = out["padding_mask"][0].to(bool).cpu()
        valid_len = int(padding_mask.sum().item())
        if valid_len == 0:
            raise RuntimeError("No valid frames found in the generated trajectory.")

        traj_matrices = out["gen_samples"][0].cpu() # (num_cams, 4, 4) (Padded)
        generated_cam_traj = traj_matrices[padding_mask] # (N, 4, 4)
        generated_cam_traj_npy = generated_cam_traj.detach().cpu().numpy()

        smoothed_pos = smooth_camera_parameters(generated_cam_traj_npy[:, :3, 3], generated_cam_traj_npy[:, :3, :3])
        generated_cam_traj_npy[:, :3, 3] = smoothed_pos
        
        for i, c2w in enumerate(generated_cam_traj_npy):
            R_c2w = self.R_scene_up_to_director_up.T @ c2w[:3, :3]
            t_c2w = (self.R_scene_up_to_director_up.T @ c2w[:3, 3] + self.char_start[0]) * self.smplx_scale
            R_w2c = R_c2w.T
            t_w2c = -R_c2w.T @ t_c2w
            cam_param = {
                "R": R_w2c.tolist(),
                "t": t_w2c.tolist(),
                "focal": self.focal,
                "princpt": self.princpt
            }
            with open(os.path.join(target_cam_params_path, f'{i}.json'), 'w') as f:
                json.dump(cam_param, f)

        print("Generating Camera Parameter Complete!")

        return out, padding_mask
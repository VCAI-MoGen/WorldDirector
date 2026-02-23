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
from core_pkg.common.utils.gaussian_utils import GaussianPlyUtils

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def sample_from_N_gaussians_torch(
    means, scales, rotations, n_samples_per_gaussian, rgb
):
    """
    means:     (N, 3)
    scales:    (N, 3)
    rotations: (N, 3, 3)
    n_samples_per_gaussian: (N, ),
    """
    N = means.shape[0]
    device = means.device
    samples = []

    for i in range(N):
        scale = scales[i].clamp_min(1e-6)
        # cov = rotations[i] @ torch.diag(scales[i] ** 2) @ rotations[i].T
        z = torch.randn(n_samples_per_gaussian[i], 3, device=device) * scale
        x = means[i] + z @ rotations[i].T
        samples.append(x)

    return torch.cat(samples, dim=0)

def quat_to_rotmat_torch(q):
    """
    q: (..., 4) quaternion (w, x, y, z)
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)

    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))

def gaussian_volumes(scales):
    """
    scales: (N, 3)  # std
    return: (N,)
    """
    return torch.prod(scales, dim=-1)

def compute_sample_counts(
    scales,
    opacities=None,
    base_samples=8,
    min_samples=1,
    max_samples=128,
):
    """
    scales:    (N, 3)
    opacities: (N,) or None
    """
    volumes = gaussian_volumes(scales)  # (N,)

    weights = volumes
    if opacities is not None:
        weights = weights * opacities

    weights = weights / (weights.mean() + 1e-8)
    counts = base_samples * weights
    counts = torch.clamp(counts, min_samples, max_samples)

    return counts.long()

class CamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_path = '/data1/cympyc1785/SceneData/DL3DV/scenes'

        self.data_names, self.cam_params, self.first_extrinsics, self.scene_pcs, self.scene_pcs_offsets, self.text_prompts, self.frames_list = self.load_data()

    def load_data(self):

        # data_list = os.listdir(self.dataset_path)
        data_list = []
        for data_part in os.listdir(self.dataset_path):
            data_list += [data_part + '/' + data_name for data_name in os.listdir(os.path.join(self.dataset_path, data_part))]

        print("Loading data...")
        for data_name in tqdm(sorted(data_list)):
            try:
                if data_name != "006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed":
                    continue
                data_path = osp.join(self.dataset_path, data_name)
                # Scene processing
                scene_path = osp.join(data_path, 'scene.ply')
                target_scene_path = scene_path.replace('.ply', '.pt')
                if os.path.exists(target_scene_path):
                    continue
                scene = GaussianPlyUtils.load_from_ply(scene_path).to_parameter_structure()
                """
                All values are torch
                xyz: (N_gs, 3)
                opacities: (N_gs, 1)
                features_dc: (N_gs, 1, 3)
                features_rest: (N_gs, 15, 3)
                scales: (N_gs, 3)
                rotations: (N_gs, 4)
                """
                xyz = scene.xyz
                opacities = scene.opacities
                scales = scene.scales
                rotations = scene.rotations
                
                # TODO: features_dc and features_rest should be converted to rgb
                rgb = C0 * scene.features_dc[:, 0, :]
                rgb = torch.clamp(rgb + 0.5, 0.0, 1.0)

                # print(xyz.shape)
                # print(rgb.shape)

                opacity_threshold = 0.01
                mask = (opacities > opacity_threshold).repeat(1, 3)
                xyz = xyz[mask].view(-1, 3)
                rgb = rgb[mask].view(-1, 3)
                scales = torch.exp(scales[mask].view(-1, 3))
                rotations = quat_to_rotmat_torch(rotations[mask[:, 0:1].repeat(1, 4)].view(-1, 4))

                sample_counts = compute_sample_counts(scales, base_samples=50, min_samples=1, max_samples=128)
                # sample_counts = torch.ones((xyz.shape[0])).long()
                xyz_new = sample_from_N_gaussians_torch(xyz, scales, rotations, sample_counts, rgb)
                rgb_new = torch.repeat_interleave(
                    rgb, sample_counts, dim=0
                )
                # print(xyz_new.shape)
                # print(rgb_new.shape)

                # rgb = torch.ones(rgb.shape)
                # rgb[:, 0] = 0
                # rgb_new = torch.ones(rgb_new.shape)
                # rgb_new[:, 0] = 0


                # print(xyz.shape)
                # print(rgb.shape)

                # scene_data = torch.cat((xyz, rgb), dim=-1) # (N_gs, 6)
                scene_data = torch.cat((xyz_new, rgb_new), dim=-1) # (N_gs, 6)

                torch.save(scene_data, target_scene_path)
            except Exception as e:
                print(e)
                with open('error_name.txt', 'a') as f:
                    f.writelines(f'{data_name},{e}\n')


dataset = CamDataset(cfg=None)
dataset.load_data()
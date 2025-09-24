from pathlib import Path
import random

from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
import torch.nn.functional as F
from typing import Tuple

from utils.file_utils import load_txt
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

import pickle
from smplx import SMPLX
from pytorch3d.pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.pytorch3d.transforms import matrix_to_rotation_6d


# ------------------------------------------------------------------------------------- #

num_cams = None

# ------------------------------------------------------------------------------------- #


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        name: str,
        set_name: str,
        dataset_dir: str,
        num_rawfeats: int,
        num_feats: int,
        num_cams: int,
        standardize: bool,
        align_first_frame: bool,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.set_name = set_name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / "traj"
        self.intrinsics_dir = self.dataset_dir / "intrinsics"
        self.char_traj_dir = self.dataset_dir / "char"

        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams

        self.augmentation = None
        self.standardize = standardize # False
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean"])
            self.norm_std = torch.Tensor(mean_std["norm_std"])
            self.shift_mean = torch.Tensor(mean_std["shift_mean"])
            self.shift_std = torch.Tensor(mean_std["shift_std"])
            self.velocity = mean_std["velocity"]
        else:
            self.velocity = True
        self.translation_scale = float(kwargs.get("translation_scale", 1.0))
        if self.translation_scale <= 0:
            raise ValueError("translation_scale must be positive")
        if self.standardize:
            scale = self.translation_scale
            self.norm_mean *= scale
            self.norm_std *= scale
            self.shift_mean *= scale
            self.shift_std *= scale
        self.align_first_frame = align_first_frame

        # Add smplx
        self.smpl_dir = self.dataset_dir / "smplh"
        self.smplx_model_path = self.model_root = self.dataset_dir / "human_model"
        self.smplx_model = SMPLX(model_path=str(self.smplx_model_path), model_type="smplx", gender="neutral")

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split
        split_path = Path(self.dataset_dir) / f"{self.set_name}_{split}_split.txt"
        split_traj = load_txt(split_path).split("\n")

        if self.split == "train":
            train_size = int(len(split_traj) * train_rate)
            train_split_traj = random.sample(split_traj, train_size)
            self.filenames = sorted(train_split_traj)
        else:
            self.filenames = sorted(split_traj)

        return self

    # --------------------------------------------------------------------------------- #

    def get_smplx_info(self, filename: str):
        smplx_info = {}
        smpl_path = self.smpl_dir / (filename + ".pkl")
        with open(smpl_path, "rb") as f:
            params = pickle.load(f, encoding="latin1")
        
        # Remove gradients
        for key in params.keys():
            params[key] = params[key].detach()
        # params.keys(): ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl']
        # shape: [(N, 10), (N, 1, 3, 3), (N, 21, 3, 3), (N, 15, 3, 3), (N, 15, 3, 3), (N, 3)]
        smplx_info["params"] = params
        
        return smplx_info

    def get_feature(
        self, raw_matrix_trajectory: TensorType["num_cams", 4, 4]
    ) -> TensorType[9, "num_cams"]:
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        #raw_trans = torch.clone(matrix_trajectory[:, :3, 3]) * self.translation_scale
        raw_trans = torch.clone(matrix_trajectory[:, :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        if self.standardize:
            device = raw_trans.device
            shift_mean = self.shift_mean.to(device)
            shift_std = self.shift_std.to(device)
            norm_mean = self.norm_mean.to(device)
            norm_std = self.norm_std.to(device)
            raw_trans[0] -= shift_mean
            raw_trans[0] /= shift_std
            raw_trans[1:] -= norm_mean
            raw_trans[1:] /= norm_std

        # Compute the 6D continuous rotation
        raw_rot = matrix_trajectory[:, :3, :3]
        rot6d = raw_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        # Stack rotation 6D and translation
        rot6d_trajectory = torch.hstack([rot6d, raw_trans]).permute(1, 0)

        return rot6d_trajectory

    def get_matrix(
        self, raw_rot6d_trajectory: TensorType[9, "num_cams"]
    ) -> TensorType["num_cams", 4, 4]:
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device)[None].repeat(num_cams, 1, 1)

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        if self.standardize:
            raw_trans[0] *= self.shift_std.to(device)
            raw_trans[0] += self.shift_mean.to(device)
            raw_trans[1:] *= self.norm_std.to(device)
            raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        raw_trans = raw_trans / self.translation_scale
        matrix_trajectory[:, :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0)
        raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
        matrix_trajectory[:, :3, :3] = raw_rot

        return matrix_trajectory

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> Tuple[str, TensorType["num_cams", 4, 4]]:
        filename = self.filenames[index]

        trajectory_filename = filename + ".txt"
        trajectory_path = self.data_dir / trajectory_filename

        trajectory = read_kitti_poses_file(trajectory_path)
        matrix_trajectory = torch.from_numpy(np.array(trajectory.poses_se3)).to(
            torch.float32
        ) # (N, 4, 4)

        if self.align_first_frame:
            # Load SMPL-X params
            char_filename = filename + ".npy"
            char_path = self.char_traj_dir / char_filename
            char_feature = torch.from_numpy(np.load((char_path))).to(torch.float32) # (N, 3)

            smplx_info = self.get_smplx_info(filename)
            smplx_params = smplx_info['params']
            root_pose = smplx_params['global_orient'] # (N, 1, 3, 3)
            root_pose = root_pose[:, 0, :, :] # (N, 3, 3)

            # first-frame SMPL-X forward
            global_orient = matrix_to_axis_angle(smplx_params['global_orient'][0])
            transl = smplx_params['transl'][0].view(1,3)
            body_pose = matrix_to_axis_angle(smplx_params['body_pose'][0].view(21,3,3)).view(1,21*3)
            left_hand_pose = matrix_to_axis_angle(smplx_params['left_hand_pose'][0].view(15,3,3)).view(1,15*3)
            right_hand_pose = matrix_to_axis_angle(smplx_params['right_hand_pose'][0].view(15,3,3)).view(1,15*3)
            betas = smplx_params['betas'][0].view(1,-1)
            self.smplx_model.use_pca = False
            out = self.smplx_model(
                global_orient = global_orient,
                body_pose = body_pose,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                betas = betas,
                transl = transl,
                jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                expression=torch.zeros(1, 10),
            )
                
            joints0 = out.joints

            R0 = root_pose[0, :, :] # (3, 3)
            R0_inv = R0.T
            #t0 = char_feature[0:1, :] # (1, 3)
            t0 = joints0[0][0]

            R_cam = matrix_trajectory[:, :3, :3]
            t_cam = matrix_trajectory[:, :3, 3]

            R_cam = R0_inv @ R_cam
            t_cam = (t_cam - t0) @ R0_inv.T

            matrix_trajectory[:, :3, :3] = R_cam
            matrix_trajectory[:, :3, 3] = t_cam

        intrinsics_filename = filename + ".npy"
        intrinsics_path = self.intrinsics_dir / intrinsics_filename
        intrinsics = np.load(intrinsics_path)

        trajectory_feature = self.get_feature(matrix_trajectory)

        padded_trajectory_feature = F.pad(
            trajectory_feature, (0, self.num_cams - trajectory_feature.shape[1])
        )
        # Padding mask: 1 for valid cams, 0 for padded cams
        padding_mask = torch.ones((self.num_cams))
        padding_mask[trajectory_feature.shape[1] :] = 0

        return_value = (
            trajectory_filename,
            padded_trajectory_feature,
            padding_mask,
            intrinsics,
        )

        # Return R0_inv and t0
        if self.align_first_frame:
            return_value = (
                trajectory_filename,
                padded_trajectory_feature,
                padding_mask,
                intrinsics,
                R0_inv,
                t0
            )

        return return_value

    def __len__(self):
        return len(self.filenames)

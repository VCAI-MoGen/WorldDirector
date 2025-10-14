from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import warnings
import pickle
from smplx import SMPLX
from pytorch3d.pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.pytorch3d.transforms import matrix_to_rotation_6d


class CharacterDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        standardize: bool,
        align_first_frame: bool,
        num_feats: int,
        num_cams: int,
        sequential: bool,
        num_vertices: int,
        num_faces: int,
        load_vertices: bool,
        **kwargs,
    ):
        super().__init__()
        self.modality = "char"
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.traj_dir = self.dataset_dir / "traj"
        self.data_dir = self.dataset_dir / self.name # char
        self.vert_dir = self.dataset_dir / "vert_decimated"
        self.center_dir = self.dataset_dir / "char_raw"

        self.filenames = None
        self.standardize = standardize # False
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean_h"])[:, None]
            self.norm_std = torch.Tensor(mean_std["norm_std_h"])[:, None]
            self.velocity = mean_std["velocity"]
        else:
            self.velocity = True
        self.align_first_frame = align_first_frame
      

        self.num_cams = num_cams
        self.num_feats = num_feats
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.sequential = sequential # True

        self.load_vertices = load_vertices # True when visualization

        custom_dir_arg = kwargs.get('custom_dir')
        if custom_dir_arg is None:
            custom_candidate = (self.dataset_dir.parent / 'custom').resolve()
        else:
            custom_candidate = Path(custom_dir_arg).resolve()
        self.custom_dir = custom_candidate if custom_candidate.exists() else None


        # Add smplx
        self.smpl_dir = self.dataset_dir / "smplh"
        self.smplx_model_path = self.model_root = self.dataset_dir / "human_model"
        self.smplx_model = SMPLX(model_path=str(self.smplx_model_path), model_type="smplx", gender="neutral")

    

    def get_smplx_info(self, filename: str, frame_count: int | None = None):
        smplx_info = {}
        smpl_path = self.smpl_dir / (filename + ".pkl") 
        with open(smpl_path, "rb") as f:
            params = pickle.load(f, encoding="latin1")

        smplx_info["params"] = params
        return smplx_info

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        char_filename = filename + ".npy"
        char_path = self.data_dir / char_filename

        # raw data for visualization
        raw_char_feature = torch.from_numpy(np.load(char_path)).to(torch.float32) # (N, 3)
        padding_size = self.num_cams - raw_char_feature.shape[0]
        padded_raw_char_feature = F.pad(
            raw_char_feature, (0, 0, 0, padding_size)
        ).permute(1, 0)

        # Load SMPL-X params
        N = raw_char_feature.shape[0]
        smplx_info = self.get_smplx_info(filename, frame_count=N)
        smplx_params = smplx_info['params']

        if self.load_vertices:  # True when visualization
            B = raw_char_feature.shape[0]

            global_orient = smplx_params['global_orient'].view(B, 1, 3, 3)
            global_orient = matrix_to_axis_angle(global_orient).view(B, 3)

            transl = smplx_params['transl'].view(B, 3)
            body_pose = matrix_to_axis_angle(
                smplx_params['body_pose'].view(B * 21, 3, 3)
            ).view(B, 21 * 3)
            left_hand_pose = matrix_to_axis_angle(
                smplx_params['left_hand_pose'].view(B * 15, 3, 3)
            ).view(B, 15 * 3)
            right_hand_pose = matrix_to_axis_angle(
                smplx_params['right_hand_pose'].view(B * 15, 3, 3)
            ).view(B, 15 * 3)
            betas = smplx_params['betas']
            if betas.shape[0] == 1:
                betas = betas.expand(B, -1)
            else:
                betas = betas.view(B, -1)

            self.smplx_model.use_pca = False
            out = self.smplx_model(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas,
                transl=transl,
                jaw_pose=torch.zeros(B, 3, device=transl.device),
                leye_pose=torch.zeros(B, 3, device=transl.device),
                reye_pose=torch.zeros(B, 3, device=transl.device),
                expression=torch.zeros(B, 10, device=transl.device),
                return_verts=True,
            )

            verts = out.vertices
            faces = torch.as_tensor(self.smplx_model.faces, device=verts.device)
            faces = faces.unsqueeze(0).repeat(verts.shape[0], 1, 1).contiguous()

            padded_verts = F.pad(verts, (0, 0, 0, 0, 0, padding_size))
            padded_faces = F.pad(faces, (0, 0, 0, 0, 0, padding_size))
        
        # Input data for model
        char_feature = raw_char_feature.clone()
        if self.align_first_frame:
            root_pose = smplx_params['global_orient'] # (N, 1, 3, 3)
            root_pose = root_pose[:, 0, :, :] # (N, 3, 3)

            R0 = root_pose[0, :, :] # (3, 3)
            R0_inv = R0.T
            
            # t0 from the raw centers was used during recent debugging sessions
            t0 = char_feature[0:1, :]

        # Normalize
        if self.align_first_frame:
            char_feature = (char_feature - t0) @ R0_inv.T
            root_pose = R0_inv @ root_pose
            #root_pose_6d = matrix_to_rotation_6d(root_pose)
            

            
        
        if self.velocity:
            velocity = char_feature[1:].clone() - char_feature[:-1].clone()
            char_feature = torch.cat([raw_char_feature[0][None], velocity])

        if self.standardize: # set your flag in configs/datasets/trajectory/rot_6d_trajectory.yml
            # Normalize the first frame (orgin) and the rest (velocity) separately
            if len(self.norm_mean) == 6:
                char_feature[0] -= self.norm_mean[:3, 0].to(raw_char_feature.device)
                char_feature[0] /= self.norm_std[:3, 0].to(raw_char_feature.device)
                char_feature[1:] -= self.norm_mean[3:, 0].to(raw_char_feature.device)
                char_feature[1:] /= self.norm_std[3:, 0].to(raw_char_feature.device)
            # Normalize all in one
            else:
                char_feature -= self.norm_mean[:, 0].to(raw_char_feature.device)
                char_feature /= self.norm_std[:, 0].to(raw_char_feature.device)
        
        #concat root pose 6d with char_feature(character trajectory)
        #char_feature = torch.cat([char_feature, root_pose_6d], dim=1)

        padded_char_feature = F.pad(
            char_feature,
            (0, 0, 0, self.num_cams - char_feature.shape[0]),
        )

        if self.sequential:
            padded_char_feature = padded_char_feature.permute(1, 0)
        else:
            padded_char_feature = padded_char_feature.reshape(-1)

        raw_feats = {"char_raw_feat": padded_raw_char_feature}
        raw_feats["char_centers"] = padded_raw_char_feature
        if self.load_vertices:
            raw_feats["char_vertices"] = padded_verts
            raw_feats["char_faces"] = padded_faces

        return char_filename, padded_char_feature, raw_feats

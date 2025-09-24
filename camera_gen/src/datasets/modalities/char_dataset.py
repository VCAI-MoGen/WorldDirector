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

DEBUG_SMPL = os.getenv("SMPL_DEBUG_SHAPES", "0").lower() in {"1", "true", "yes"}


def _debug_param_shapes(sample_id: str, params: dict) -> None:
    if not DEBUG_SMPL:
        return
    entries = []
    for key in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl", "betas"):
        val = params.get(key)
        if val is None:
            entries.append(f"{key}=None")
            continue
        try:
            if isinstance(val, torch.Tensor):
                arr = val.detach()
                entries.append(
                    f"{key}=tensor{tuple(arr.shape)} requires_grad={val.requires_grad}"
                )
            else:
                arr = np.asarray(val)
                entries.append(f"{key}=array{arr.shape}")
        except Exception as exc:
            entries.append(f"{key}=error({exc})")
    warnings.warn(
        f"[SMPL debug] {sample_id}: " + ", ".join(entries),
        RuntimeWarning,
    )


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

        # Add smplx
        self.smpl_dir = self.dataset_dir / "smplh"
        self.smplx_model_path = self.model_root = self.dataset_dir / "human_model"
        self.smplx_model = SMPLX(model_path=str(self.smplx_model_path), model_type="smplx", gender="neutral")

    def get_smplx_info(self, filename: str):
        smplx_info = {}
        smpl_path = self.smpl_dir / (filename + ".pkl")
        with open(smpl_path, "rb") as f:
            params = pickle.load(f, encoding="latin1")
        
        # Remove gradients
        for key in params.keys():
            params[key] = params[key].detach()
        
        # Catch trash data
        """
        flag = False
        for key in params.keys():
            if torch.isnan(params[key]).any() or torch.isinf(params[key]).any():
                flag = True
        if flag:
            with open('trash.txt', 'a') as f:
                f.writelines(f"{filename}\n")
        """
        # params.keys(): ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl']
        # shape: [(N, 10), (N, 1, 3, 3), (N, 21, 3, 3), (N, 15, 3, 3), (N, 15, 3, 3), (N, 3)]
        smplx_info["params"] = params
        
        # TODO: render mesh
        if self.load_vertices:
            pass
        
        return smplx_info

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        char_filename = filename + ".npy"
        char_path = self.data_dir / char_filename

        # raw data for visualization
        raw_char_feature = torch.from_numpy(np.load((char_path))).to(torch.float32) # (N, 3)
        padding_size = self.num_cams - raw_char_feature.shape[0]
        padded_raw_char_feature = F.pad(
            raw_char_feature, (0, 0, 0, padding_size)
        ).permute(1, 0)
        N = raw_char_feature.shape[0]
        # Load SMPL-X params
        smplx_info = self.get_smplx_info(filename)
        smplx_params = smplx_info['params']

        center_path = self.center_dir / char_filename  # Center to offset mesh
        center_offset = torch.from_numpy(np.load(center_path)[0]).to(torch.float32)
        if self.load_vertices: # True when visualization
            vert_path = self.vert_dir / char_filename
            raw_verts = np.load(vert_path, allow_pickle=True)[()]
            if raw_verts["vertices"] is None:
                padded_verts = torch.zeros(
                    (self.num_cams, self.num_vertices, 3), dtype=torch.float32
                )
                padded_faces = torch.zeros(
                    (self.num_cams, self.num_faces, 3), dtype=torch.int16
                )
            else:
                verts = torch.from_numpy(raw_verts["vertices"]).to(torch.float32)
                faces = torch.from_numpy(raw_verts["faces"]).to(torch.int16)
                verts -= center_offset
                padded_verts = F.pad(verts, (0, 0, 0, 0, 0, padding_size))
                padded_faces = F.pad(faces, (0, 0, 0, 0, 0, padding_size))

        #first frame forward smplx output
    
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

        
        # Input data for model
        char_feature = raw_char_feature.clone()

        root_pose = smplx_params['global_orient'] # (N, 1, 3, 3)
        root_pose = root_pose[:, 0, :, :] # (N, 3, 3)

        R0 = root_pose[0, :, :] # (3, 3)
        R0_inv = R0.T
        # TODO : smplx forward and t0 should be the first frame of pelvis joing position (out.joints)
        
        #t0 = char_feature[0:1, :] # (1, 3)
        t0 = joints0[0][0]

        # Normalize
        if self.align_first_frame:
            char_feature = (char_feature - t0) @ R0_inv.T
            root_pose = R0_inv @ root_pose
            root_pose_6d = matrix_to_rotation_6d(root_pose)
            
        
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
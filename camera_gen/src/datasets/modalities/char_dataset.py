from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
from smplx import SMPLX
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.io import save_obj, save_ply

#####debug via MeshLab
def export_mesh_ply(verts,faces,joints=None, out_dir="./debug_vis",prefix="seq"):
    output_dir = os.path.join(out_dir,prefix)
    os.makedirs(output_dir,exist_ok=True)
    T = verts.shape[0]
    for i in range(T):
        save_path = os.path.join(output_dir,f"{prefix}_frame{i:03d}.obj")
        save_obj(save_path, verts[i].cpu(), faces.cpu())
    if joints is not None:
        joint_path = os.path.join(output_dir,f"{prefix}_joint{i:03d}.ply")
        save_ply(joint_path, joints[i].cpu())

class CharacterDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        standardize: bool,
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
        self.data_dir = self.dataset_dir / self.name
        self.vert_dir = self.dataset_dir / "smplh"
        self.center_dir = self.dataset_dir / "char_raw"

        self.filenames = None
        self.standardize = standardize
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean_h"])[:, None]
            self.norm_std = torch.Tensor(mean_std["norm_std_h"])[:, None]
            self.velocity = mean_std["velocity"]

        self.smpl_model = SMPLX(
            model_path="./et-data/human_model",
            model_type="smplx",
            gender="neutral"
        )
        self.num_cams = num_cams
        self.num_feats = num_feats
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.sequential = sequential
        self.load_vertices = load_vertices

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        char_filename = filename + ".npy"
        char_path = self.data_dir / char_filename

        device = torch.device("cpu")

        # load char traj (root translations)
        raw_char_feature = torch.from_numpy(np.load(char_path)).to(torch.float32).to(device)
        padding_size = max(0, self.num_cams - raw_char_feature.shape[0])
        padded_raw_char_feature = F.pad(
            raw_char_feature, (0, 0, 0, padding_size)
        ).permute(1, 0)

        # optional center offset
        center_path = self.center_dir / char_filename
        if os.path.exists(center_path):
            center_offset = torch.from_numpy(np.load(center_path)[0]).to(torch.float32).to(device)
        else:
            center_offset = torch.zeros(3, device=device)

        if self.load_vertices:
            pkl_path = (self.vert_dir / filename).with_suffix(".pkl")
            self.smpl_model = self.smpl_model.to(device)
            if os.path.exists(pkl_path):
                # (A) ET 샘플 데이터: pkl 존재
                with open(pkl_path, "rb") as f:
                    smpl_params = pickle.load(f, encoding="latin1")

                body_pose = torch.tensor(smpl_params["body_pose"], dtype=torch.float32, device=device)
                betas = torch.tensor(smpl_params["betas"], dtype=torch.float32, device=device)
                global_orient_mats = torch.tensor(smpl_params["global_orient"], dtype=torch.float32, device=device)
                transl = torch.tensor(smpl_params["transl"], dtype=torch.float32, device=device)
                left_hand_pose = torch.tensor(smpl_params["left_hand_pose"], dtype=torch.float32, device=device)
                right_hand_pose = torch.tensor(smpl_params["right_hand_pose"], dtype=torch.float32, device=device)

                self.smpl_model.use_pca = False
                B = transl.shape[0]

                if global_orient_mats.dim() == 4:
                    R_smpl_all = global_orient_mats[:, 0]
                else:
                    R_smpl_all = global_orient_mats

                output = self.smpl_model(
                    betas=betas,
                    body_pose=matrix_to_axis_angle(body_pose.view(B*21, 3, 3)).view(B, 21*3),
                    global_orient=matrix_to_axis_angle(R_smpl_all),
                    transl=transl,
                    left_hand_pose=matrix_to_axis_angle(left_hand_pose.view(B*15, 3, 3)).view(B, 15*3),
                    right_hand_pose=matrix_to_axis_angle(right_hand_pose.view(B*15, 3, 3)).view(B, 15*3),
                    jaw_pose=torch.zeros(B, 3, device=device),
                    leye_pose=torch.zeros(B, 3, device=device),
                    reye_pose=torch.zeros(B, 3, device=device),
                    expression=torch.zeros(B, 10, device=device),
                )

                verts_smpl = output.vertices 
                joints_smpl = output.joints
                root_idx= 0

                # 첫 프레임 root pose 기준 normalization
                R0 = R_smpl_all[0].detach()
                p0 = joints_smpl[0, root_idx].detach()
                R_inv = torch.inverse(R0)

                verts_norm = (verts_smpl - p0) @ R_inv
                joints_norm = (joints_smpl - p0) @ R_inv

                char_traj_norm = (raw_char_feature - p0) @ R_inv

                # faces
                faces = torch.tensor(self.smpl_model.faces, dtype=torch.long, device=device)
                faces = faces.unsqueeze(0).expand(verts_norm.shape[0], -1, -1).contiguous()

                padded_verts = F.pad(verts_norm, (0,0,0,0,0,padding_size)) if padding_size>0 else verts_norm
                padded_faces = F.pad(faces, (0,0,0,0,0,padding_size)) if padding_size>0 else faces
                padded_raw_char_feature = F.pad(char_traj_norm, (0,0,0,padding_size)).permute(1,0)

                R0, t0 = R0, p0

            else:
                # if .pkl not exists
                B = raw_char_feature.shape[0]
                transl = raw_char_feature.clone()

                verts = self.smpl_model(
                    betas=torch.zeros(1, 10, device=device).expand(B, -1),
                    body_pose=torch.zeros(B, 21*3, device=device),
                    global_orient=torch.zeros(B, 3, device=device),
                    transl=transl,
                    left_hand_pose=torch.zeros(B, 15*3, device=device),
                    right_hand_pose=torch.zeros(B, 15*3, device=device),
                    jaw_pose=torch.zeros(B, 3, device=device),
                    leye_pose=torch.zeros(B, 3, device=device),
                    reye_pose=torch.zeros(B, 3, device=device),
                    expression=torch.zeros(B, 10, device=device),
                ).vertices

                t0 = transl[0].clone()
                verts = verts - t0
                char_traj_norm = transl - t0

                faces = torch.tensor(self.smpl_model.faces, dtype=torch.long, device=device)
                faces = faces.unsqueeze(0).expand(verts.shape[0], -1, -1).contiguous()

                padded_verts = F.pad(verts, (0,0,0,0,0,padding_size)) if padding_size>0 else verts
                padded_faces = F.pad(faces, (0,0,0,0,0,padding_size)) if padding_size>0 else faces
                padded_raw_char_feature = F.pad(char_traj_norm, (0,0,0,padding_size)).permute(1,0)

                R0 = torch.eye(3, device=device)
                t0 = transl[0].detach()

        # char_feature (velocity + standardization)
        char_feature = raw_char_feature.clone()
        if self.velocity:
            velocity = char_feature[1:] - char_feature[:-1]
            char_feature = torch.cat([char_feature[0:1], velocity], dim=0)

        if self.standardize:
            if len(self.norm_mean) == 6:
                char_feature[0] -= self.norm_mean[:3, 0].to(device)
                char_feature[0] /= self.norm_std[:3, 0].to(device)
                char_feature[1:] -= self.norm_mean[3:, 0].to(device)
                char_feature[1:] /= self.norm_std[3:, 0].to(device)
            else:
                char_feature -= self.norm_mean[:, 0].to(device)
                char_feature /= self.norm_std[:, 0].to(device)

        padded_char_feature = F.pad(
            char_feature, (0, 0, 0, self.num_cams - char_feature.shape[0])
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
            raw_feats["char_root_ref"] = {"R0": R0, "t0": t0}
        
        # export_mesh_ply(
        #     verts_norm.cpu(),
        #     faces[0].cpu(),
        #     joints=joints_norm,
        #     out_dir="./debug_vis",
        #     prefix = filename,
        # )

        return char_filename, padded_char_feature, raw_feats

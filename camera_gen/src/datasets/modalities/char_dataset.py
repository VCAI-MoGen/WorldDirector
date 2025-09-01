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

# --- Meshlab debugging supplementary function ---
def export_mesh_ply(verts, faces, joints=None, out_dir="./debug_vis", prefix="seq"):
    output_dir = os.path.join(out_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    T = verts.shape[0]
    for i in range(T):
        save_path = os.path.join(output_dir, f"{prefix}_frame{i:03d}.obj")
        save_obj(save_path, verts[i].cpu(), faces.cpu())
        if joints is not None:
            joint_path = os.path.join(output_dir, f"{prefix}_joint{i:03d}.ply")
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
        device: torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        self.modality = "char"
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / self.name
        self.vert_dir = self.dataset_dir / "smplh"

        self.filenames = None  
        self.standardize = standardize
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean_h"])[:, None]
            self.norm_std = torch.Tensor(mean_std["norm_std_h"])[:, None]
            self.velocity = mean_std["velocity"]

        # device initialization
        self.device = torch.device("cpu")

        # SMPL-X model initialization
        self.smpl_model = SMPLX(
            model_path="./et-data/human_model",
            model_type="smplx",
            gender="neutral"
        )
        self.smpl_model.use_pca = False  # hands 45D 

        # faces는 
        self.faces_tensor = self.smpl_model.faces_tensor  # (F, 3)

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

        device = self.device
        raw_char_feature = torch.from_numpy(np.load(char_path)).to(torch.float32)
        B = raw_char_feature.shape[0]

        # initial buffer
        verts_norm     = torch.zeros(B, self.num_vertices, 3, device=device)
        char_traj_norm = torch.zeros(B, 3, device=device)
        faces = self.faces_tensor
        R0 = torch.eye(3, device=device)
        t0 = torch.zeros(3, device=device)

        pkl_path = (self.vert_dir / filename).with_suffix(".pkl")
        if  os.path.exists(pkl_path):
            self.smpl_model = self.smpl_model.to(device)

            with open(pkl_path, "rb") as f:
                smpl_params = pickle.load(f, encoding="latin1")

           
            body_mat  = torch.tensor(smpl_params["body_pose"],       dtype=torch.float32)    # (Tp,21,3,3)
            lhand_mat = torch.tensor(smpl_params.get("left_hand_pose",  np.zeros((1,15,3,3))), dtype=torch.float32)
            rhand_mat = torch.tensor(smpl_params.get("right_hand_pose", np.zeros((1,15,3,3))), dtype=torch.float32)
            go_mat    = torch.tensor(smpl_params["global_orient"],   dtype=torch.float32)    # (Tp,1,3,3) or (Tp,3,3)
            betas     = torch.tensor(smpl_params.get("betas",  np.zeros((1,10))), dtype=torch.float32)   # (Tp,10) or (1,10)
            transl    = torch.tensor(smpl_params.get("transl", np.zeros((1,3))),  dtype=torch.float32)   # (Tp,3)  or (1,3)

            # mat → axis-angle
            def mat_to_aa(mat, J):
                Tp = mat.shape[0]
                return matrix_to_axis_angle(mat.reshape(Tp * J, 3, 3)).reshape(Tp, J * 3)

            Tp = body_mat.shape[0]
            if go_mat.dim() == 4:  # (Tp,1,3,3) → (Tp,3,3)
                go_mat = go_mat.squeeze(1)
            global_orient_aa = matrix_to_axis_angle(go_mat)  # (Tp,3)
            body_pose_aa     = mat_to_aa(body_mat, 21)       # (Tp,63)
            lhand_pose_aa    = mat_to_aa(lhand_mat, 15)      # (Tp,45)
            rhand_pose_aa    = mat_to_aa(rhand_mat, 15)      # (Tp,45)

            #align timestep
            def align_time(x, T):
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                Tx = x.shape[0]
                if Tx == T:
                    return x
                if Tx == 1:
                    return x.expand(T, *x.shape[1:])
                return x[:min(Tx, T)]

            body_pose_aa     = align_time(body_pose_aa,     B)
            lhand_pose_aa    = align_time(lhand_pose_aa,    B)
            rhand_pose_aa    = align_time(rhand_pose_aa,    B)
            global_orient_aa = align_time(global_orient_aa, B)
            betas            = align_time(betas,            B)
            transl           = align_time(transl,           B)

            # SMPL-X forward
            with torch.no_grad():
                out = self.smpl_model(
                    betas=betas,                         # (B,10)
                    body_pose=body_pose_aa,              # (B,63)
                    global_orient=global_orient_aa,      # (B,3)
                    transl=transl,                       # (B,3)
                    left_hand_pose=lhand_pose_aa,        # (B,45)
                    right_hand_pose=rhand_pose_aa,       # (B,45)
                    jaw_pose=torch.zeros(B, 3),
                    leye_pose=torch.zeros(B, 3),
                    reye_pose=torch.zeros(B, 3),
                    expression=torch.zeros(B, 10),
                )
            if self.load_vertices:
                verts_smpl = out.vertices  # (B,V,3)
            joints_smpl = out.joints    # (B,J,3)

            #1st frame normalization
            p0 = joints_smpl[0, 0].detach()
            R0 = go_mat[0].detach() 
            if self.load_vertices:
                verts_norm  = (verts_smpl - p0) @ R0
            char_traj_norm = (raw_char_feature - p0) @ R0
            #6d rotation from normalized root pose
            #R_norm = (go_mat.transpose(-1,-2)@R0).transpose(-1,-2)
            #rot6d = R_norm[:,:,2].permute(0,2,1).contiguous().reshape(B,6)
            t0 = p0


        else:
            # if pkl files does not exist
            p0 = raw_char_feature[0].clone()
            with torch.no_grad():
                out = self.smpl_model(transl=raw_char_feature) 
            if self.load_vertices:
                verts_smpl = out.vertices# (B,V,3)
            verts_norm = verts_smpl - p0
            char_traj_norm = raw_char_feature - p0
            t0 = p0

        #char_feature velocity & standardization + feature padding
        char_feature = char_traj_norm.clone()
        if self.velocity:
            velocity = char_feature[1:] - char_feature[:-1]
            char_feature = torch.cat([char_feature[0:1], velocity], dim=0)

        if self.standardize:
            pass
        #adding 6D rotation to charactr trajectory
        #char_feature = torch.cat([char_feature,rot6d],dim=-1)
        padding_size = max(0, self.num_cams - char_feature.shape[0])
        padded_char_feature = F.pad(char_feature, (0, 0, 0, padding_size))
        padded_verts = F.pad(verts_norm, (0, 0, 0, 0, 0, padding_size))

        if self.sequential:
            padded_char_feature = padded_char_feature.permute(1, 0)  # (3,T)
        else:
            padded_char_feature = padded_char_feature.reshape(-1)

        # ---  Return ---
        raw_feats = {
            "char_raw_feat": padded_char_feature,
            "char_centers": padded_char_feature,
        }
        if self.load_vertices:
            padded_faces = faces.unsqueeze(0).expand(self.num_cams, -1, -1)
            raw_feats["char_vertices"] = padded_verts
            raw_feats["char_faces"] = padded_faces
        raw_feats["char_root_ref"] = {"R0": R0, "t0": t0}

        return char_filename, padded_char_feature, raw_feats
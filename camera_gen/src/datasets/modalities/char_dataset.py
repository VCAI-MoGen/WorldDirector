from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle 
from smplx import SMPLX
import smplx
from pytorch3d.transforms import matrix_to_axis_angle
# ------------------------------------------------------------------------------------- #


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
        #self.vert_dir = self.dataset_dir / "vert_decimated"
        self.vert_dir = self.dataset_dir / "smplh"
        self.center_dir = self.dataset_dir / "char_raw"

        self.filenames = None
        self.standardize = standardize
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean_h"])[:, None]
            self.norm_std = torch.Tensor(mean_std["norm_std_h"])[:, None]
            self.velocity = mean_std["velocity"]

    
        self.smpl_model = SMPLX(model_path = "/data/danielk0112/ET/DIRECTOR/et-data/human_model", model_type='smplx',gender = "neutral")
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

        raw_char_feature = torch.from_numpy(np.load((char_path))).to(torch.float32)
        padding_size = max(0, self.num_cams - raw_char_feature.shape[0])
        padded_raw_char_feature = F.pad(
            raw_char_feature, (0, 0, 0, padding_size)
        ).permute(1, 0)

        device = raw_char_feature.device
        M = torch.tensor(
            [[-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]],
            dtype= torch.float32,
            device = device
        )

        center_path = self.center_dir / char_filename  # Center to offset mesh
        if os.path.exists(center_path):
            center_offset = torch.from_numpy(np.load(center_path)[0]).to(torch.float32).to(device)
        else:
            center_offset = raw_char_feature[0].clone()
        # if self.load_vertices:
        #     vert_path = self.vert_dir / char_filename
        #     raw_verts = np.load(vert_path, allow_pickle=True)[()]
        #     print(type(raw_verts))
        #     if raw_verts["vertices"] is None:
        #         padded_verts = torch.zeros(
        #             (self.num_cams, self.num_vertices, 3), dtype=torch.float32
        #         )
        #         padded_faces = torch.zeros(
        #             (self.num_cams, self.num_faces, 3), dtype=torch.int16
        #         )
        #     else:
        #         verts = torch.from_numpy(raw_verts["vertices"]).to(torch.float32)
        #         faces = torch.from_numpy(raw_verts["faces"]).to(torch.int16)
        #         verts -= center_offset
        #         padded_verts = F.pad(verts, (0, 0, 0, 0, 0, padding_size))
        #         padded_faces = F.pad(faces, (0, 0, 0, 0, 0, padding_size))

        # betas: (58, 10)
        # global_orient: (58, 1, 3, 3)
        # body_pose: (58, 21, 3, 3)
        # left_hand_pose: (58, 15, 3, 3)
        # right_hand_pose: (58, 15, 3, 3)
        # transl: (58, 3)

        if self.load_vertices:
            pkl_path = (self.vert_dir / filename).with_suffix(".pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    smpl_params = pickle.load(f, encoding = 'latin1')
                body_pose = torch.tensor(smpl_params['body_pose'], dtype=torch.float32, device=device)
                betas = torch.tensor(smpl_params['betas'], dtype=torch.float32,device=device)
                global_orient = torch.tensor(smpl_params['global_orient'], dtype=torch.float32,device=device)
                transl = torch.tensor(smpl_params['transl'],dtype=torch.float32,device=device)
                left_hand_pose = torch.tensor(smpl_params['left_hand_pose'],dtype=torch.float32,device=device)
                right_hand_pose = torch.tensor(smpl_params['right_hand_pose'], dtype=torch.float32,device=device)
                
                
                self.smpl_model.use_pca = False

                B = global_orient.shape[0]
                global_orient_aa = matrix_to_axis_angle(global_orient[:,0])
                body_pose_aa = matrix_to_axis_angle(body_pose.view(B * 21, 3, 3)).view(B, 21 * 3)
                left_hand_pose_aa = matrix_to_axis_angle(left_hand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)
                right_hand_pose_aa = matrix_to_axis_angle(right_hand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)

                jaw_pose = torch.zeros(B, 3, device=device)
                leye_pose= torch.zeros(B, 3, device=device)
                reye_pose= torch.zeros(B, 3, device=device)
                expression= torch.zeros(B, 10, device=device)


                if jaw_pose.shape[0] == 1:
                    jaw_pose = jaw_pose.expand(B, -1)

                if leye_pose.shape[0] == 1:
                    leye_pose = leye_pose.expand(B, -1)

                if reye_pose.shape[0] == 1:
                    reye_pose = reye_pose.expand(B, -1)

                if expression.shape[0] == 1:
                    expression = expression.expand(B, -1)
                
                R0 = global_orient[0,0].detach()
                t0 = transl[0].detach()
            else:
                self.smpl_model.use_pca = False
                B = raw_char_feature.shape[0]

                betas = torch.zeros(1,10,device=device).expand(B,-1)
                body_pose_aa = torch.zeros(B,21*3,device=device)
                transl = raw_char_feature.clone()
                global_orient_aa = torch.zeros(B,3,device=device)
                left_hand_pose_aa = torch.zeros(B,15*3,device=device)
                right_hand_pose_aa = torch.zeros(B,15*3,device=device)
                jaw_pose = torch.zeros(B, 3,device=device)
                leye_pose= torch.zeros(B, 3, device=device)
                reye_pose= torch.zeros(B, 3, device=device)
                expression= torch.zeros(B, 10, device=device)

                #R0=torch.eye(3,dtype=torch.float32,device=device)
                R0=global_orient_aa[0].detach()
                t0=raw_char_feature[0].detach()

            output = self.smpl_model(
                betas=betas,
                body_pose=body_pose_aa,
                global_orient=global_orient_aa,
                transl=transl,
                left_hand_pose=left_hand_pose_aa,
                right_hand_pose=right_hand_pose_aa,
                jaw_pose = jaw_pose,
                leye_pose = leye_pose,
                reye_pose = reye_pose,
                expression = expression
            )
            
            if os.path.exists(pkl_path):
                verts = output.vertices - center_offset
            else:
                verts = (output.vertices @ M) - center_offset
            faces = torch.tensor(self.smpl_model.faces, dtype = torch.long, device = verts.device)
            faces = faces.unsqueeze(0).expand(verts.shape[0],-1,-1).contiguous()


            if padding_size>0:
                padded_verts = F.pad(verts, (0, 0, 0, 0, 0, padding_size))
                padded_faces = F.pad(faces, (0, 0, 0, 0, 0, padding_size))
            else:
                padded_verts = verts
                padded_faces = faces
            


        char_feature = raw_char_feature.clone()
        if self.velocity:
            velocity = char_feature[1:].clone() - char_feature[:-1].clone()
            char_feature = torch.cat([char_feature[0:1], velocity], dim=0)

        if self.standardize:
            # Normalize the first frame (orgin) and the rest (velocity) separately
            if len(self.norm_mean) == 6:
                char_feature[0] -= self.norm_mean[:3, 0].to(device)
                char_feature[0] /= self.norm_std[:3, 0].to(device)
                char_feature[1:] -= self.norm_mean[3:, 0].to(device)
                char_feature[1:] /= self.norm_std[3:, 0].to(device)
            # Normalize all in one
            else:
                char_feature -= self.norm_mean[:, 0].to(device)
                char_feature /= self.norm_std[:, 0].to(device)
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
            ################
            raw_feats["char_root_ref"] = {"R0":R0, "t0":t0}

        return char_filename, padded_char_feature, raw_feats

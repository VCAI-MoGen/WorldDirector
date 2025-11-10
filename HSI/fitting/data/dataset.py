import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, get_patch_img
from utils.transforms import change_kpt_name
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from pytorch3d.ops import corresponding_points_alignment
import json
import math
import open3d as o3d

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, cfg):
        self.cfg = cfg
        self.root_path = self.cfg.data_root_path
        self.transform = transform
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.cam_params, self.kpts, self.depth_paths, self.smplx_params, self.smplx_shape_param, self.flame_params, self.flame_shape_param, self.frame_idx_list, self.img_paths, self.depth_vggt_paths = self.load_data()
        self.get_smplx_trans_init() # get initial smplx translation 
        self.get_flame_root_init() # get initial flame root pose and translation

    def load_data(self):
        # load camera parameter
        cam_params = {}
        cam_param_path_list = glob(osp.join(self.root_path, 'cam_params_scene', '*.json'))
        for cam_param_path in cam_param_path_list:
            frame_idx = int(cam_param_path.split('/')[-1][:-5])
            with open(cam_param_path) as f:
                cam_params[frame_idx] = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}

        # load keypoints
        kpts = {}
        kpt_path_list = glob(osp.join(self.root_path, 'keypoints_whole_body', '*.json'))
        for kpt_path in kpt_path_list:
            frame_idx = int(kpt_path.split('/')[-1][:-5])
            with open(kpt_path) as f:
                kpt = np.array(json.load(f), dtype=np.float32)
            kpt = change_kpt_name(kpt, self.kpt['name'], smpl_x.kpt['name'])
            kpts[frame_idx] = kpt

        # sapiens depth map paths
        depth_paths = {}
        depth_path_list = glob(osp.join(self.root_path, 'depths', '*.npy'))
        for depth_path in depth_path_list:
            if '_mask.npy' in depth_path:
                continue
            frame_idx = int(depth_path.split('/')[-1][:-4])
            depth_paths[frame_idx] = depth_path

        # vggt dpeth map paths 
        depth_vggt_paths = {}
        depth_vggt_path_list = glob(osp.join(self.root_path, 'depths_vggt_optimized', '*.npy'))
        for depth_vggt_path in depth_vggt_path_list:
            frame_idx = int(depth_vggt_path.split('/')[-1][:-4])
            depth_vggt_paths[frame_idx] = depth_vggt_path

        # load initial smplx parameters
        smplx_params = {}
        smplx_param_path_list = glob(osp.join(self.root_path, 'smplx_init', '*.json'))
        for smplx_param_path in smplx_param_path_list:
            frame_idx = int(smplx_param_path.split('/')[-1][:-5])
            with open(smplx_param_path) as f:
                smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}
        
        smplx_shape_param = torch.zeros((smpl_x.shape_param_dim))

        # load initial flame parameters 
        flame_params = {}
        flame_param_path_list = glob(osp.join(self.root_path, 'flame_init', 'flame_params', '*.json'))
        for flame_param_path in flame_param_path_list:
            frame_idx = int(flame_param_path.split('/')[-1][:-5])
            with open(flame_param_path) as f:
                flame_param = json.load(f)
            if not flame_param['is_valid']:
                for k in flame_param.keys():
                    if 'pose' in k:
                        flame_param[k] = torch.zeros((3)).float() # dummy
                flame_param['expr'] = torch.zeros((flame.expr_param_dim)).float() # dummy
            for k,v in flame_param.items():
                if k == 'is_valid':
                    continue
                else:
                    flame_param[k] = torch.FloatTensor(v)
            flame_params[frame_idx] = flame_param
        with open(osp.join(self.root_path, 'flame_init', 'shape_param.json')) as f:
            flame_shape_param = torch.FloatTensor(json.load(f))

        frame_idx_list = [int(x) for x in cam_params.keys()]

        img_paths = {}
        for frame_idx in frame_idx_list:
            img_path = osp.join(self.root_path, 'frames', str(frame_idx) + '.png')
            img_paths[frame_idx] = img_path
        return cam_params, kpts, depth_paths, smplx_params, smplx_shape_param, flame_params, flame_shape_param, frame_idx_list, img_paths, depth_vggt_paths

    def get_smplx_trans_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            focal, princpt = self.cam_params[frame_idx]['focal'], self.cam_params[frame_idx]['princpt'].copy()

            kpt = self.kpts[frame_idx]
            kpt_img = kpt[:,:2]
            kpt_valid = (kpt[:,2:] > 0.1).astype(np.float32)
            bbox = get_bbox(kpt_img, kpt_valid[:,0])
            bbox = set_aspect_ratio(bbox, aspect_ratio=self.cfg.kpt_proj_shape[1]/self.cfg.kpt_proj_shape[0])
            
            # get the root translation in the camera coordinate system
            t_z = math.sqrt(focal[0]*focal[1]*self.cfg.body_3d_size*self.cfg.body_3d_size/(bbox[2]*bbox[3])) # meter
            t_x = bbox[0] + bbox[2]/2 # pixel
            t_y = bbox[1] + bbox[3]/2 # pixel
            t_x = (t_x - princpt[0]) / focal[0] * t_z # meter
            t_y = (t_y - princpt[1]) / focal[1] * t_z # meter
            t_xyz = torch.FloatTensor([t_x, t_y, t_z]) 
            self.smplx_params[frame_idx]['trans'] = t_xyz

    def get_flame_root_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            
            smplx_root_pose = axis_angle_to_matrix(self.smplx_params[frame_idx]['root_pose'])
            smplx_root_trans = self.smplx_params[frame_idx]['trans']
            smplx_init = torch.matmul(smplx_root_pose, smpl_x.layer['neutral'].v_template.permute(1,0)).permute(1,0)
            smplx_init = (smplx_init - smplx_init.mean(0)[None,:] + smplx_root_trans[None,:])[smpl_x.face_vertex_idx,:]
            
            # get initial root pose and translation with the rigid alignment
            flame_init = flame.layer.v_template
            RTs = corresponding_points_alignment(flame_init[None], smplx_init[None])
            R = RTs.R.permute(0,2,1)[0]
            flame_init = torch.matmul(R, flame_init.permute(1,0)).permute(1,0)
            self.flame_params[frame_idx]['root_pose'] = matrix_to_axis_angle(R)
            self.flame_params[frame_idx]['trans'] = -flame_init.mean(0) + smplx_init.mean(0)

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_idx_list[idx]

        # 2D keypoint
        kpt_img = self.kpts[frame_idx][:,:2] # W, H
        kpt_valid = (self.kpts[frame_idx][:,2:] > 0.3).astype(np.float32)
        bbox = get_bbox(kpt_img, kpt_valid[:,0])
        bbox_orig = bbox
        bbox = set_aspect_ratio(bbox, aspect_ratio=self.cfg.kpt_proj_shape[1]/self.cfg.kpt_proj_shape[0])

        # keypoint affine transformation
        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox, self.cfg.kpt_proj_shape) # get transformation matrix with dummy image
        kpt_img_xy1 = np.concatenate((kpt_img, np.ones_like(kpt_img[:,:1])),1)
        kpt_img = np.dot(img2bb_trans, kpt_img_xy1.transpose(1,0)).transpose(1,0)

        # load depth (by sapiens and vggt)
        # vggt depth shape is different with original image shape. we should handle this.
        depth = np.load(self.depth_paths[frame_idx])
        depth_mask = np.load(self.depth_paths[frame_idx][:-4] + '_mask.npy').astype(np.float32)
        human_mask = depth_mask
        depth_vggt = np.load(self.depth_vggt_paths[frame_idx])
        depth_vggt_mask = cv2.resize(depth_mask, (depth_vggt.shape[1], depth_vggt.shape[0]), interpolation=cv2.INTER_NEAREST)

        origin_h, origin_w = depth.shape
        vggt_h, vggt_w = depth_vggt.shape

        kpt_img_vggt = self.kpts[frame_idx][:,:2]
        kpt_img_vggt[:, 0] = kpt_img_vggt[:, 0] /  origin_w * vggt_w
        kpt_img_vggt[:, 1] = kpt_img_vggt[:, 1] /  origin_h * vggt_h
        bbox_vggt = get_bbox(kpt_img_vggt, kpt_valid[:,0])
        bbox_vggt = set_aspect_ratio(bbox_vggt, aspect_ratio=self.cfg.kpt_proj_shape[1]/self.cfg.kpt_proj_shape[0])
        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox_vggt, self.cfg.kpt_proj_shape) # get transformation matrix with dummy image
        kpt_img_vggt_xy1 = np.concatenate((kpt_img_vggt, np.ones_like(kpt_img_vggt[:,:1])),1)
        kpt_img_vggt = np.dot(img2bb_trans, kpt_img_vggt_xy1.transpose(1,0)).transpose(1,0)

        # lift 2d keypoint to 3d using depth map
        kpt_2d = self.kpts[frame_idx][:, :2]
        kpt_2d[:, 0] = kpt_2d[:, 0] / origin_w * vggt_w
        kpt_2d[:, 1] = kpt_2d[:, 1] / origin_h * vggt_h
        u = np.clip(np.round(kpt_2d[:, 0]).astype(np.int32), 0, vggt_w - 1)
        v = np.clip(np.round(kpt_2d[:, 1]).astype(np.int32), 0, vggt_h - 1)
        fx, fy = self.cam_params[frame_idx]['focal']
        cx, cy = self.cam_params[frame_idx]['princpt']
        fx, fy = np.array([fx / origin_w * vggt_w, fy / origin_h * vggt_h], dtype=np.float32)
        cx, cy = np.array([cx // 2, cy // 2],  dtype=np.float32)
        z = depth_vggt[v, u]
        x = (kpt_2d[:, 0] - cx) * z / fx
        y = (kpt_2d[:, 1] - cy) * z / fy
        kpt_3d = np.stack((x, y, z), axis=1)
        

        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox, self.cfg.depth_render_shape) # get transformation matrix with dummy image
        depth = cv2.warpAffine(depth, img2bb_trans, (self.cfg.depth_render_shape[1], self.cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth_mask = cv2.warpAffine(depth_mask, img2bb_trans, (self.cfg.depth_render_shape[1], self.cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth = (depth * (depth_mask==1))[None,:,:]

        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox_vggt, self.cfg.depth_render_shape) # get transformation matrix with dummy image
        depth_vggt = cv2.warpAffine(depth_vggt, img2bb_trans, (self.cfg.depth_render_shape[1], self.cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth_vggt_mask = cv2.warpAffine(depth_vggt_mask, img2bb_trans, (self.cfg.depth_render_shape[1], self.cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth_vggt = (depth_vggt * (depth_vggt_mask==1))[None,:,:]

        # smplx parameter
        smplx_param = self.smplx_params[frame_idx]

        # flame parameter
        if np.sum(kpt_valid[smpl_x.kpt['part_idx']['face'],0]) == 0:
            self.flame_params[frame_idx]['is_valid'] = False
        flame_param = self.flame_params[frame_idx]
        flame_param['shape'] = self.flame_shape_param
        flame_valid = flame_param['is_valid']

        # modify intrincis
        focal = self.cam_params[frame_idx]['focal']
        princpt = self.cam_params[frame_idx]['princpt']
        focal_vggt = np.array([focal[0] / origin_w * vggt_w, focal[1] / origin_h * vggt_h], dtype=np.float32)
        princpt_vggt = np.array([vggt_w // 2, vggt_h // 2],  dtype=np.float32)
        focal_kpt = np.array([focal[0] / bbox[2] * self.cfg.kpt_proj_shape[1], focal[1] / bbox[3] * self.cfg.kpt_proj_shape[0]], dtype=np.float32)
        princpt_kpt = np.array([(princpt[0] - bbox[0]) / bbox[2] * self.cfg.kpt_proj_shape[1], (princpt[1] - bbox[1]) / bbox[3] * self.cfg.kpt_proj_shape[0]], dtype=np.float32)
        focal_depth = np.array([focal[0] / bbox[2] * self.cfg.depth_render_shape[1], focal[1] / bbox[3] * self.cfg.depth_render_shape[0]], dtype=np.float32)
        princpt_depth = np.array([(princpt[0] - bbox[0]) / bbox[2] * self.cfg.depth_render_shape[1], (princpt[1] - bbox[1]) / bbox[3] * self.cfg.depth_render_shape[0]], dtype=np.float32)

        focal_vggt = np.array([focal[0] / origin_w * vggt_w, focal[1] / origin_h * vggt_h])
        princpt_vggt = np.array([princpt[0] / origin_w * vggt_w, princpt[1] / origin_h * vggt_h])
        focal_kpt_vggt = np.array([focal_vggt[0] / bbox_vggt[2] * self.cfg.kpt_proj_shape[1], focal_vggt[1] / bbox_vggt[3] * self.cfg.kpt_proj_shape[0]], dtype=np.float32)
        princpt_kpt_vggt = np.array([(princpt_vggt[0] - bbox_vggt[0]) / bbox_vggt[2] * self.cfg.kpt_proj_shape[1], (princpt_vggt[1] - bbox_vggt[1]) / bbox_vggt[3] * self.cfg.kpt_proj_shape[0]], dtype=np.float32)
        focal_depth_vggt = np.array([focal_vggt[0] / bbox_vggt[2] * self.cfg.depth_render_shape[1], focal_vggt[1] / bbox_vggt[3] * self.cfg.depth_render_shape[0]], dtype=np.float32)
        princpt_depth_vggt = np.array([(princpt_vggt[0] - bbox_vggt[0]) / bbox_vggt[2] * self.cfg.depth_render_shape[1], (princpt_vggt[1] - bbox_vggt[1]) / bbox_vggt[3] * self.cfg.depth_render_shape[0]], dtype=np.float32)

        img_orig = load_img(self.img_paths[frame_idx])

        # get extrinsics
        cam_rotation = np.array(self.cam_params[frame_idx]['R']) # shape: 3x3
        cam_translation = np.array(self.cam_params[frame_idx]['t']) #  shape: 3
        data = {
                'kpt_img': kpt_img,
                'kpt_img_vggt': kpt_img_vggt,
                'kpt_valid': kpt_valid, 
                'kpt_3d': kpt_3d,
                'bbox': bbox_orig,
                'depth': depth, 
                'depth_vggt': depth_vggt,
                'mask': human_mask, 
                'smplx_param': smplx_param, 
                'flame_param': flame_param, 
                'flame_valid': flame_valid, 
                'cam_param': {'focal': focal, 'princpt': princpt, 'R': cam_rotation, 't': cam_translation}, 
                'cam_param_vggt': {'focal': focal_vggt, 'princpt': princpt_vggt, 'R': cam_rotation, 't': cam_translation},
                'cam_param_kpt': {'focal': focal_kpt, 'princpt': princpt_kpt, 'R': cam_rotation, 't': cam_translation}, 
                'cam_param_kpt_vggt': {'focal': focal_kpt_vggt, 'princpt': princpt_kpt_vggt, 'R': cam_rotation, 't': cam_translation}, 
                'cam_param_depth': {'focal': focal_depth, 'princpt': princpt_depth, 'R': cam_rotation, 't': cam_translation}, 
                'cam_param_depth_vggt': {'focal': focal_depth_vggt, 'princpt': princpt_depth_vggt, 'R': cam_rotation, 't': cam_translation}, 
                'frame_idx': frame_idx, 
                'img_orig': img_orig,
                }
        return data

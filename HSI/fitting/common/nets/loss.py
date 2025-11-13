import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from utils.smpl_x import smpl_x
import math

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt):
        batch_size = pose_out.shape[0]

        pose_out = pose_out.view(batch_size,-1,3)
        pose_gt = pose_gt.view(batch_size,-1,3)
        
        pose_out = axis_angle_to_matrix(pose_out)
        pose_gt = axis_angle_to_matrix(pose_gt)

        loss = torch.abs(pose_out - pose_gt)
        return loss

class PoseReg(nn.Module):
    def __init__(self):
        super(PoseReg, self).__init__()

    def forward(self, pose_out):
        batch_size = pose_out.shape[0]
        pose_out = pose_out.view(batch_size,-1,3)
        
        # to prevent forward head posture
        head_weight = torch.zeros_like(pose_out)
        head_weight[:,[i for i in range(smpl_x.joint['num']) if smpl_x.joint['name'][i] in ['Spine_1', 'Spine_2', 'Spine_3', 'Neck', 'Head']],:] = 1
        
        # to prevent meaningless foot pose
        foot_weight = torch.zeros_like(pose_out)
        foot_weight[:,[i for i in range(smpl_x.joint['num']) if smpl_x.joint['name'][i] in ['L_Foot', 'R_Foot']],:] = 1

        # to prevent distorted jaw pose
        jaw_weight = torch.zeros_like(pose_out)
        jaw_weight[:,[i for i in range(smpl_x.joint['num']) if smpl_x.joint['name'][i] in ['Jaw']],:] = 1

        loss = pose_out[:,:,0,None] ** 2 * head_weight + pose_out ** 2 * foot_weight + pose_out ** 2 * jaw_weight
        return loss

class ColLoss(nn.Module):
    def __init__(self):
        super(ColLoss, self).__init__()

    def world_to_voxel(self, sdf_dict, xyz):

        # world coordinate ranges (meters)
        x_min, x_max = sdf_dict['x_range']
        y_min, y_max = sdf_dict['y_range']
        z_min, z_max = sdf_dict['z_range']

        # split coordinates
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # normalize each axis to [0, 1]
        x = (x - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)
        z = (z - z_min) / (z_max - z_min)

        # half-voxel shift
        X, Y, Z = sdf_dict['sdf'].shape
        x = (x * (X - 1) + 0.5) / X
        y = (y * (Y - 1) + 0.5) / Y
        z = (z * (Z - 1) + 0.5) / Z
    
        # stack as (B, 3)
        xyz = torch.stack([x, y, z], dim=1)
        return xyz

    def forward(self, sdf_dict, xyz_world):
        batch_size = xyz_world.shape[0]

        # world -> voxel coordinates
        xyz_world = xyz_world.view(batch_size*smpl_x.vertex_num,3)
        xyz_voxel = self.world_to_voxel(sdf_dict, xyz_world)
        xyz_voxel = xyz_voxel.view(batch_size,smpl_x.vertex_num,3)

        # interpolate sdf
        xyz_voxel = xyz_voxel * 2 - 1 # [0,1] -> [-1,1]
        xyz_voxel = xyz_voxel.view(batch_size,1,1,smpl_x.vertex_num,3)
        sdf = sdf_dict['sdf'].permute(2,1,0) # xyz -> zyx
        sdf = sdf[None,None,:,:,:].repeat(batch_size,1,1,1,1)
        sdf_out = F.grid_sample(sdf, xyz_voxel, mode='bilinear', align_corners=True).view(batch_size,smpl_x.vertex_num)

        # return loss
        loss = F.relu(-sdf_out) # do not use outside regions (positive values)
        return loss


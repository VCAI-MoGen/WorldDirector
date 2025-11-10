import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from utils.smpl_x import smpl_x

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

        loss = pose_out[:,:,0,None] ** 2 * head_weight + pose_out ** 2 * foot_weight
        return loss



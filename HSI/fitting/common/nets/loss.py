import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils.smpl_x import smpl_x
from pytorch3d.transforms import axis_angle_to_matrix
from config import cfg

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

# https://github.com/facebookresearch/sapiens/blob/3e829ac27476e4a70b6a01f85e487492afe02df1/seg/mmseg/models/losses/metric_silog_loss.py
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_out, depth_target):
        batch_size = depth_out.shape[0]
        
        loss = 0
        for i in range(batch_size):
            # get foreground
            is_valid = (depth_out[i] > 0) * (depth_target[i] > 0)
            depth_out_i = depth_out[i][is_valid]
            depth_target_i = depth_target[i][is_valid]
            
            # normalize in [0,1]
            depth_out_normalized = (depth_out_i - torch.min(depth_out_i)) / (torch.max(depth_out_i) - torch.min(depth_out_i) + 1e-4)
            depth_target_normalized = (depth_target_i - torch.min(depth_target_i)) / (torch.max(depth_target_i) - torch.min(depth_target_i) + 1e-4)
            is_valid = (depth_out_normalized > 0) * (depth_target_normalized > 0)
            depth_out_normalized, depth_target_normalized = depth_out_normalized[is_valid], depth_target_normalized[is_valid]
            
            # compute loss
            diff_log = torch.log(depth_target_normalized) - torch.log(depth_out_normalized)
            diff_log_mean = torch.mean(diff_log)
            diff_log_sq_mean = torch.mean(diff_log**2)
            loss += torch.sqrt(diff_log_sq_mean - 0.5*diff_log_mean**2)
        loss = loss / batch_size
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

class DepthL1Loss(nn.Module):
    def __init__(self):
        super(DepthL1Loss, self).__init__()

    def forward(self, depth_out, depth_target):
        batch_size = depth_out.shape[0]
        
        loss = 0
        for i in range(batch_size):

            # get foreground
            is_valid = (depth_out[i] > 0) * (depth_target[i] > 0)
            depth_out_i = depth_out[i][is_valid]
            depth_target_i = depth_target[i][is_valid]

            loss += torch.sum(torch.abs(depth_out_i - depth_target_i))
            
        loss = loss / batch_size
        return loss
    
class DepthL2Loss(nn.Module):
    def __init__(self):
        super(DepthL2Loss, self).__init__()

    def forward(self, depth_out, depth_target):
        batch_size = depth_out.shape[0]
        
        loss = 0
        for i in range(batch_size):

            # get foreground
            is_valid = (depth_out[i] > 0) * (depth_target[i] > 0)
            depth_out_i = depth_out[i][is_valid]
            depth_target_i = depth_target[i][is_valid]

            loss += torch.sum((depth_out_i - depth_target_i) ** 2)
            
        loss = loss / batch_size
        return loss


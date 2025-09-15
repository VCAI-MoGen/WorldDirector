import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss, PoseLoss, DepthLoss, PoseReg, DepthL2Loss, RGBLoss, DepthL1Loss
from nets.layer import rasterize_xy, render_mesh_segmentation
from nets.module import HumanGaussian, GaussianRenderer, SceneGaussian
from utils.smpl_x import smpl_x
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import math
from config import cfg

class Model_image(nn.Module):
    def __init__(self):
        super(Model_image, self).__init__()
        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()


        self.gaussian_renderer = GaussianRenderer()

        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
        self.depth_loss = DepthLoss()
        self.depth_l2_loss = DepthL2Loss()
        self.depth_l1_loss = DepthL1Loss()
        self.pose_reg = PoseReg()
        self.rgb_loss = RGBLoss()

    def process_input_smplx_param(self, smplx_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(smplx_param[key]))

        # others
        out['trans'] = smplx_param['trans']
        out['expr'] = smplx_param['expr']
        out['shape'] = smplx_param['shape']
        out['scale'] = smplx_param['scale']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, root_rel=False):
        batch_size = smplx_param['root_pose'].shape[0]
       
        if use_pose:
            root_pose = smplx_param['root_pose']
            body_pose = smplx_param['body_pose']
            jaw_pose = smplx_param['jaw_pose']
            leye_pose = smplx_param['leye_pose']
            reye_pose = smplx_param['reye_pose']
            lhand_pose = smplx_param['lhand_pose']
            rhand_pose = smplx_param['rhand_pose']
        else:
            root_pose = torch.zeros_like(smplx_param['root_pose'])
            body_pose = torch.zeros_like(smplx_param['body_pose'])
            jaw_pose = torch.zeros_like(smplx_param['jaw_pose'])
            leye_pose = torch.zeros_like(smplx_param['leye_pose'])
            reye_pose = torch.zeros_like(smplx_param['reye_pose'])
            lhand_pose = torch.zeros_like(smplx_param['lhand_pose'])
            rhand_pose = torch.zeros_like(smplx_param['rhand_pose'])
        
        if use_expr:
            expr = smplx_param['expr']
        else:
            expr = torch.zeros_like(smplx_param['expr'])

        smplx_scale = smplx_param['scale']
        smplx_trans = smplx_param['trans'] * smplx_scale

        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=smplx_param['shape'])

        # camera-centered 3D coordinate
        mesh_cam = output.vertices * smplx_scale[:, None, :]
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[:, None, :]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + smplx_trans[:, None, :]
        kpt_cam = kpt_cam - root_cam[:,None,:] + smplx_trans[:, None, :]
        
        # project to the 2D space
        if cam_param is not None:
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2)

        if root_rel:
            mesh_cam = mesh_cam - smplx_trans[:, None, :]
            kpt_cam = kpt_cam - smplx_trans[:, None, :]

        traj = 0.5 * (kpt_cam[:, smpl_x.kpt['name'].index('L_Hip'), :] + kpt_cam[:, smpl_x.kpt['name'].index('R_Hip'), :])

        if cam_param is not None:
            return mesh_cam, kpt_cam, root_cam, smplx_trans, kpt_proj, traj
        else:
             return mesh_cam, kpt_cam, root_cam, smplx_trans, traj

    def check_face_visibility(self, face_mesh, leye, reye):
        center = face_mesh.mean(1)
        eye = (leye + reye)/2.

        eye_vec = eye - center
        cam_vec = center - 0

        eye_vec = F.normalize(torch.stack((eye_vec[:,0], eye_vec[:,2]),1), p=2, dim=1)
        cam_vec = F.normalize(torch.stack((cam_vec[:,0], cam_vec[:,2]),1), p=2, dim=1)

        dot_prod = torch.sum(eye_vec * cam_vec, 1)
        face_valid = dot_prod < math.cos(math.pi/4*3)
        return face_valid

    def get_smplx_full_pose(self, smplx_param):
        pose = torch.cat((smplx_param['root_pose'][:,None,:], smplx_param['body_pose'], smplx_param['jaw_pose'][:,None,:], smplx_param['leye_pose'][:,None,:], smplx_param['reye_pose'][:,None,:], smplx_param['lhand_pose'], smplx_param['rhand_pose']),1) # follow smpl_x.joint['name']
        return pose
    
    def forward(self, smplx_inputs, data, return_output):
        smplx_inputs = self.process_input_smplx_param(smplx_inputs) 
        # get coordinates from optimizable parameters
        smplx_mesh_cam, smplx_kpt_cam, smplx_root_cam, smplx_trans, smplx_kpt_proj, traj = self.get_smplx_coord(smplx_inputs, data['cam_param_kpt'])
        
        with torch.no_grad():
            # get coordinates from the initial parameters
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            data['smplx_param']['scale'] = smplx_inputs['scale'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _, _, _ = self.get_smplx_coord(data['smplx_param'], data['cam_param_kpt'])
        
            # check face visibility
            face_valid = self.check_face_visibility(smplx_mesh_cam_init[:,smpl_x.face_vertex_idx,:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('L_Eye'),:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('R_Eye'),:])
            face_valid = face_valid * data['flame_valid']

        bg = torch.rand(3).float().cuda()

        # loss functions
        loss = {}
        weight = torch.ones_like(smplx_kpt_proj)
        if not cfg.warmup:
            weight[:,[i for i in range(smpl_x.kpt['num']) if 'Face' in smpl_x.kpt['name'][i]],:] = 0
            weight[face_valid,:,:] = 1 # do not use 2D loss if face is not visible
        if cfg.use_kpt_loss:
            loss['smplx_kpt_proj'] = self.coord_loss(smplx_kpt_proj, data['kpt_img_vggt'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        
        # rasterize depth map and compute depthmap loss
        if cfg.use_depthmap_loss:
            smplx_depth = rasterize_xy(smplx_mesh_cam, smpl_x.face, data['cam_param_depth_vggt'], cfg.depth_render_shape).zbuf[:,None,:,:,0]
            # loss['depth_l1'] = self.depth_l1_loss(smplx_depth, data['depth_vggt']) * 0.01
            loss['depth_l2'] = self.depth_l2_loss(smplx_depth, data['depth_vggt']) * 0.01
            # loss['depth'] = self.depth_loss(smplx_depth, data['depth_vggt']) * 0.01

        target_image = data['img_orig'].permute(0, 3, 1, 2) / 255
        human_mask = data['mask'][:, None, :, :]
            
        # For Deubgging
        """
        debug_id = 1
        import numpy as np
        import cv2
        sorted_idx = data['frame_idx'].argsort().detach().cpu()
        sorted_depth = smplx_depth[sorted_idx].detach().cpu().numpy()
        np.save(f'pytorch3d_depth{debug_id}.npy', sorted_depth)
        sorted_depth = data['depth_vggt'][sorted_idx].detach().cpu().numpy()
        np.save(f'vggt_depth{debug_id}.npy', sorted_depth)
        print(f"Depth{debug_id} saved...")
        """

        if not cfg.warmup:
                
            loss['smplx_shape_reg'] = smplx_inputs['shape'] ** 2 * 0.01
            loss['smplx_mesh'] = torch.abs((smplx_mesh_cam - smplx_kpt_cam[:,smpl_x.kpt['root_idx'],None,:]) - \
                                            (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
            smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
            smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
            loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.01
            loss['smplx_pose_reg'] = self.pose_reg(smplx_input_pose)
        
        if not return_output:
            return loss, None
        else:
            # outputs
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_trans'] = smplx_trans - smplx_root_cam
            out['human_traj'] = traj
            
            return loss, out

def get_model():
    model = Model_image()
    return model

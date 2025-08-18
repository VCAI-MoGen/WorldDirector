import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply
import os.path as osp
from pytorch3d.io import load_ply
from pytorch3d.ops import knn_points
from smplx.lbs import batch_rigid_transform, blend_shapes
from config import cfg
from utils.transforms import get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from utils.smplx import smplx

from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer

class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()
    
    def forward(self, gaussian_assets, img_shape, cam_param, bg=torch.ones((3)).float().cuda()):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen)
        #render_img, radius, render_others = rasterizer(
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None
        )
        # Debug
        # import numpy as np
        # np.save('gs_mask.npy', render_mask.detach().cpu().numpy())

        out = {'img': render_img, 
               'depth_map': render_depthmap, 
               'mask': render_mask, 
               'mean_2d': mean_2d, 
               # 'is_vis': radius > 0, 
               'radius': radius}
        return out
    
class HumanGaussian(nn.Module):
    def __init__(self):
        super(HumanGaussian, self).__init__()
        self.smplx_layer = smplx.create(
            cfg.human_model_path,
            "smplx",
            gender="neutral",
            num_betas=smpl_x.shape_param_dim,
            num_expression_coeffs=smpl_x.expr_param_dim,
            use_pca=False,
            use_face_contour=False,
            flat_hand_mean=True,
            **smpl_x.layer_arg,
        ).cuda()  # follow LHM
        
    def init(self, gaussian_attr): 
        query_xyz = gaussian_attr['query_points']  # (40k, 3), REC-MV pose, no id info, jaw_zero_pose == True, transl is same w/ smplx_layer.v_template
        rotation = gaussian_attr['rotation']
        opacity = gaussian_attr['opacity']
        rgb = gaussian_attr['shs']
        scale = gaussian_attr['scaling']
        
        self.sample_points_num = query_xyz.shape[0]  # points are sampled from smplx_layer.v_template, following strategy of LHM (SMPLXVoxelMeshModel in smpl_x_voxel_dense_sampling.py)
        sample_points_path = osp.join(cfg.sample_points_path, f"1_{self.sample_points_num}.ply")
        sample_pts, _ = load_ply(sample_points_path)
        template_verts = self.smplx_layer.v_template
        template_verts_upsampled = smpl_x.upsample_mesh(template_verts)
        
        nn_vertex_idxs = knn_points(
            sample_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0),
            K=1,
            return_nn=True,
        ).idx
        query_idx = nn_vertex_idxs.squeeze(0, -1)
        nn_vertex_idxs_upsampled = knn_points(
            sample_pts.unsqueeze(0).cuda(),
            template_verts_upsampled.unsqueeze(0),
            K=1,
            return_nn=True,
        ).idx
        query_idx_upsampled = nn_vertex_idxs_upsampled.squeeze(0,-1)
        
        self.query_xyz = query_xyz.cuda()
        
        is_rhand, is_lhand, is_face = torch.zeros((smpl_x.vertex_num)).float().cuda(), torch.zeros((smpl_x.vertex_num)).float().cuda(), torch.zeros((smpl_x.vertex_num)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[smpl_x.face_vertex_idx] = 1.0, 1.0, 1.0
        is_rhand, is_lhand, is_face = is_rhand[query_idx], is_lhand[query_idx], is_face[query_idx]
        
        voxel_skinning_weight, voxel_bbox = self.voxel_skinning_init(voxel_size=192)
        skinning_weight = self.smplx_layer.lbs_weights.float()
        skinning_weight = skinning_weight[query_idx,:]
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num, 3*smpl_x.expr_param_dim)
        expr_dirs = expr_dirs[query_idx,:]
        shape_dirs = self.smplx_layer.shapedirs.view(smpl_x.vertex_num, 3*smpl_x.shape_param_dim)
        shape_dirs = shape_dirs[query_idx,:]
        
        self.register_buffer('voxel_bbox', voxel_bbox)
        self.register_buffer('voxel_ws', voxel_skinning_weight)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('expr_dirs', expr_dirs)
        self.register_buffer('shape_dirs', shape_dirs)
        self.register_buffer('query_idx', query_idx)
        self.register_buffer('query_idx_upsampled', query_idx_upsampled)  # for lapreg
        
        self.register_buffer('shape_param', smpl_x.shape_param.float().cuda())
        self.register_buffer('face_offset', smpl_x.face_offset.float().cuda())
        self.register_buffer('joint_offset', smpl_x.joint_offset.float().cuda())
        self.is_rhand = is_rhand
        self.is_lhand = is_lhand
        self.is_face = is_face
        
        self.mean_offset = gaussian_attr['offset_xyz']
        
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        
        rgb = gaussian_attr['shs'].squeeze(1)  # gaussian_attr['use_rgb'] == True, (40k,1,3)->(40k,3)
        rgb = torch.clamp(rgb, min=1e-4, max=1-1e-4)
        rgb = torch.log(rgb / (1 - rgb)) # inverse of sigmoid
        self.rgb = rgb
    
    def get_rec_pose_human(self, jaw_zero_pose, use_id_info, return_joint=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        rec_body_pose = smpl_x.rec_body_pose.view(1,-1).cuda() # REC-MV pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            face_offset = self.face_offset[None,:,:]
            joint_offset = self.joint_offset[None,:,:]
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=rec_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        vert_neutral_pose = output.vertices[0] # REC-MV pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint['num'],:] # REC-MV pose human
        if not return_joint:
            return vert_neutral_pose
        else:
            return vert_neutral_pose, joint_neutral_pose
    
    def voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):  # from LHM
        mesh_neutral_pose_wo_upsample = self.get_rec_pose_human(
            jaw_zero_pose=True,
            use_id_info=False
        )
        template_verts = mesh_neutral_pose_wo_upsample.squeeze(0)
        
        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)
        
        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2
        
        voxel_flat = torch.load(
            osp.join(cfg.voxel_grid_path, f"voxel_{voxel_size}.pth"),
            map_location='cuda',
        )
        
        _, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        # [W H D 55]->[55, D, H, W]
        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)
        
        
        return voxel_grid, mini_size_bbox
    
    def get_zero_pose_human(self, return_vert=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint['part_idx']['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = self.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint['num'],:] # zero pose human
        if not return_vert:
            return joint_zero_pose
        else: 
            vert_zero_pose = output.vertices[0] # zero pose human
            vert_zero_pose_upsampled = smpl_x.upsample_mesh(vert_zero_pose) # zero pose human
            return vert_zero_pose_upsampled, vert_zero_pose, joint_zero_pose
        
    def query_voxel_skinning_weights(self, vs):  # from LHM
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]
        # mapping to [-1, 1] **3
        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),  # 1 C D H W
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws  # [B N C]
    
    def get_transform_mat_vertex(self, transform_mat_joint, points, fix_mask):  # mask shape: (1,vertex_num)
        batch_size = transform_mat_joint.shape[0]

        query_skinning = self.query_voxel_skinning_weights(points)
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_skinning[fix_mask] = skinning_weight[fix_mask]

        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.reshape(batch_size, smpl_x.joint['num'], 16),
        ).view(batch_size, self.sample_points_num, 4, 4)
        return transform_mat_vertex
    
    def lbs(self, xyz, transform_mat_vertex):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:,:1])), dim=2)  # (B,N,4)
        xyz = torch.matmul(transform_mat_vertex, xyz[:,:,:,None]).view(-1,self.sample_points_num,4)[:,:,:3]
        return xyz

    def forward(self, smplx_param, cam_param, gaussian_posed=None): 
        mean_offset = self.mean_offset
        scale = torch.exp(self.scale)
        rotation = self.rotation
        opacity = self.opacity
        rgb = torch.sigmoid(self.rgb)
        mean_3d = self.query_xyz + mean_offset  # (40k, 3) REC-MV pose
        
        # Animating process follows LHM (transform_to_posed_verts_from_neutral_pose in LHM/LHM/models/rendering/smpl_x_voxel_dense_sampling.py)
        '''
        PERSONA: 大 pose -> zero pose -> image pose
        also LHM: REC-MV pose -> zero pose -> image pose
        LHM model, we choose, uses REC-MV as canonical pose, see LHM/configs/inference/human-lrm-500M.yaml
        
        There's no shape blend in query points (get_query_points in smpl_x_voxel_dense_sampling.py)
        Shape blend shape during animating phase (add on zero posed mean3d)
        '''
        
        smplx_expr_offset = 0.0
        mean_3d = mean_3d + smplx_expr_offset # REC-MV pose.
        
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        mask = ((self.is_rhand + self.is_lhand + self.is_face) > 0)[None,:]
        
        # 1. REC-MV pose pose -> zero pose
        zero_pose = torch.zeros((1,3)).float().cuda()
        rec_body_pose = smpl_x.rec_body_pose.view(len(smpl_x.joint['part_idx']['body'])-1,3).cuda() # REC-MV pose
        jaw_pose = torch.zeros((1,3)).float().cuda()
        zero_hand_pose = torch.zeros((len(smpl_x.joint['part_idx']['lhand']),3)).float().cuda()
        neutral_pose = torch.cat((zero_pose, rec_body_pose, jaw_pose, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) # follow smpl_x.joint['name']
        neutral_pose = axis_angle_to_matrix(neutral_pose)
        joint_zero_pose = self.get_zero_pose_human()
        _, transform_mat_joint_1 = batch_rigid_transform(neutral_pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_1 = torch.inverse(transform_mat_joint_1)  # 1,N,4,4
        
        # compute vertices-LBS function
        transform_mat_vertex_1 = self.get_transform_mat_vertex(transform_mat_joint_1, mean_3d[None,:,:], mask)  # jane: non-continuous
        '''mean_3d = self.lbs(mean_3d, transform_mat_vertex_1, torch.zeros_like(smplx_param["trans"]))'''
        mean_3d_zero_pose = self.lbs(mean_3d[None,:,:], transform_mat_vertex_1)[0]
        
        # blend_shape offset
        blend_shape_offset = blend_shapes(self.shape_param[None,:], self.shape_dirs.view(self.sample_points_num, -1, smpl_x.shape_param_dim))
        mean_3d_zero_pose = mean_3d_zero_pose + blend_shape_offset[0]
        
        # get transformation matrix of the nearest vertex and perform lbs
        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(-1, 1, 3)
        body_pose = smplx_param['body_pose'].view(-1, len(smpl_x.joint['part_idx']['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(-1, 1, 3)
        leye_pose = smplx_param['leye_pose'].view(-1, 1, 3)
        reye_pose = smplx_param['reye_pose'].view(-1, 1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(-1, len(smpl_x.joint['part_idx']['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(-1, len(smpl_x.joint['part_idx']['rhand']),3)
        trans = smplx_param['trans'].view(-1, 1, 3)

        batch_size = root_pose.shape[0]

        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose), dim=1) # follow smpl_x.joint['name']
        pose = axis_angle_to_matrix(pose)
        
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None, :,:,:], joint_zero_pose[None,:,:].repeat(batch_size, 1, 1), self.smplx_layer.parents)
        translation_mat = torch.zeros((batch_size, smpl_x.joint['num'],4,4)).float().cuda()
        translation_mat[:,:,:3,3] = trans # global translation
        transform_mat_joint_2 += translation_mat
        
        transform_mat_vertex_2 = self.get_transform_mat_vertex(transform_mat_joint_2, 
                                                               mean_3d[None,:,:].repeat(batch_size, 1, 1), 
                                                               mask[:, :, None].repeat(batch_size, 1, transform_mat_joint_2.shape[1]))
        mean_3d = self.lbs(mean_3d_zero_pose[None,:,:].repeat(batch_size, 1, 1), transform_mat_vertex_2)
        
        # camera coordinate system -> world coordinate system
        mean_3d = torch.matmul((mean_3d - cam_param['t'][:,None,:]), torch.inverse(cam_param['R']))

        # rotation appearance from canonical space to view_posed
        transform_matrix =torch.matmul(transform_mat_vertex_2, transform_mat_vertex_1)
        num_view, N, _, _ = transform_matrix.shape
        transform_rotation = transform_matrix[:, :, :3, :3]

        rigid_rotation_matrix = torch.nn.functional.normalize(
            matrix_to_quaternion(transform_rotation), dim=-1
        )
        

        # inference constrain
        # I = matrix_to_quaternion(torch.eye(3)).cuda()
        # is_constrain_body = self.smplx_model.is_constrain_body
        # rigid_rotation_matrix[:, is_constrain_body] = I
        # rotation_neutral_pose = rotation.unsqueeze(0).repeat(num_view, 1, 1)

        # QUATERNION MULTIPLY
        rotation = quaternion_multiply(
            rigid_rotation_matrix, rotation.unsqueeze(0).repeat(num_view, 1, 1)
        )

        opacity = opacity[None, :, :].repeat(batch_size, 1, 1)
        scale = scale[None, :, :].repeat(batch_size, 1, 1)
        rotation = rotation
        rgb = rgb[None, :, :].repeat(batch_size, 1, 1)
        asset = {
            'mean_3d': mean_3d, 
            'opacity': opacity,
            'scale': scale,
            'rotation': rotation,
            'rgb': rgb
        }
        
        offsets = {
            'mean_offset': mean_offset
        }

        return asset, offsets
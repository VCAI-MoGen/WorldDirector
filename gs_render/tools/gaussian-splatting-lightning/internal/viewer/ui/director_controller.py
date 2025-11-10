import sys, os, time
import json
import math

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import itertools
import torch.multiprocessing as mp
from dataclasses import fields

from typing import Tuple

import imageio

from internal.cameras.cameras import BatchCameras, Cameras, Camera, CamerasInterface

sys.path.insert(0, os.path.join('../../../HSI/fitting', 'tools'))
sys.path.insert(0, os.path.join('../../../HSI/fitting', 'main'))
sys.path.insert(0, os.path.join('../../../HSI/fitting', 'common'))
sys.path.insert(0, '../DIRECTOR')

from utils.smpl_x import smpl_x

from remote_run_director import RemoteDirector
from src.datasets.modalities.trajectory_dataset import TrajectoryDataset
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d
from remote_run_director import encode_char_traj
from remote_extract_masks_from_traj_video import remote_extract_video


class DirectorController:
    def __init__(self, sc_panel):
        self.is_generating_cam_params = False
        self.sc_panel = sc_panel
    
    def generate_cam_params(self, client):
        if self.is_generating_cam_params:
            print("Already Generating")
            return

        self.diffuser_client = client
        self.is_generating_cam_params = True

        start_time = time.time()
        
        self.init_director()

        out, padding_mask = self.director.remote_run_director(camera_start_pose=self.cam_start_pose)
        print("Remote Generating Done!")

        if self.director.control_cfg.visualize_output:
            self.visualize_cam_output(out['raw_gen_samples'], padding_mask[None])

        # ===== Rendering Video =====
        self.render_video(client)
        
        self.is_generating_cam_params = False
        print("It took", time.time() - start_time)
    
    def generate_cam_params_and_video(self, client):
        if self.is_generating_cam_params:
            print("Already Generating")
            return

        self.diffuser_client = client
        self.is_generating_cam_params = True

        start_time = time.time()
        
        self.init_director()

        out, padding_mask = self.director.remote_run_director(camera_start_pose=self.cam_start_pose)
        print("Remote Generating Done!")

        if self.director.control_cfg.visualize_output:
            self.visualize_cam_output(out['raw_gen_samples'], padding_mask[None])

        # ===== Rendering Video =====
        self.render_video(client)

        # ===== Rendering Human Video =====
        ctrl = self.director.control_cfg.custom_guidance_ctrl
        video_name = f'{self.director.control_cfg.video_name_prefix}'
        for var_name in ctrl.keys():
            if ctrl[var_name] is not None and ctrl[var_name]:
                video_name += f"_{var_name}_{ctrl[var_name]}"
        video_name += '.mp4'

        remote_extract_video(self.get_root_path(), human_render_video_name=video_name)

        self.is_generating_cam_params = False
        print("It took", time.time() - start_time)

    def render_video(self, client, cam_param_dir_path=None, scene_render_path=None, render_video_path=None):
        self.set_path('detailed')
        if cam_param_dir_path is None:
            cam_param_dir_path = self.get_cam_param_path()
        cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
        cameras = self.get_cameras_from_json(cam_params_path)
        
        # Render Video
        self.sc_panel.server.scene.set_global_visibility(False)
        image = self.render(cameras, render_mode="rgb") # (N, H, W, 3)
        self.sc_panel.server.scene.set_global_visibility(True)

        print("Converting to video...")
        if scene_render_path is None:
            scene_render_path = self.get_scene_render_path()
        for i in tqdm(range(len(cameras))):
            image_path = os.path.join(scene_render_path, f"{i}.png")
            imageio.imwrite(image_path, image[i])

        if render_video_path is None:
            render_video_path = os.path.join(self.get_data_path(), 'scene_render_video.mp4')
        imageio.mimsave(render_video_path, image, fps=16)
        print(f"Video saved in {render_video_path}")

        self.update_camera(client, cameras[0])

    # region ========== Interface ==========
    def set_path(self, split):
        self.sc_panel.set_path(split)

    def get_root_path(self):
        return self.sc_panel.root_path
    
    def get_cam_param_path(self):
        return self.sc_panel.cam_param_path
    
    def get_scene_render_path(self):
        return self.sc_panel.scene_render_path
    
    def get_data_path(self):
        return self.sc_panel.data_path
    
    def get_smplx_params_path(self):
        return self.sc_panel.smplx_params_path

    def get_smplx_scale_path(self):
        return self.sc_panel.smplx_scale_path

    def get_smplx_shape_path(self):
        return self.sc_panel.smplx_shape_path

    def get_cameras_from_json(self, path):
        return self.sc_panel.get_cameras_from_json(path)

    def get_priors(self):
        return self.sc_panel.get_priors()

    def get_world_up_vector(self):
        return self.sc_panel.world_up_vector
    
    def get_render_wh(self):
        return self.sc_panel.render_wh
    
    def get_K(self, fov: float, img_wh: Tuple[int, int]):
        return self.sc_panel.get_K(fov=fov, img_wh=img_wh)

    def set_rgb_render_mode(self):
        self.sc_panel.set_rgb_render_mode()

    def update_camera(self, client, camera: Camera):
        return self.sc_panel.update_camera(client=client, camera=camera)
    
    def render(self, cameras: Cameras, render_mode="rgb", optimizable=False):
        return self.sc_panel.render(cameras=cameras, render_mode=render_mode, optimizable=optimizable)

    def render_smplx_at_3d_position(self, points_world, R_w2c, t_w2c, focal, princpt, img, smplx_scale=1.0, smplx_up_vector=torch.tensor([0.0, 0.0, 1.0])):
        return self.sc_panel.render_smplx_at_3d_position(points_world, R_w2c, t_w2c, focal, princpt, img, smplx_scale=smplx_scale, smplx_up_vector=smplx_up_vector)
    # endregion

    # ========== Custom Guidance ==========

    def init_director(self):
        self.set_path('traj')

        print("Remote Generating Cam Traj")
        self.director = RemoteDirector()

        self.director.diffuser.register_cfg_callback(self.get_scene_aware_grad)
        self.director.diffuser.register_full_step_guidance_callback(self.get_human_aware_grad)
        self.director.diffuser.register_roll_replace_callback(self.get_roll_fixed_feat)

        self.get_matrix = self.director.diffuser.get_matrix
        self.get_feature = self.director.dataset.get_feature

        # self.prev_loss = None

        self.world_head_positions, self.world_heading_vectors, self.meshes = self.get_world_head_poses()

        self.world_upper_body_positions, self.world_upper_body_heading_vectors = self.get_world_upper_body_poses()

        self.cam_start_pose = None
        if self.director.control_cfg.start_pose_replace:
            self.cam_start_pose = self.get_camera_start_pose_prior()

        self.cam_end_pose = None
        if self.director.control_cfg.end_pose_replace:
            self.cam_end_pose = self.get_camera_end_pose_prior()
        
        # Visualize world head positions
        # self.diffuser_client.scene.add_point_cloud(
        #     name="head",
        #     points=np.array(self.world_head_positions.cpu()),       # (N, 3)
        #     colors=(1.0, 0.0, 1.0),     # (N, 3) in [0, 1]
        #     point_size=0.01
        # )

        # self.diffuser_client.scene.add_point_cloud(
        #     name="heading",
        #     points=np.array((self.world_head_positions + 3 * self.world_heading_vectors).cpu()),       # (N, 3)
        #     colors=(0.0, 1.0, 1.0),     # (N, 3) in [0, 1]
        #     point_size=0.01
        # )

    def get_roll_fixed_feat(self, cam_feat, batch_valid_mask):
        device = cam_feat.device
        dtype = cam_feat.dtype

        R_batch_world, t_batch_world = traj_feat_to_world_space_matrix(cam_feat,
                                                    self.get_matrix,
                                                    self.director.R_scene_up_to_director_up,
                                                    self.director.char_start,
                                                    self.director.smplx_scale)

        world_up_vector = torch.tensor(self.get_world_up_vector(), device=device, dtype=dtype)

        R_batch_world_new = roll_axis_up_vector_aligned_rotation(R_batch_world, world_up_vector)

        B, T, _, _ = R_batch_world.shape
        new_world_pose = torch.zeros((B, T, 4, 4), device=device, dtype=dtype)
        new_world_pose[..., :3, :3] = R_batch_world_new
        new_world_pose[..., :3, 3:4] = t_batch_world
        new_world_pose[..., 3, 3] = 1

        new_feat = world_space_matrix_to_traj_feat(new_world_pose,
                                                self.get_feature,
                                                self.director.R_scene_up_to_director_up.to(device),
                                                self.director.char_start.to(device),
                                                self.director.smplx_scale)

        self.debug_cam_change(cam_feat, new_feat, self.director.K, 5)

        return new_feat

    def get_human_aware_loss(self, cam_feat, batch_valid_mask):
        device = cam_feat.device
        dtype = cam_feat.dtype
        
        smplx_scale, smplx_up_vector = self.get_priors()
        smplx_scale = self.director.smplx_scale
        smplx_up_vector = torch.tensor(self.get_world_up_vector()).to(device)

        # World Space
        R_c2w, t_c2w = traj_feat_to_world_space_matrix(
                                        cam_feat,
                                        self.get_matrix,
                                        self.director.R_scene_up_to_director_up,
                                        self.director.char_start,
                                        self.director.smplx_scale,
                                    )
        
        t_c2w = t_c2w.squeeze(-1)

        batch_losses = []
        total_loss_infos = []
        ctrl = self.director.control_cfg.custom_guidance_ctrl

        for valid_mask, R, t in zip(batch_valid_mask, R_c2w, t_c2w):
            batch_loss = 0
            loss_info = {}

            valid_mask = valid_mask.bool() # (N)

            R = R[valid_mask] # (N, 3, 3)
            t = t[valid_mask] # (N, 3)

            if ctrl.look_target is not None and ctrl.look_target > 0.0:
                camera_to_targets = normalize(self.world_upper_body_positions.to(device) - t).detach() # (N, 3)
                camera_forwards = normalize(R[:, :, 2]) # (N, 3)
                loss_camera_look_head = ctrl.look_target * margin_range_loss(dot(camera_to_targets, camera_forwards), \
                                                                    lower=ctrl.look_threshold, \
                                                                    upper=1, \
                                                                    margin=0).mean()

                batch_loss += loss_camera_look_head
                loss_info["loss_camera_look_head"] = f"{loss_camera_look_head:.2f}"

            if ctrl.head_look_camera is not None and ctrl.head_look_camera > 0.0:
                target_to_cameras = normalize(t - self.world_head_positions.to(device)) # (N, 3)
                heading_vectors = normalize(self.world_heading_vectors.to(device))
                loss_head_look_camera = ctrl.head_look_camera * margin_range_loss(dot(target_to_cameras, heading_vectors), \
                                                                    lower=ctrl.head_threshold, \
                                                                    upper=1, \
                                                                    margin=0).mean()
                batch_loss += loss_head_look_camera
                loss_info["loss_head_look_camera"] = f"{loss_head_look_camera:.2f}"
                         
            if ctrl.look_distance is not None and ctrl.look_distance > 0.0:
                camera_to_targets = self.world_upper_body_positions.to(device) - t # (N, 3)
                loss_target_cam_distance = ctrl.look_distance * margin_range_loss(torch.linalg.norm(camera_to_targets, dim=-1), \
                                                                    lower=ctrl.dist_threshold/2*smplx_scale, \
                                                                    upper=ctrl.dist_threshold*smplx_scale, \
                                                                    margin=0).mean()
                batch_loss += loss_target_cam_distance
                loss_info["loss_target_cam_distance"] = f"{loss_target_cam_distance:.2f}"

            if ctrl.roll_constraint is not None and ctrl.roll_constraint > 0.0:
            
                _, angles = roll_axis_and_angle(R[None], torch.tensor(self.get_world_up_vector()).float().to(device))
                
                loss_roll_constraint = ctrl.roll_constraint * margin_range_loss(angles, \
                                                                lower=torch.deg2rad(torch.zeros_like(angles)).to(device),\
                                                                upper=torch.deg2rad(torch.zeros_like(angles)).to(device),\
                                                                margin=0.0).mean()

                loss_info["loss_roll_constraint"] = loss_roll_constraint

            # Discarded (included in look loss)
            if ctrl.cam_position_constraint is not None and ctrl.cam_position_constraint > 0.0:
                heading_vectors = normalize(self.world_heading_vectors.to(device))
                head_position = self.world_head_positions.to(device)
                target_camera_position = head_position + heading_vectors * 1.5 * smplx_scale
                target_to_camera_distance = torch.linalg.norm(target_camera_position - t, dim=-1)

                loss_cam_position = ctrl.cam_position_constraint * margin_range_loss(target_to_camera_distance, \
                                                                    lower=0, \
                                                                    upper=smplx_scale, \
                                                                    margin=0).mean()
                
                batch_loss += loss_cam_position
                loss_info["loss_cam_position"] = f"{loss_cam_position:.2f}"

            if self.cam_end_pose is not None:
                # world_pose = rot_trans_to_matrix(R, t)
                loss_end_pose = ctrl.end_pose_weight * torch.abs(t[-1] - self.cam_end_pose.squeeze()[:3, 3]).mean() # translation only for now

                batch_loss += loss_end_pose
                loss_info["loss_end_pose"] = f"{loss_end_pose:.2f}"

            if len(loss_info.keys()) > 0:
                # Relative Loss Scaling
                # 전 step과 비교해서 loss 적게 줄어든 애는 강화 많이 줄어든 애는 약화
                # converge aware이나 loss 절대량에 의한 scaling도 필요할지도..
                # if self.prev_loss is not None:
                #     delta_loss = 0
                #     num_loss_type = 0
                #     for key, val in loss_info.items():
                #         if key in self.prev_loss.keys():
                #             delta_loss += self.prev_loss[key] - val
                #             num_loss_type += 1
                #     mean_delta_loss = delta_loss / num_loss_type

                #     for key, val in loss_info.items():
                #         if key in self.prev_loss.keys():
                #             loss_info[key] *= (mean_delta_loss / (self.prev_loss[key] - val))

                # self.prev_loss = loss_info

                batch_losses.append(batch_loss)
                total_loss_infos.append(loss_info)
                print(loss_info)


        total_loss = None
        if len(batch_losses) > 0:
            total_loss = torch.stack(batch_losses) # batch independent loss

        return total_loss, total_loss_infos

    def get_human_aware_grad(self, cam_feat, batch_valid_mask):
        cam_feat_copy = cam_feat.detach().clone().requires_grad_(True)

        total_loss, batch_loss_infos = self.get_human_aware_loss(cam_feat_copy, batch_valid_mask)

        print("Human Total Loss :", total_loss)
        
        grad = None
        if total_loss is not None:
            grad = torch.autograd.grad(total_loss, cam_feat_copy, create_graph=False)[0]

        self.set_rgb_render_mode()

        # ===== Visualize One Batch =====

        ctrl = self.director.control_cfg

        # self.visualize_cam_output(next_cam_feat)
        # breakpoint()

        if grad is not None and ctrl.custom_guidance_ctrl.debug:
            next_cam_feat = cam_feat - ctrl.custom_guidance_weight * grad

            self.debug_look_loss(cam_feat, next_cam_feat, batch_valid_mask, 5)

        return grad

    def get_scene_aware_loss(self, cam_feat, batch_valid_mask):
        device = 'cuda:0'
        dtype = cam_feat.dtype
        
        smplx_scale, smplx_up_vector = self.get_priors()
        smplx_scale = self.director.smplx_scale
        smplx_up_vector = torch.tensor(self.get_world_up_vector()).to(smplx_up_vector)
        
        data_type = 'traj'

        cameras = None
        img = None
        bbox_masks = None

        maximum_mask_ratio = 0.3
        minimum_mask_ratio = 0.15
        camera_human_elevation_min_degree = 50.0
        camera_human_elevation_max_degree = 60.0

        roll_degree_range = 5.0
        pixel_margin = 50.0

        if smplx_up_vector is not None:
            smplx_up_vector = torch.tensor(smplx_up_vector).to(device)
        
        # Get arbitrary point on plane
        if self.get_world_up_vector() is None:
            vx, vy, vz = path_vector
            x, y = 1.0, 0.0
            if vz != 0:
                z = -(vx*x + vy*y) / vz
            else:
                z = 1.0
            up_vector = torch.tensor([x, y, z]).float().to(device)
            up_vector = up_vector / torch.linalg.norm(up_vector)
        else:
            up_vector = torch.tensor(self.get_world_up_vector()).float().to(device)
            up_vector = up_vector / torch.linalg.norm(up_vector)

        # Camera Space
        R_w2c, t_w2c = traj_feat_to_camera_space_matrix(
                                        cam_feat,
                                        self.get_matrix,
                                        self.director.R_scene_up_to_director_up,
                                        self.director.char_start,
                                        smplx_scale,
                                    )
        
        R_batch = R_w2c.to(device)
        t_batch = t_w2c.to(device)
        K = torch.tensor(self.get_K(np.radians(75), self.get_render_wh()), device=device).float()

        batch_cameras = build_cameras(R_batch, t_batch, K)

        # cond_sample 받아와서 decoding하는건 비효율적이라 raw data 가져옴
        points_world = torch.tensor(self.director.raw_char_traj, device=device, dtype=dtype) # (N, 3)

        batch_losses = []
        ctrl = self.director.control_cfg.custom_guidance_ctrl
        for full_cams, valid_mask, R, t in zip(batch_cameras, batch_valid_mask, R_batch, t_batch):
            N = valid_mask.sum().int().item()
            batch_loss = 0
            loss_info = {}

            valid_mask = valid_mask.bool() # (N)
            cameras = [full_cams[i] for i in range(N)] # (N)

            R = R[valid_mask] # (N, 3, 3)
            t = t[valid_mask] # (N, 3)
            fx = torch.tensor([cam.fx for cam in cameras])
            fy = torch.tensor([cam.fy for cam in cameras])
            cx = torch.tensor([cam.cx for cam in cameras])
            cy = torch.tensor([cam.cy for cam in cameras])
            width = torch.tensor([cam.width for cam in cameras])
            height = torch.tensor([cam.height for cam in cameras])

            # world space camera pose
            R_c2w = R.transpose(-1, -2)
            t_c2w = torch.bmm(-R.transpose(-1, -2), t.unsqueeze(-1)).squeeze(-1)

            if ctrl.cam_over_point is not None and ctrl.cam_over_point > 0.0:
                # 지금은 batch 하나라서 여기서 하고 batch로 하려면 밖에서 padding해서 사용
                points_to_camera = normalize(t_c2w - points_world) # (N, 3)
                loss_camera_over_points =  ctrl.cam_over_point * margin_range_loss(torch.sum(points_to_camera * smplx_up_vector.unsqueeze(0), dim=-1), \
                                                                    lower=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_max_degree))).to(device), \
                                                                    upper=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_min_degree))).to(device), \
                                                                    margin=0).mean()
                batch_loss += loss_camera_over_points

                loss_info["loss_camera_over_points"] = f"{loss_camera_over_points.item():.2f}"

            if (ctrl.mask_in_image is not None or ctrl.mask_depth is not None) and \
                (ctrl.mask_in_image > 0.0 or ctrl.mask_depth > 0.0):

                scene_depth = self.render(cameras, render_mode='exp_depth', optimizable=True) # (N, H, W)

                focal_batch = torch.stack((fx, fy), dim=-1)
                princpt_batch = torch.stack((cx, cy), dim=-1)
                img = self.render(cameras, render_mode="rgb", optimizable=True) * 255

                _, human_depths, human_masks, bb = self.render_smplx_at_3d_position(points_world, R, t, focal_batch, princpt_batch, img, smplx_scale, smplx_up_vector)

                # human mask filtering
                target_depth = scene_depth.unsqueeze(-1)
                target_depth[human_masks <= 0] = 0
                human_depths[human_masks <= 0] = 0

                # human mask visible filter
                in_frame_threshold = 0 # 이론상 최대 면적의 몇퍼센트 만들어야함
                human_mask_in_frame = (human_masks[:, :, :, 0] > 0).sum(dim=(-1, -2)) > in_frame_threshold # (N)

                print("Visible Frame :", human_mask_in_frame.sum())

                # angles between points to camera vec and human up vec >= thresh
                if ctrl.mask_in_image is not None and ctrl.mask_in_image > 0.0:
                    loss_human_mask_in_image = ctrl.mask_in_image * scaled_margin_range_loss(bb, \
                                                                    lower=torch.zeros((N, 4)).to(device), \
                                                                    upper=torch.stack([width, height, width, height], dim=-1).to(device), \
                                                                    margin=0)[human_mask_in_frame].mean()
                    batch_loss += loss_human_mask_in_image

                    loss_info["loss_human_mask_in_image"] = f"{loss_human_mask_in_image.item():.2f}"
                
                if ctrl.mask_depth is not None and ctrl.mask_depth > 0.0:
                    loss_human_mask_visible = ctrl.mask_depth * margin_range_loss(human_depths, \
                                                                    lower=torch.zeros(human_depths.shape).to(device), \
                                                                    upper=target_depth, \
                                                                    margin=0.0)[human_mask_in_frame].mean()

                
                    batch_loss += loss_human_mask_visible

                    loss_info["loss_human_mask_visible"] = f"{loss_human_mask_visible.item():.5f}"




            if len(loss_info.keys()) > 0:
                batch_losses.append(batch_loss)
                print(loss_info)

        total_loss = None
        if len(batch_losses) > 0:
            total_loss = torch.stack(batch_losses) # batch independent loss

        return total_loss

    def get_scene_aware_grad(self, cam_feat, batch_valid_mask):
        """
        cam_feat : (B, P, T) - P : 6d + 3t = 9
        """
        cam_feat_copy = cam_feat.detach().clone().requires_grad_(True)

        total_loss = self.get_scene_aware_loss(cam_feat_copy, batch_valid_mask)
        print("Scene Total Loss :", total_loss)
        grad = torch.zeros_like(cam_feat, device=cam_feat.device, dtype=cam_feat.dtype)
        if total_loss is not None:
            grad = torch.autograd.grad(total_loss, cam_feat_copy, create_graph=False)[0]

        next_cam_feat = cam_feat - 1.0 * grad

        self.debug_cam_change(cam_feat, next_cam_feat, self.director.K, stop=81)

        return grad

    # ===== Camera Pose Priors =====

    def get_camera_start_pose_prior(self):
        return self.get_camera_pose_prior(0, distance=2*self.director.smplx_scale, target="upper_body")
    
    def get_camera_end_pose_prior(self):
        # Not feature since current setting is velocity based model
        return self.get_camera_pose_prior(-1, target="upper_body", distance=2*self.director.smplx_scale, is_feat=False)

    def get_camera_pose_prior(self, frame_idx=0, target="head", distance=None, is_feat=True):
        # smplx_scale, smplx_up_vector = self.get_priors()
        device = 'cpu'
        dtype = torch.float32

        if distance is None:
            distance = self.director.smplx_scale

        if target == "head":
            target_position = self.world_head_positions[frame_idx]
            heading_vector = self.world_heading_vectors[frame_idx]
        elif target == "upper_body":
            target_position = self.world_upper_body_positions[frame_idx]
            heading_vector = self.world_upper_body_heading_vectors[frame_idx]
        mesh = self.meshes[frame_idx]

        target_position = torch.tensor(target_position, device=device, dtype=dtype)
        heading_vector = torch.tensor(heading_vector, device=device, dtype=dtype)
        world_up_vector = torch.tensor(self.get_world_up_vector(), device=device, dtype=dtype)

        camera_trans = target_position + normalize(heading_vector) * distance + world_up_vector * 0.5 * self.director.smplx_scale
        # camera_trans = torch.zeros((1, 3), device=device, dtype=dtype)
        camera_forward = normalize(target_position - camera_trans).squeeze(0)
        camera_right = normalize(torch.cross(camera_forward, normalize(world_up_vector)))
        camera_up = normalize(torch.cross(camera_forward, camera_right))

        camera_rotation = torch.zeros((3, 3), device=device, dtype=dtype)
        camera_rotation[:, 0] = camera_right
        camera_rotation[:, 1] = camera_up
        camera_rotation[:, 2] = camera_forward
        
        camera_pose = torch.zeros((1, 4, 4), device=device, dtype=dtype)
        camera_pose[:, :3, :3] = camera_rotation
        camera_pose[:, :3, 3] = camera_trans
        camera_pose[:, 3, 3] = 1

        if self.director.control_cfg.visualize_input_frame:
            # Visualize Prior
            cam_space_R = camera_rotation.T
            cam_space_t = (-camera_rotation.T @ camera_trans.T).T

            camera = build_single_camera(cam_space_R, cam_space_t, self.director.K)
            self.update_camera(self.diffuser_client, camera)

            self.diffuser_client.scene.add_point_cloud(
                name="mesh",
                points=np.array(mesh.squeeze().cpu()),       # (N, 3)
                colors=(0.4, 0.5, 0.4),     # (N, 3) in [0, 1]
                point_size=0.005
            )

            time.sleep(1)

        cam_pose_output = camera_pose
        if is_feat:
            cam_pose_output = world_space_matrix_to_traj_feat(camera_pose.unsqueeze(0),
                                                            self.get_feature,
                                                            self.director.R_scene_up_to_director_up.to(device),
                                                            self.director.char_start.to(device),
                                                            self.director.smplx_scale).squeeze(0).transpose(-1, -2) # (1, P)
        
        return cam_pose_output

    def get_world_head_poses(self):
        device = 'cuda:0'
        self.set_path('traj')
        json_list = [f'{i}.json' for i in range(len(os.listdir(self.get_cam_param_path())))]
        head_position_list = []
        heading_vector_list = []
        mesh_list = []
        for filename in json_list:
            with open(os.path.join(self.get_cam_param_path(), filename), 'r') as f:
                cam_param = json.load(f)
                R_w2c = torch.tensor(cam_param['R'], device=device)
                t_w2c = torch.tensor(cam_param['t'], device=device)
                R_c2w = R_w2c.T
                t_c2w = -R_w2c.T @ t_w2c
            with open(self.get_smplx_scale_path(), 'r') as f:
                smplx_scale = torch.tensor(json.load(f), device=device)
            with open(os.path.join(self.get_smplx_params_path(), filename), 'r') as f:
                smplx_param = json.load(f)
            with open(self.get_smplx_shape_path(), 'r') as f:
                smplx_shape = json.load(f)
            smpl_x.layer['neutral'] = smpl_x.layer['neutral'].to(device)
            output = smpl_x.layer['neutral'](
                global_orient = torch.tensor(smplx_param['root_pose'], device=device)[None], \
                body_pose = torch.tensor(smplx_param['body_pose'], device=device)[None], \
                jaw_pose = torch.tensor(smplx_param['jaw_pose'], device=device)[None], \
                leye_pose = torch.tensor(smplx_param['leye_pose'], device=device)[None], \
                reye_pose = torch.tensor(smplx_param['reye_pose'], device=device)[None], \
                left_hand_pose = torch.tensor(smplx_param['lhand_pose'], device=device)[None], \
                right_hand_pose = torch.tensor(smplx_param['rhand_pose'], device=device)[None], \
                expression = torch.tensor(smplx_param['expr'], device=device)[None], \
                betas = torch.tensor(smplx_shape, device=device)[None], \
            )
            mesh_cam = output.vertices * smplx_scale[None, None, :]
            kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[None, None, :]
            
            kpt_cam += torch.tensor(smplx_param['trans'], device=device)[None, None, :]
            kpt_world = kpt_cam @ R_c2w.T + t_c2w[None, None, :]

            mesh_cam += torch.tensor(smplx_param['trans'], device=device)[None, None, :]
            mesh_world = mesh_cam @ R_c2w.T + t_c2w[None, None, :]

            left_ear_kpt = kpt_world[:, smpl_x.kpt['name'].index('L_Ear'), :]
            right_ear_kpt = kpt_world[:, smpl_x.kpt['name'].index('R_Ear'), :]
            left_eye_kpt = kpt_world[:, smpl_x.kpt['name'].index('L_Eye'), :]
            right_eye_kpt = kpt_world[:, smpl_x.kpt['name'].index('R_Eye'), :]
            

            head_position = (left_ear_kpt + right_ear_kpt) / 2
            heading_vector = (left_eye_kpt + right_eye_kpt) / 2 - head_position

            head_position_list.append(head_position.squeeze())
            heading_vector_list.append(heading_vector.squeeze())
            mesh_list.append(mesh_world.squeeze())

        head_positions = torch.stack(head_position_list)
        heading_vectors = torch.stack(heading_vector_list)
        meshes = torch.stack(mesh_list)

        return head_positions, heading_vectors, meshes

    def get_world_upper_body_poses(self):
        device = 'cuda:0'
        self.set_path('traj')
        json_list = [f'{i}.json' for i in range(len(os.listdir(self.get_cam_param_path())))]
        target_position_list = []
        heading_vector_list = []
        for filename in json_list:
            with open(os.path.join(self.get_cam_param_path(), filename), 'r') as f:
                cam_param = json.load(f)
                R_w2c = torch.tensor(cam_param['R'], device=device)
                t_w2c = torch.tensor(cam_param['t'], device=device)
                R_c2w = R_w2c.T
                t_c2w = -R_w2c.T @ t_w2c
            with open(self.get_smplx_scale_path(), 'r') as f:
                smplx_scale = torch.tensor(json.load(f), device=device)
            with open(os.path.join(self.get_smplx_params_path(), filename), 'r') as f:
                smplx_param = json.load(f)
            with open(self.get_smplx_shape_path(), 'r') as f:
                smplx_shape = json.load(f)
            smpl_x.layer['neutral'] = smpl_x.layer['neutral'].to(device)
            output = smpl_x.layer['neutral'](
                global_orient = torch.tensor(smplx_param['root_pose'], device=device)[None], \
                body_pose = torch.tensor(smplx_param['body_pose'], device=device)[None], \
                jaw_pose = torch.tensor(smplx_param['jaw_pose'], device=device)[None], \
                leye_pose = torch.tensor(smplx_param['leye_pose'], device=device)[None], \
                reye_pose = torch.tensor(smplx_param['reye_pose'], device=device)[None], \
                left_hand_pose = torch.tensor(smplx_param['lhand_pose'], device=device)[None], \
                right_hand_pose = torch.tensor(smplx_param['rhand_pose'], device=device)[None], \
                expression = torch.tensor(smplx_param['expr'], device=device)[None], \
                betas = torch.tensor(smplx_shape, device=device)[None], \
            )
            kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[None, None, :]
            
            kpt_cam += torch.tensor(smplx_param['trans'], device=device)[None, None, :]
            kpt_world = kpt_cam @ R_c2w.T + t_c2w[None, None, :]

            pelvis_kpt = kpt_world[:, smpl_x.kpt['name'].index('Pelvis'), :]
            l_shoulder_kpt = kpt_world[:, smpl_x.kpt['name'].index('L_Shoulder'), :]
            r_shoulder_kpt = kpt_world[:, smpl_x.kpt['name'].index('R_Shoulder'), :]

            upper_body_position = (pelvis_kpt + l_shoulder_kpt + r_shoulder_kpt) / 3

            heading_vector = normalize(torch.cross((l_shoulder_kpt - r_shoulder_kpt).squeeze(),
                                                    torch.tensor(self.get_world_up_vector()).float().to(r_shoulder_kpt.device)))

            target_position_list.append(upper_body_position.squeeze())
            heading_vector_list.append(heading_vector.squeeze())

        target_positions = torch.stack(target_position_list)
        heading_vectors = torch.stack(heading_vector_list)

        return target_positions, heading_vectors

    # ===== Debugging and Visualization =====

    def debug_cam_change(self, cam_feat, next_cam_feat, K, stop=1, data=None):
        device = cam_feat.device
        dtype = cam_feat.dtype
        ctrl = self.director.control_cfg.custom_guidance_ctrl

        if not ctrl.debug:
            return

        cam_feat_copy = cam_feat.detach().clone()
        next_cam_feat_copy = next_cam_feat.detach().clone()

        R_batch, t_batch = traj_feat_to_camera_space_matrix(cam_feat_copy.to(device),
                                                self.get_matrix,
                                                self.director.R_scene_up_to_director_up,
                                                self.director.char_start,
                                                self.director.smplx_scale)

        R_batch_next, t_batch_next = traj_feat_to_camera_space_matrix(next_cam_feat_copy.to(device),
                                                self.get_matrix,
                                                self.director.R_scene_up_to_director_up,
                                                self.director.char_start,
                                                self.director.smplx_scale)

        cur_cameras = build_cameras(R_batch, t_batch, K)
        next_cameras = build_cameras(R_batch_next, t_batch_next, K)

        for i, (camera, next_camera) in enumerate(itertools.islice(zip(cur_cameras[0], next_cameras[0]), stop)):
            if data is not None:
                print("prev:", data[0][i])
                print("next:", data[1][i])

            while True:
                self.update_camera(self.diffuser_client, camera)
                control = input("press any button to see updated scene")
                self.update_camera(self.diffuser_client, next_camera)
                control = input("look again? (y/n)")
                if control != "y":
                    break

    def debug_look_loss(self, cam_feat, next_cam_feat, batch_valid_mask, stop=1, data=None):
        ctrl = self.director.control_cfg.custom_guidance_ctrl
        device = cam_feat.device

        R_world, t_world = traj_feat_to_world_space_matrix(
                                    cam_feat,
                                    self.get_matrix,
                                    self.director.R_scene_up_to_director_up,
                                    self.director.char_start,
                                    self.director.smplx_scale,
                                )

        R_world = R_world.squeeze()[batch_valid_mask[0].bool()]
        t_world = t_world.squeeze()[batch_valid_mask[0].bool()]

        R_world_next, t_world_next = traj_feat_to_world_space_matrix(
                                    next_cam_feat,
                                    self.get_matrix,
                                    self.director.R_scene_up_to_director_up,
                                    self.director.char_start,
                                    self.director.smplx_scale,
                                )

        R_world_next = R_world_next.squeeze()[batch_valid_mask[0].bool()]
        t_world_next = t_world_next.squeeze()[batch_valid_mask[0].bool()]

        camera_to_targets = normalize(self.world_head_positions.to(device) - t_world) # (N, 3)
        camera_forwards = normalize(R_world[:, :, 2]) # (N, 3)
        loss_camera_look_head = ctrl.look_target * margin_range_loss(dot(camera_to_targets, camera_forwards), \
                                                            lower=1, \
                                                            upper=1, \
                                                            margin=0) # (N)

        camera_to_targets = normalize(self.world_head_positions.to(device) - t_world_next) # (N, 3)
        camera_forwards = normalize(R_world_next[:, :, 2]) # (N, 3)
        loss_camera_look_head_next = ctrl.look_target * margin_range_loss(dot(camera_to_targets, camera_forwards), \
                                                            lower=1, \
                                                            upper=1, \
                                                            margin=0) # (N)
        

        self.debug_cam_change(cam_feat, next_cam_feat, self.director.K, stop, (loss_camera_look_head, loss_camera_look_head_next))
    
    def debug_depth_loss(self, cam_feat, next_cam_feat, batch_valid_mask, stop=1, data=None):
        ctrl = self.director.control_cfg.custom_guidance_ctrl
        device = cam_feat.device

        R_world, t_world = traj_feat_to_world_space_matrix(
                                    cam_feat,
                                    self.get_matrix,
                                    self.director.R_scene_up_to_director_up,
                                    self.director.char_start,
                                    self.director.smplx_scale,
                                )

        R_world = R_world.squeeze()[batch_valid_mask[0].bool()]
        t_world = t_world.squeeze()[batch_valid_mask[0].bool()]

        R_world_next, t_world_next = traj_feat_to_world_space_matrix(
                                    next_cam_feat,
                                    self.get_matrix,
                                    self.director.R_scene_up_to_director_up,
                                    self.director.char_start,
                                    self.director.smplx_scale,
                                )

        R_world_next = R_world_next.squeeze()[batch_valid_mask[0].bool()]
        t_world_next = t_world_next.squeeze()[batch_valid_mask[0].bool()]

        camera_to_targets = normalize(self.world_head_positions.to(device) - t_world) # (N, 3)
        camera_forwards = normalize(R_world[:, :, 2]) # (N, 3)
        loss_camera_look_head = ctrl.look_target * margin_range_loss(dot(camera_to_targets, camera_forwards), \
                                                            lower=1, \
                                                            upper=1, \
                                                            margin=0) # (N)

        camera_to_targets = normalize(self.world_head_positions.to(device) - t_world_next) # (N, 3)
        camera_forwards = normalize(R_world_next[:, :, 2]) # (N, 3)
        loss_camera_look_head_next = ctrl.look_target * margin_range_loss(dot(camera_to_targets, camera_forwards), \
                                                            lower=1, \
                                                            upper=1, \
                                                            margin=0) # (N)
        

        self.debug_cam_change(cam_feat, next_cam_feat, self.director.K, stop, (loss_camera_look_head, loss_camera_look_head_next))

    def visualize_cam_output(self, cam_feat, batch_valid_mask):
        R_world, t_world = traj_feat_to_world_space_matrix(cam_feat,
                                                    self.get_matrix,
                                                    self.director.R_scene_up_to_director_up.to(cam_feat.device),
                                                    self.director.char_start.to(cam_feat.device),
                                                    self.director.smplx_scale)
        
        t_world = t_world.squeeze(-1)[0][batch_valid_mask[0].bool()]
        
        for head_pos, head_vec, mesh, cam_t in zip(self.world_head_positions, self.world_heading_vectors, self.meshes, t_world):
            time.sleep(0.1)
            # Visualize world head positions
            self.diffuser_client.scene.add_point_cloud(
                name="head",
                points=np.array(head_pos[None].cpu()),       # (N, 3)
                colors=(1.0, 0.0, 1.0),     # (N, 3) in [0, 1]
                point_size=0.01
            )

            self.diffuser_client.scene.add_point_cloud(
                name="heading",
                points=np.array((head_pos + 2 * head_vec)[None].cpu()),       # (N, 3)
                colors=(0.0, 1.0, 1.0),     # (N, 3) in [0, 1]
                point_size=0.01
            )

            self.server.scene.add_point_cloud(
                name='cam_output',
                points=np.array(cam_t[None].detach().numpy()),
                colors=(1.0, 0.0, 0.0),
                point_size=0.01,
                point_shape='circle'
            )

            self.diffuser_client.scene.add_point_cloud(
                name="mesh",
                points=np.array(mesh.cpu()),       # (N, 3)
                colors=(0.4, 0.5, 0.4),     # (N, 3) in [0, 1]
                point_size=0.005
            )

    def visualize_cam_output_in_one_step(self, cam_feat, batch_valid_mask, name="cam_output"):
        R_world, t_world = traj_feat_to_world_space_matrix(cam_feat,
                                                    self.get_matrix,
                                                    self.director.R_scene_up_to_director_up.to(cam_feat.device),
                                                    self.director.char_start.to(cam_feat.device),
                                                    self.director.smplx_scale)
        
        t_world = t_world.squeeze(-1)[0][batch_valid_mask[0].bool()]

        self.server.scene.add_point_cloud(
            name=name,
            points=np.array(t_world.squeeze().detach().numpy()),
            colors=(1.0, 0.0, 0.0),
            point_size=0.01,
            point_shape='circle'
        )

# ===== Utils =====

def roll_axis_and_angle(R, world_up=None, w2c=False, eps=1e-8):
    """
    Input:
        R: (B, T, 3, 3) rotation matrices (camera→world)
        world_up: (3,), default [0,1,0]
        w2c: True if R is world→camera (it will be transposed inside)
    Output:
        roll_axis: (B, T, 3)
        roll_angle: (B, T) [radians]
    """
    if world_up is None:
        world_up = torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype)
    if w2c:
        R = R.transpose(-1, -2)  # convert to camera→world

    cam_right   = R[..., :, 0]
    cam_up      = R[..., :, 1]  # viser camera up vector is -Y
    cam_forward = R[..., :, 2]  # roll axis

    roll_axis = normalize(cam_forward, eps)  # roll axis

    x_ref = torch.cross(roll_axis, world_up.expand_as(roll_axis), dim=-1)
    bad = x_ref.norm(dim=-1, keepdim=False) < 1e-6 # roll_axis, world_up이 비슷해지면 roll 정의 불가
    x_ref = normalize(x_ref, eps)
    y_ref = normalize(torch.cross(roll_axis, x_ref, dim=-1), eps)

    # signed angle
    cos_ = dot(y_ref, cam_up, keepdim=False)
    sin_ = dot(x_ref, cam_up, keepdim=False)
    theta = torch.atan2(sin_, cos_)
    theta = theta.masked_fill(bad, float('nan'))

    return roll_axis, theta

def roll_axis_up_vector_aligned_rotation(R, world_up=None, w2c=False, eps=1e-8):
    if world_up is None:
        world_up = torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype)
    if w2c:
        R = R.transpose(-1, -2)  # convert to camera→world

    cam_right   = R[..., :, 0]
    cam_up      = R[..., :, 1]  # viser camera up vector is -Y
    cam_forward = R[..., :, 2]  # roll axis

    roll_axis = normalize(cam_forward, eps)  # roll axis

    x_ref = torch.cross(roll_axis, world_up.expand_as(roll_axis), dim=-1)
    bad = x_ref.norm(dim=-1, keepdim=True) < 1e-6 # roll_axis, world_up이 비슷해지면 roll 정의 불가
    x_ref = normalize(x_ref, eps)
    y_ref = normalize(torch.cross(roll_axis, x_ref, dim=-1), eps)

    cam_R_c2w = torch.zeros_like(R, device=R.device, dtype=R.dtype)
    cam_R_c2w[..., :, 0] = x_ref
    cam_R_c2w[..., :, 1] = y_ref
    cam_R_c2w[..., :, 2] = roll_axis

    cam_R_c2w = cam_R_c2w.masked_fill(bad.unsqueeze(-1).expand_as(cam_R_c2w), float('nan'))

    return cam_R_c2w


def traj_feat_to_camera_space_matrix(batch_feat, get_matrix, R_scene_up_to_director_up, char_start, smplx_scale):

    R_c2w, t_c2w = traj_feat_to_world_space_matrix(batch_feat, get_matrix, R_scene_up_to_director_up, char_start, smplx_scale)

    # Scene World Space to Camera Space
    R_w2c = R_c2w.transpose(-1, -2) # (B, T, 3, 3)
    t_w2c = (-R_c2w.transpose(-1, -2) @ t_c2w.float()).squeeze(-1) # (B, T, 3)

    return R_w2c, t_w2c

def traj_feat_to_world_space_matrix(batch_feat, get_matrix, R_scene_up_to_director_up, char_start, smplx_scale):
    """
    Input:
        feat: (B, P, T)
    """
    # Latent Space to Traj Space
    batch_matrix = torch.stack([get_matrix(feat) for feat in batch_feat]) # (B, T, 4, 4)
    batch_R = batch_matrix[:, :, :3, :3] # (B, T, 3, 3)
    batch_t = batch_matrix[:, :, :3, 3:4] # (B, T, 3, 1)

    R_scene_up_to_director_up = torch.tensor(R_scene_up_to_director_up, device=batch_feat.device, dtype=batch_feat.dtype)
    char_start = torch.tensor(char_start, device=batch_feat.device, dtype=batch_feat.dtype)

    # Traj Space to Scene World Space
    R_c2w = R_scene_up_to_director_up.T @ batch_R
    t_c2w = (R_scene_up_to_director_up.T @ batch_t + char_start.unsqueeze(-1)) * smplx_scale

    return R_c2w, t_c2w

def world_space_matrix_to_traj_feat(camera_to_world, get_feature, R_scene_up_to_director_up, char_start, smplx_scale):
    """
    Input:
        camera_to_world: (B, T, 4, 4) - Scene World Space Matrix
    Output:
        batch_feat: (B, P, T)
    """
    B, T, _, _ = camera_to_world.shape
    device = camera_to_world.device
    dtype = camera_to_world.dtype

    R_scene_up_to_director_up = torch.tensor(R_scene_up_to_director_up, dtype=dtype, device=device)
    char_start = torch.tensor(char_start, dtype=dtype, device=device)

    # R_w2c = world_to_camera[:, :, :3, :3]
    # t_w2c = world_to_camera[:, :, :3, 3:4]

    # # Camera Space to Scene World Space
    # R_c2w = R_w2c.transpose(-1, -2) # (B, T, 4, 4)
    # t_c2w = (-R_w2c.transpose(-1, -2) @ t_w2c) # (B, T, 3, 1)

    R = camera_to_world[:, :, :3, :3]
    t = camera_to_world[:, :, :3, 3:4]

    # Scene World Space to Traj Space
    # print("input_smplx_scale:", smplx_scale)
    # print("input_shift:", char_start)
    # print("input_rotation:", R_scene_up_to_director_up)
    batch_t = t / smplx_scale
    batch_t = batch_t - char_start.unsqueeze(-1)
    batch_R = R_scene_up_to_director_up @ R
    batch_t = R_scene_up_to_director_up @ batch_t
    batch_mat = torch.zeros((B, T, 4, 4), dtype=dtype, device=device)

    # print("input_director_trans:", batch_t[0, 0])

    batch_mat[:, :, :3, :3] = batch_R
    batch_mat[:, :, :3, 3:4] = batch_t

    # Traj Space to Latent Space
    batch_feat = torch.stack([get_feature(mat) for mat in batch_mat]) # (B, P, T)

    return batch_feat


def dot(a, b, keepdim=False):
    return (a * b).sum(dim=-1, keepdim=keepdim) # (..., 1)

def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

def rot_trans_to_matrix(R, t):
    base_matrix = torch.zeros((*R.shape[:-2], 4, 4), device=R.device, dtype=R.dtype)
    # expand base_matrix as (..., 3, 3) -> (..., 4, 4)

    base_matrix[..., :3, :3] = R
    base_matrix[..., :3, 3] = t

    return base_matrix

def build_cameras(R_batch, t_batch, K):
    B, T, _, _ = R_batch.shape
    device = R_batch.device

    fx_batch = K[0][0][None].repeat(B, T)
    fy_batch = K[1][1][None].repeat(B, T)
    cx = K[0][2][None].float()
    cy = K[1][2][None].float()
    width_batch = torch.tensor(int(cx * 2), device=device, dtype=torch.int)[None].repeat(B, T)
    height_batch = torch.tensor(int(cy * 2), device=device, dtype=torch.int)[None].repeat(B, T)
    appearance_id_batch = torch.zeros(1, device=device, dtype=torch.int).repeat(B, T)
    normalized_appearance_id_batch = torch.zeros(1, device=device, dtype=torch.int).repeat(B, T)
    distortion_params_batch = torch.zeros((1, 4), device=device, dtype=torch.int).repeat(B, T, 1)
    camera_type_batch = torch.zeros(1, device=device, dtype=torch.int).repeat(B, T)
    cx_batch, cy_batch = cx.repeat(B, T), cy.repeat(B, T)

    batch_cameras = BatchCameras(
        R=R_batch,
        T=t_batch,
        fx=fx_batch,
        fy=fy_batch,
        cx=cx_batch,
        cy=cy_batch,
        width=width_batch,
        height=height_batch,
        appearance_id=appearance_id_batch,
        normalized_appearance_id=normalized_appearance_id_batch,
        distortion_params=distortion_params_batch,
        camera_type=camera_type_batch
    )

    return batch_cameras

def build_single_camera(R, t, K):
    device = R.device

    R = R.unsqueeze(0)
    t = t.unsqueeze(0)

    fx = K[0][0][None]
    fy = K[1][1][None]
    cx = K[0][2][None].float()
    cy = K[1][2][None].float()
    width_batch = torch.tensor(int(cx * 2), device=device, dtype=torch.int)[None]
    height_batch = torch.tensor(int(cy * 2), device=device, dtype=torch.int)[None]
    appearance_id_batch = torch.zeros(1, device=device, dtype=torch.int)
    normalized_appearance_id_batch = torch.zeros(1, device=device, dtype=torch.int)
    distortion_params_batch = torch.zeros((1, 4), device=device, dtype=torch.int)
    camera_type_batch = torch.zeros(1, device=device, dtype=torch.int)

    cameras = Cameras(
        R=R,
        T=t,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width_batch,
        height=height_batch,
        appearance_id=appearance_id_batch,
        normalized_appearance_id=normalized_appearance_id_batch,
        distortion_params=distortion_params_batch,
        camera_type=camera_type_batch
    )

    return cameras[0]

# ====================

def margin_range_loss(x, lower=None, upper=None, margin=0.0):
    loss = 0
    if lower is not None:
        inner_lower = lower + margin
        lower_loss = F.relu(inner_lower - x)
        loss += lower_loss
    if upper is not None:
        inner_upper = upper - margin
        upper_loss = F.relu(x - inner_upper)
        loss += upper_loss
    return loss

def scaled_margin_range_loss(x, lower=None, upper=None, margin=0.0):
    loss = 0

    if lower is not None:
        inner_lower = lower + margin
    if upper is not None:
        inner_upper = upper - margin

    if lower is not None and upper is not None:
        scale = torch.abs(inner_upper - inner_lower)
        scale = torch.max(torch.ones_like(scale), scale)
    elif lower is not None:
        scale = torch.max(torch.ones_like(inner_lower), torch.abs(inner_lower))
    elif upper is not None:
        scale = torch.max(torch.ones_like(inner_upper), torch.abs(inner_upper))
    else:
        scale = torch.tensor(1.0, device=x.device)

    if lower is not None:
        lower_loss = F.relu(inner_lower - x)
        loss += lower_loss / scale

    if upper is not None:
        upper_loss = F.relu(x - inner_upper)
        loss += upper_loss / scale

    return loss
import traceback
import datetime
import os.path
from typing import Tuple
from internal.cameras.cameras import Cameras, Camera

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re
import imageio
import json
import cv2
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
import torch.nn.functional as F
import math

from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
import sys
sys.path.insert(0, os.path.join('../../../HSI/fitting', 'main'))
sys.path.insert(0, os.path.join('../../../HSI/fitting', 'common'))
from utils.smpl_x import smpl_x
from lang_sam import LangSAM
import torchvision.transforms as T
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import copy


class ScPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab
        self.renderer = self.viewer.viewer_renderer
        self.gaussian_model = self.renderer.gaussian_model.gaussian_model # means: (N, 3), opacities: (N, 1), scales: (N, 3), rotations: (N, 4), shs: (N, 16, 3)
        self.means = self.gaussian_model.means # (N, 3)
        self.opacities = self.gaussian_model.opacities # (N, 1)
        self.scales = self.gaussian_model.scales # (N, 3)
        self.rotations = self.gaussian_model.rotations # (N, 4)
        
        self.render_wh = (720, 480)
        self.frame_num = 81
        self.render_mode = "rgb"
        self.root_path = self.viewer.root_path

        if os.path.exists(os.path.join(self.root_path, 'data.json')):
            with open(os.path.join(self.root_path, 'data.json'), 'r') as f:
                data = json.load(f)
                data['scene'] = self.viewer.model_paths[0]
            with open(os.path.join(self.root_path, 'data.json'), 'w') as f:
                json.dump(data, f, indent=4)
    
        self.start_point_path = os.path.join(self.root_path, 'start_point.json')
        if os.path.exists(self.start_point_path):
            with open(self.start_point_path, 'r') as f:
                self.init_start_point = np.array(json.load(f))
                self.server.scene.add_point_cloud(
                    name='start_point',
                    points=self.init_start_point[None],
                    colors=(1.0, 0.0, 0.0),
                    point_size=0.01,
                    point_shape='circle'
                )
        else:
            self.init_start_point = np.array([0.0, 0.0, 0.0])
        
        self.end_point_path = os.path.join(self.root_path, 'end_point.json')
        if os.path.exists(self.end_point_path):
            with open(self.end_point_path, 'r') as f:
                self.init_end_point = np.array(json.load(f))
                self.server.scene.add_point_cloud(
                    name='end_point',
                    points=self.init_end_point[None],
                    colors=(0.0, 0.0, 1.0),
                    point_size=0.01,
                    point_shape='circle'
                )
        else:
            self.init_end_point = np.array([0.0, 0.0, 0.0])

        self.data_type_list = ['static', 'traj', 'detailed']
        self.current_data_type = self.data_type_list[0]

        self.running_mesh = False
        self.running_camera = False
        self.running_pcs = False
        self.running_idx = 0

        self.pc_xyz = self.viewer.gaussian_model.get_xyz
        self.min_x, self.min_y, self.min_z = self.pc_xyz.min(dim=0).values
        self.max_x, self.max_y, self.max_z = self.pc_xyz.max(dim=0).values

        self.langsam = LangSAM(sam_type="sam2.1_hiera_large")
        self.to_pil = T.ToPILImage()

        self._build_data_dir()

        self._setup_root_path()
        self._setup_traj_folder()
        self._setup_refine_folder()
        self._setup_vis_folder()

        self.up_vector_path = os.path.join(self.root_path, 'up_vector.json')
        if os.path.exists(self.up_vector_path):
            with open(self.up_vector_path, 'r') as f:
                plane_info = json.load(f)
                self.world_up_vector = torch.tensor(plane_info[:3])
                self.plane_offset = plane_info[-1]
                self.ref_dist = self.calculate_ref_distance()
            print(f"World up vector exists: {self.world_up_vector.tolist()}")
        else:
            self.world_up_vector = None
            self.plane_offset = None
            self.ref_dist = 2.0

    def _setup_root_path(self):
        root_path_input = self.server.gui.add_text(
            "Root Path",
            initial_value=self.root_path,
            hint="Root Path for data"
        )
        @root_path_input.on_update
        def _(_) -> None:
            self.root_path = root_path_input.value
            self._build_data_dir()
            self.set_path('static')

    def _setup_traj_folder(self):
        with self.server.gui.add_folder("Traj"):

            self.start_point = self.server.gui.add_vector3(
                "Start Point",
                min=(self.min_x, self.min_y, self.min_z),
                max=(self.max_x, self.max_y, self.max_z),
                step=0.01,
                initial_value=self.init_start_point,
            )

            self.end_point = self.server.gui.add_vector3(
                "End Point",
                min=(self.min_x, self.min_y, self.min_z),
                max=(self.max_x, self.max_y, self.max_z),
                step=0.01,
                initial_value=self.init_end_point,
            )

            @self.start_point.on_update
            def _(event: viser.GuiEvent):
                client = event.client
                assert client is not None
                scene = client.scene
                with open(os.path.join(self.root_path, 'start_point.json'), 'w') as f:
                    json.dump(self.start_point.value, f)
                scene.add_point_cloud(
                    name='start_point',
                    points=np.array(self.start_point.value)[None],
                    colors=(1.0, 0.0, 0.0),
                    point_size=0.01,
                    point_shape='circle'
                )

            @self.end_point.on_update
            def _(event: viser.GuiEvent):
                client = event.client
                assert client is not None
                scene = client.scene
                with open(os.path.join(self.root_path, 'end_point.json'), 'w') as f:
                    json.dump(self.end_point.value, f)
                scene.add_point_cloud(
                    name='end_point',
                    points=np.array(self.end_point.value)[None],
                    colors=(0.0, 0.0, 1.0),
                    point_size=0.01,
                    point_shape='circle'
                )

            self.start_point.value = self.init_start_point.tolist()
            self.end_point.value = self.init_end_point.tolist()

            pick_start_point_button = self.server.gui.add_button(
                "Pick Start point",
                color="red",
                icon=viser.Icon.CIRCLE_LETTER_S,
                hint="Pick Start point",
            )
            @pick_start_point_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None
                scene = client.scene
                @scene.on_pointer_event(event_type='click')
                def _(event: viser.ScenePointerEvent) -> None: 
                    screen_pos = event.screen_pos # (N=1, 2)
                    
                    scene.remove_pointer_callback()
                    cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)
                    depth = self.render(cameras, render_mode="exp_depth")[0]
                    depth_value = depth[:, :, 0:1] # (H, W, 1)

                    # convert render mode
                    image = self.render(cameras, render_mode="rgb")[0]

                    H, W, _ = depth_value.shape

                    c2w, K = self.get_cam_params(client, img_wh=(W, H))
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]

                    pc = depth_to_pc(depth_value[:, :, 0], K, R, t, screen_pos[0]) # (N=1, 3)
                    self.start_point.value = pc[0].tolist()
                    print(f"start point: {self.start_point.value}")
                    with open(os.path.join(self.root_path, 'start_point.json'), 'w') as f:
                        json.dump(self.start_point.value, f)
                    scene.add_point_cloud(
                        name='start_point',
                        points=pc,
                        colors=(1.0, 0.0, 0.0),
                        point_size=0.01,
                        point_shape='circle'
                    )

            pick_end_point_button = self.server.gui.add_button(
                "Pick End point",
                color="blue",
                icon=viser.Icon.CIRCLE_LETTER_E,
                hint="Pick End point",
            )
            @pick_end_point_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None
                scene = client.scene
                @scene.on_pointer_event(event_type='click')
                def _(event: viser.ScenePointerEvent) -> None: 
                    screen_pos = event.screen_pos # (N=1, 2)
                    
                    scene.remove_pointer_callback()
                    cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)
                    depth = self.render(cameras, render_mode="exp_depth")[0]
                    depth_value = depth[:, :, 0:1] # (H, W, 1)

                    # convert render mode
                    image = self.render(cameras, render_mode="rgb")[0]

                    H, W, _ = depth_value.shape

                    c2w, K = self.get_cam_params(client, img_wh=(W, H))
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]

                    pc = depth_to_pc(depth_value[:, :, 0], K, R, t, screen_pos[0]) # (N=1, 3)
                    self.end_point.value = pc[0].tolist()
                    print(f"end point: {self.end_point.value}")
                    with open(os.path.join(self.root_path, 'end_point.json'), 'w') as f:
                        json.dump(self.end_point.value, f)
                    scene.add_point_cloud(
                        name='end_point',
                        points=pc,
                        colors=(0.0, 0.0, 1.0),
                        point_size=0.01,
                        point_shape='circle'
                    )

            compute_scale_button = self.server.gui.add_button(
                "Compute Scale",
                color="green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Compute Scale",
            )
            @compute_scale_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None
                scene = client.scene

                print("Computing Scale...")
                dist = self.median_self_knn(device="cuda:0")
                print(dist)
                smplx_scale, _ = self.get_priors()

                k = 1.72 * smplx_scale / dist
                print(k)
            
            estimate_up_vector_button = self.server.gui.add_button(
                "Estimate Up Vector",
                color="yellow",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Estimate Up Vector",
            )
            @estimate_up_vector_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                # Dump for static dir
                self.set_path('static')
                segment_result_dir = os.path.join(self.data_path, 'ground_floor_segment')
                os.makedirs(segment_result_dir, exist_ok=True)

                cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)

                cameras, _, _ = self.optimize_cam_params() # w2c

                cam_pos = (-cameras.R[0] @ cameras.T[0]).detach().cpu().numpy()
                
                start_point = np.array(self.start_point.value)
                end_point = np.array(self.end_point.value)
                center_point = (start_point + end_point) / 2
                r_dist = np.linalg.norm(cam_pos - center_point)
                cam_params = get_cam_params_on_arc(start_point, end_point, r_dist=r_dist, bin=36) # (N, 4, 4), c2w

                bin = 12
                text_prompt = "floor. ground."
                center_point = (np.array(self.start_point.value) + np.array(self.end_point.value)) / 2
                normal_vectors = []
                intercepts = []
                for cam_idx, cam_param in enumerate(cam_params):

                    c2ws_roll = get_cam_params_by_rolling(cam_param, bin)

                    images = []
                    depths = []
                    Ks = []
                    c2ws = []

                    for i, c2w_roll in enumerate(c2ws_roll):
                        
                        cameras.R[0] = torch.tensor(c2w_roll[:3, :3].T)
                        cameras.T[0] = torch.tensor(-c2w_roll[:3, :3].T @ c2w_roll[:3, 3])
                        cameras.__post_init__()
                        self.update_camera(client, cameras[0])
                        c2w, K = self.get_cam_params(client, self.render_wh)
                        self.server.scene.set_global_visibility(False)
                        depth = self.render(cameras, render_mode="exp_depth")[0, :, :, 0]
                        image = self.render(cameras, render_mode="rgb")[0]
                        self.server.scene.set_global_visibility(True)

                        uv = project_points_to_image(np.stack([start_point, end_point], axis=0), K, c2w[:3, :3], c2w[:3, 3], include_depth=True)

                        if check_pc_visibility(uv, self.render_wh[0], self.render_wh[1], depth) is False:
                            continue

                        c2ws.append(c2w)
                        Ks.append(K)
                        depths.append(depth)
                        images.append(image)

                    if len(images) == 0:
                        continue

                    pil_images = [self.to_pil(image) for image in images]
                    results = self.langsam.predict(pil_images, [text_prompt] * len(pil_images))
                    masks = []
                    for result in results:
                        mask = np.zeros((self.render_wh[1], self.render_wh[0], 1))
                        for pred in result['masks']:
                            mask = np.logical_or(mask, pred[:, :, None])
                        masks.append(mask)
                    masks = np.array(masks)
                    pcs = [depth_to_pc(depth, K, c2w[:3, :3], c2w[:3, 3]) for depth, c2w, K in zip(depths, c2ws, Ks)]
                    pcs_image = np.array([pc.reshape(self.render_wh[1], self.render_wh[0], 3) for pc in pcs]) # (N, H, W, 3)

                    for i in range(len(images)):
                        image = images[i]
                        mask = masks[i].repeat(3, axis=-1)
                        image[mask > 0.5] = 127
                        imageio.imwrite(os.path.join(segment_result_dir, f'{cam_idx}_{360 // bin * i}.png'), image)
                
                    normals_per_frame, intercepts_per_frame = extract_ground_normals(pcs_image, masks, ref_point=center_point, max_planes=3)
                    
                    normals = [normal for normal in normals_per_frame if len(normal) > 0]
                    normals = np.concatenate(normals, axis=0) # (M, 3)
                    intercept = np.concatenate(intercepts_per_frame, axis=0)
                    normal_vectors.append(normals)
                    intercepts.append(intercept)
                    
                normal_vectors = np.concatenate(normal_vectors, axis=0) # (K, 3)
                intercepts = np.concatenate(intercepts, axis=0) / np.linalg.norm(normal_vectors, axis=1)
                normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1, keepdims=True)

                X = normal_vectors
                dominant_vector, labels, counts, dominant_idx = spherical_kmeans(X, K=10, n_iter=5000) # (3,)
                dominant_intercept = intercepts[labels == dominant_idx].mean()
                dominant_vector = dominant_vector / np.linalg.norm(dominant_vector)
                # dominant_vector = find_dominant_up_vector(normal_vectors)

                print(counts)
                print(dominant_vector)
                print(dominant_intercept)

                self.world_up_vector = torch.tensor(dominant_vector)
                self.plane_offset = dominant_intercept
                with open(self.up_vector_path, 'w') as f:
                    json.dump(self.world_up_vector.tolist() + [self.plane_offset], f)
                self.ref_dist = self.calculate_ref_distance()

                # visualization
                client.scene.add_point_cloud(
                    name=f'dominant',
                    points=dominant_vector[None],
                    colors=(0.5, 0.5, 0.5),
                    point_size=0.05,
                    point_shape='circle'
                )
                client.scene.add_point_cloud(
                    name=f'norm_clusters',
                    points=normal_vectors,
                    colors=np.array([0.25, 0.25, 0.25]),
                    point_size=0.01,
                    point_shape='circle'
                )

            dump_image_for_static = self.server.gui.add_button(
                "Dump Image (Static)",
                color="green",
                icon=viser.Icon.LIVE_PHOTO,
                hint="Dump the current render image as an image for prior",
            )
            @dump_image_for_static.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                # Set data type for setting path
                self.set_path('static')

                cameras, _, _ = self.optimize_cam_params()
                self.update_camera(client, cameras[0])
                render_wh = self.render_wh

                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")[0]
                image = self.render(cameras, render_mode="rgb")[0]
                alpha = self.render(cameras, render_mode="alpha")[0]
                n_contribs = self.render(cameras, render_mode="n_contribs")[0]
                self.server.scene.set_global_visibility(True)
                self.set_rgb_render_mode()

                depth_value = depth[:, :, 0:1] # (H, W, 1)
                depth_map = depth[:, :, 1:].astype(np.uint8) # (H, W, 3)
                
                confidence = np.expand_dims(-np.log(1 - alpha + 1e-6) * n_contribs, axis=-1) # (H, W, 1)
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                confidence_vis = (confidence * 255).astype(np.uint8)
                confidence_vis = np.repeat(confidence_vis, 3, axis=-1) # (H, W, 3)

                c2w, K = self.get_cam_params(client, render_wh)
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                points = np.array([self.start_point.value, self.end_point.value])
                uv = project_points_to_image(points, K, R, t)
                start_uv = uv[0].tolist()
                end_uv = uv[1].tolist()

                image_with_start_point = image.copy()
                cv2.circle(image_with_start_point, (int(start_uv[0]), int(start_uv[1])), 10, (255, 0, 0), -1)
                image_path = os.path.join(self.data_path, 'start_point.png')
                imageio.imwrite(image_path, image_with_start_point)

                image_with_end_point = image.copy()
                cv2.circle(image_with_end_point, (int(end_uv[0]), int(end_uv[1])), 10, (0, 0, 255), -1)
                image_path = os.path.join(self.data_path, 'end_point.png')
                imageio.imwrite(image_path, image_with_end_point)

                # Save camera parameters
                R = c2w[:3, :3].T.tolist() # w2c
                t = (-c2w[:3, :3].T @ c2w[:3, 3]).tolist() # w2c
                focal = [K[0, 0], K[1, 1]]
                princpt = [render_wh[0] // 2, render_wh[1] // 2]

                cam_param = {
                    "R": R,
                    "t": t,
                    "focal": focal,
                    "princpt": princpt
                }
                
                self.set_path('static')

                frames = []
                for i in range(1):
                    depth_path = os.path.join(self.depth_vis_dir_path, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map)
                    depth_npy_path = os.path.join(self.depth_dir_path, f'{i}.npy')
                    np.save(depth_npy_path, depth_value)

                    image_path = os.path.join(self.scene_render_path, f'{i}.png')
                    imageio.imwrite(image_path, image)

                    confidence_path = os.path.join(self.scene_confidence_path, f"{i}.npy")
                    np.save(confidence_path, confidence)
                    confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"{i}.png")
                    imageio.imwrite(confidence_vis_path, confidence_vis)

                    json_path = os.path.join(self.cam_param_path, f"{i}.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)

                    frames.append(image)
                
                imageio.mimsave(os.path.join(self.data_path, 'scene_render_video.mp4'), frames, fps=16)
                
                print("Finish Dumping!")

                self.run_difx()

            dump_image_with_start_end_button = self.server.gui.add_button(
                "Dump Image (Start, End Point)",
                color="green",
                icon=viser.Icon.LIVE_PHOTO,
                hint="Dump the current render image as an image with start point and end point.",
            )
            @dump_image_with_start_end_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                # Dump for static dir
                self.set_path('static')

                cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)
                self.update_camera(client, cameras[0])
                render_wh = self.render_wh

                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")[0]
                image = self.render(cameras, render_mode="rgb")[0]
                alpha = self.render(cameras, render_mode="alpha")[0]
                n_contribs = self.render(cameras, render_mode="n_contribs")[0]
                self.server.scene.set_global_visibility(True)
                self.set_rgb_render_mode()

                depth_value = depth[:, :, 0:1] # (H, W, 1)
                depth_map = depth[:, :, 1:].astype(np.uint8) # (H, W, 3)
                
                confidence = np.expand_dims(-np.log(1 - alpha + 1e-6) * n_contribs, axis=-1) # (H, W, 1)
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                confidence_vis = (confidence * 255).astype(np.uint8)
                confidence_vis = np.repeat(confidence_vis, 3, axis=-1) # (H, W, 3)

                depth_path = os.path.join(self.depth_vis_dir_path, '0.png')
                imageio.imwrite(depth_path, depth_map)
                depth_npy_path = os.path.join(self.depth_dir_path, '0.npy')
                np.save(depth_npy_path, depth_value)
                print(f"Save depth render in {self.depth_vis_dir_path}\nRender resolution is {depth_map.shape}")
                print(f"Save depth in {self.depth_dir_path}\nDepth resolution is {depth_value.shape}")
                print(f"Max depth: {depth_value.max()}")
                print(f"Min depth: {depth_value.min()}")

                image_path = os.path.join(self.scene_render_path, '0.png')
                imageio.imwrite(image_path, image)
                print(f"Save image in {self.scene_render_path}\nRender resolution is {image.shape}")

                confidence_path = os.path.join(self.scene_confidence_path, f"0.npy")
                np.save(confidence_path, confidence)
                confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"0.png")
                imageio.imwrite(confidence_vis_path, confidence_vis)

                c2w, K = self.get_cam_params(client, render_wh)
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                points = np.array([self.start_point.value, self.end_point.value])
                uv = project_points_to_image(points, K, R, t)
                start_uv = uv[0].tolist()
                end_uv = uv[1].tolist()

                image_with_start_point = image.copy()
                cv2.circle(image_with_start_point, (int(start_uv[0]), int(start_uv[1])), 10, (255, 0, 0), -1)
                image_path = os.path.join(self.data_path, 'start_point.png')
                imageio.imwrite(image_path, image_with_start_point)

                image_with_end_point = image.copy()
                cv2.circle(image_with_end_point, (int(end_uv[0]), int(end_uv[1])), 10, (0, 0, 255), -1)
                image_path = os.path.join(self.data_path, 'end_point.png')
                imageio.imwrite(image_path, image_with_end_point)

                # Save start/end point
                points_path = os.path.join(self.data_path, 'points.json')
                points = {
                    "start_point": self.start_point.value,
                    "end_point": self.end_point.value
                }
                with open(points_path, 'w') as f:
                    json.dump(points, f)
                print(f"Save points in {points_path}")
                print(f"Start point: {points["start_point"]}")
                print(f"End point: {points["end_point"]}")

                # Save camera parameters
                R = c2w[:3, :3].T.tolist() # w2c
                t = (-c2w[:3, :3].T @ c2w[:3, 3]).tolist() # w2c
                focal = [K[0, 0], K[1, 1]]
                princpt = [render_wh[0] // 2, render_wh[1] // 2]

                cam_param = {
                    "R": R,
                    "t": t,
                    "focal": focal,
                    "princpt": princpt
                }
                json_path = os.path.join(self.cam_param_path, "0.json")
                with open(json_path, 'w') as f:
                    json.dump(cam_param, f)
                
                self.set_path('traj')

                for i in range(self.frame_num):
                    depth_path = os.path.join(self.depth_vis_dir_path, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map)
                    depth_npy_path = os.path.join(self.depth_dir_path, f'{i}.npy')
                    np.save(depth_npy_path, depth_value)

                    image_path = os.path.join(self.scene_render_path, f'{i}.png')
                    imageio.imwrite(image_path, image)

                    confidence_path = os.path.join(self.scene_confidence_path, f"{i}.npy")
                    np.save(confidence_path, confidence)
                    confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"{i}.png")
                    imageio.imwrite(confidence_vis_path, confidence_vis)

                    json_path = os.path.join(self.cam_param_path, f"{i}.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)
                
                print("Finish Dumping!")

            auto_dump_image_with_start_end_button = self.server.gui.add_button(
                "Auto Dump Image (Start, End Point)",
                color="green",
                icon=viser.Icon.LIVE_PHOTO,
                hint="Dump the current render image as an image with start point and end point automatically.",
            )
            @auto_dump_image_with_start_end_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                smplx_scale, _ = self.get_priors()

                # Dump for traj dir
                self.set_path('traj')

                cameras, render, bbox_mask = self.optimize_cam_params(smplx_scale)
                self.update_camera(client, cameras[0])
                start_image = render[0].detach().cpu().numpy().copy().astype(np.uint8)
                end_image = render[0].detach().cpu().numpy().copy().astype(np.uint8)

                bbox_mask = bbox_mask.squeeze(dim=-1).repeat(1, 1, 1, 3).detach().cpu().numpy().astype(np.uint8) * 255 # (N, H, W)

                start_image[bbox_mask[0] > 0] = 127
                end_image[bbox_mask[1] > 0] = 127

                H, W, _ = start_image.shape
                ratio = 0.01
                bb = [int(W * ratio), int(H * ratio), W - int(W * ratio), H - int(H * ratio)]
                mask_intersection = ((start_image == 127) & (end_image == 127)).all(axis=-1, keepdims=True) 

                human_masked_video = []
                human_masked_video.append(start_image)
                for i in range(self.frame_num - 2):
                    center_frame = start_image.copy()
                    center_frame[bb[1]:bb[3], bb[0]:bb[2], :] = 127
                    center_frame = np.where(mask_intersection, 127, center_frame)
                    human_masked_video.append(center_frame)
                human_masked_video.append(end_image)
                imageio.mimsave(os.path.join(self.data_path, 'human_masked_video.mp4'), human_masked_video, fps=16)

                masked_video = []
                masked_video.append(bbox_mask[0])
                for i in range(self.frame_num - 2):
                    center_mask = bbox_mask[0].copy()
                    center_mask[bb[1]:bb[3], bb[0]:bb[2], :] = 255
                    center_mask = np.where(mask_intersection, 255, center_mask)
                    masked_video.append(center_mask)
                masked_video.append(bbox_mask[1])
                imageio.mimsave(os.path.join(self.data_path, 'mask_video.mp4'), masked_video, fps=16)

                print("Optimization is finished.")
                print("Start dumping")

                render_wh = self.render_wh

                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")[0]
                image = self.render(cameras, render_mode="rgb")[0]
                alpha = self.render(cameras, render_mode="alpha")[0]
                n_contribs = self.render(cameras, render_mode="n_contribs")[0]
                self.server.scene.set_global_visibility(True)
                self.set_rgb_render_mode()

                depth_value = depth[:, :, 0:1] # (H, W, 1)
                depth_map = depth[:, :, 1:].astype(np.uint8) # (H, W, 3)
                
                confidence = np.expand_dims(-np.log(1 - alpha + 1e-6) * n_contribs, axis=-1) # (H, W, 1)
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                confidence_vis = (confidence * 255).astype(np.uint8)
                confidence_vis = np.repeat(confidence_vis, 3, axis=-1) # (H, W, 3)

                depth_path = os.path.join(self.depth_vis_dir_path, '0.png')
                imageio.imwrite(depth_path, depth_map)
                depth_npy_path = os.path.join(self.depth_dir_path, '0.npy')
                np.save(depth_npy_path, depth_value)
                print(f"Save depth render in {self.depth_vis_dir_path}\nRender resolution is {depth_map.shape}")
                print(f"Save depth in {self.depth_dir_path}\nDepth resolution is {depth_value.shape}")
                print(f"Max depth: {depth_value.max()}")
                print(f"Min depth: {depth_value.min()}")

                image_path = os.path.join(self.scene_render_path, '0.png')
                imageio.imwrite(image_path, image)
                print(f"Save image in {self.scene_render_path}\nRender resolution is {image.shape}")

                confidence_path = os.path.join(self.scene_confidence_path, f"0.npy")
                np.save(confidence_path, confidence)
                confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"0.png")
                imageio.imwrite(confidence_vis_path, confidence_vis)

                c2w, K = self.get_cam_params(client, render_wh)
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                points = np.array([self.start_point.value, self.end_point.value])
                uv = project_points_to_image(points, K, R, t)
                start_uv = uv[0].tolist()
                end_uv = uv[1].tolist()

                image_with_start_point = image.copy()
                cv2.circle(image_with_start_point, (int(start_uv[0]), int(start_uv[1])), 10, (255, 0, 0), -1)
                image_path = os.path.join(self.data_path, 'start_point.png')
                imageio.imwrite(image_path, image_with_start_point)

                image_with_end_point = image.copy()
                cv2.circle(image_with_end_point, (int(end_uv[0]), int(end_uv[1])), 10, (0, 0, 255), -1)
                image_path = os.path.join(self.data_path, 'end_point.png')
                imageio.imwrite(image_path, image_with_end_point)

                # Save camera parameters
                R = c2w[:3, :3].T.tolist() # w2c
                t = (-c2w[:3, :3].T @ c2w[:3, 3]).tolist() # w2c
                focal = [K[0, 0], K[1, 1]]
                princpt = [render_wh[0] // 2, render_wh[1] // 2]

                cam_param = {
                    "R": R,
                    "t": t,
                    "focal": focal,
                    "princpt": princpt
                }
                json_path = os.path.join(self.cam_param_path, "0.json")
                with open(json_path, 'w') as f:
                    json.dump(cam_param, f)
                
                self.set_path('traj')

                frames = []
                for i in range(self.frame_num):
                    depth_path = os.path.join(self.depth_vis_dir_path, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map)
                    depth_npy_path = os.path.join(self.depth_dir_path, f'{i}.npy')
                    np.save(depth_npy_path, depth_value)

                    image_path = os.path.join(self.scene_render_path, f'{i}.png')
                    imageio.imwrite(image_path, image)

                    confidence_path = os.path.join(self.scene_confidence_path, f"{i}.npy")
                    np.save(confidence_path, confidence)
                    confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"{i}.png")
                    imageio.imwrite(confidence_vis_path, confidence_vis)

                    json_path = os.path.join(self.cam_param_path, f"{i}.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)

                    frames.append(image)
                
                imageio.mimsave(os.path.join(self.data_path, 'scene_render_video.mp4'), frames, fps=16)
                
                print("Finish Dumping!")

                self.run_difx()

    def _setup_refine_folder(self):
        with self.server.gui.add_folder("Refine"):
            load_cam_params_button = self.server.gui.add_button(
                "Load Camera Parameters",
                color="yellow",
                icon=viser.Icon.FOLDER_OPEN,
                hint="Load a camera parameter",
            )

            @load_cam_params_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path('detailed')
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                
                # Render Video
                self.server.scene.set_global_visibility(False)
                image = self.render(cameras, render_mode="rgb") # (N, H, W, 3)
                self.server.scene.set_global_visibility(True)

                for i in range(len(cameras)):
                    image_path = os.path.join(self.scene_render_path, f"{i}.png")
                    imageio.imwrite(image_path, image[i])

                render_video_path = os.path.join(self.data_path, 'scene_render_video.mp4')
                imageio.mimsave(render_video_path, image, fps=16)
                print(f"Video saved in {render_video_path}")

                self.update_camera(client, cameras[0])

            dump_video_with_traj_SMPL = self.server.gui.add_button(
                "Dump video with traj SMPL",
                color="green",
                icon=viser.Icon.CAMERA,
                hint="Dump video with traj SMPL",
            )

            @dump_video_with_traj_SMPL.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path('detailed')
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                self.set_path('traj')
                mesh_dir_path = os.path.join(self.data_path, "smplx_optimized", "meshes_smoothed")
                mesh_path_list = [os.path.join(mesh_dir_path, f'{i}.ply') for i in range(len(os.listdir(mesh_dir_path)))]
                mesh_list = [trimesh.load_mesh(mesh_path).apply_transform(c2w[i]) for i, mesh_path in enumerate(mesh_path_list)]

                bg_depth = self.render(cameras, render_mode="exp_depth")[:, :, :, 0:1]
                bg_img = self.render(cameras, render_mode="rgb")

                images = []
                assert len(c2w) == len(mesh_list), "Length of cam params should be same with length of SMPL-X mesh"
                for i in tqdm(range(len(mesh_list))):
                    self.update_camera(client, cameras[i])
                    client.scene.set_background_image(bg_img[i], depth=bg_depth[i])
                    mesh_handle = client.scene.add_mesh_trimesh(
                        "Traj SMPL-X",
                        mesh=mesh_list[i]
                    )
                    # TODO: handle the dependency of web viewer resolution
                    image = client.camera.get_render(height=int(princpt[i][1] * 2), width=int(princpt[i][0] * 2))
                    images.append(image)

                self.set_path('detailed')
                render_video_path = os.path.join(self.data_path, 'scene_render_video_with_traj_SMPL.mp4')
                imageio.mimsave(render_video_path, images, fps=16)
                print(f"Video saved in {render_video_path}")

                self.update_camera(client, cameras[0])

            dump_button = self.server.gui.add_button(
                "Dump",
                color="black",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Dump.",
            )
            @dump_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                self.set_path('detailed')
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)

                # Render Depth
                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")
                depth_value = depth[:, :, :, 0:1] # (N, H, W, 1)
                depth_map = depth[:, :, :, 1:].astype(np.uint8) # (N, H, W, 3)
                alpha = self.render(cameras, render_mode="alpha") # (N, H, W, 1)
                n_contribs = self.render(cameras, render_mode="n_contribs") # (N, H, W, !)
                self.server.scene.set_global_visibility(True)

                confidence = np.expand_dims(-np.log(1 - alpha + 1e-6) * n_contribs, axis=-1) # (N, H, W, 1)
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                confidence_vis = (confidence * 255).astype(np.uint8)
                confidence_vis = np.repeat(confidence_vis, 3, axis=-1) # (N, H, W, 3)

                self.set_rgb_render_mode()
                self.update_camera(client, cameras[0])
                
                for i in range(depth.shape[0]):
                    depth_path = os.path.join(self.depth_vis_dir_path, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map[i])
                    depth_npy_path = os.path.join(self.depth_dir_path, f'{i}.npy')
                    np.save(depth_npy_path, depth_value[i])
                
                for i in range(confidence.shape[0]):
                    confidence_path = os.path.join(self.scene_confidence_path, f'{i}.png')
                    np.save(confidence_path, confidence[i])
                    confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f'{i}.png')
                    imageio.imwrite(confidence_vis_path, confidence_vis[i])
                
                print("Finish Dumping!")

                self.run_difx()

    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Visualization"):

            self.show_visualization_checkbox = self.server.gui.add_checkbox(
                "Show Visualization",
                initial_value=True,
            )
            @self.show_visualization_checkbox.on_update
            def _(event: viser.GuiEvent):
                client = event.client
                assert client is not None
                scene = client.scene
                if self.show_visualization_checkbox.value:
                    scene.set_global_visibility(True)
                else:
                    scene.set_global_visibility(False)

            vis_up_vector_button = self.server.gui.add_button(
                "Visualize Up Vector",
                color="blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Up vector as point clouds.",
            )
            @vis_up_vector_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                start_point = np.array(self.start_point.value)
                end_point = np.array(self.end_point.value)
                center_point = (start_point + end_point) / 2
                self.ref_dist = self.calculate_ref_distance()

                point_on_plane = center_point - self.world_up_vector.detach().cpu().numpy() * self.ref_dist
                up_vector_vis = point_on_plane + np.linspace(0, 1, 100)[:, None] * (center_point - point_on_plane)

                client.scene.add_point_cloud(
                    name="world_up_vector",
                    points=up_vector_vis,       # (N, 3)
                    colors=np.array([0.0, 0.0, 0.0]),     # (N, 3) in [0, 1]
                    point_size=0.01
                )

            data_type_dropdown = self.server.gui.add_dropdown(
                label="Type",
                options=self.data_type_list,
                initial_value=self.data_type_list[0],
            )
            @data_type_dropdown.on_update
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None
                self.current_data_type = data_type_dropdown.value
                self.set_path(self.current_data_type)

            vis_traj_button = self.server.gui.add_button(
                "Visualize Trajectory",
                color="blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize trajectory as point clouds.",
            )
            @vis_traj_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path(self.current_data_type)
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                traj_path_list = [os.path.join(self.char_traj_dir_path, f'{i}.json') for i in range(len(os.listdir(self.char_traj_dir_path)))]

                traj_list = []
                for traj_path in traj_path_list:
                    with open(traj_path, 'r') as f:
                        traj_cam = json.load(f)['traj']
                    traj_list.append(traj_cam)
                
                # traj_list = np.load(traj_path)
                traj = np.array(traj_list)
                traj = np.einsum('nij,nj->ni', c2w[:, :3, :3], traj) + c2w[:, :3, 3]
                
                np.save(os.path.join(self.data_path, 'traj_world.npy'), traj)
                N = traj.shape[0]
                colors = np.stack([
                    np.linspace(1.0, 0.0, N),  # R
                    np.zeros(N),              # G
                    np.linspace(0.0, 1.0, N)  # B
                ], axis=1)  # shape: (N, 3)

                client.scene.add_point_cloud(
                    name="trajectory_cloud",
                    points=traj,       # (N, 3)
                    colors=colors,     # (N, 3) in [0, 1]
                    point_size=0.01
                )

                self.update_camera(client, cameras[0])
                                        
            play_SMPL_button = self.server.gui.add_button(
                "Play SMPL",
                color="white",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Play SMPL-X as mesh.",
            )

            @play_SMPL_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path(self.current_data_type)
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                mesh_path_list = [os.path.join(self.mesh_dir_path, f'{i}.ply') for i in range(len(os.listdir(self.mesh_dir_path)))]
                mesh_list = [trimesh.load_mesh(mesh_path).apply_transform(c2w[i]) for i, mesh_path in enumerate(mesh_path_list)]

                self.update_camera(client, cameras[0])

                import threading
                import time
                lock = threading.Lock()
                
                def render_next_mesh():
                    while self.running_mesh:
                        with lock:
                            i = self.running_idx
                            mesh_path = mesh_path_list[i]
                            mesh = mesh_list[i]
                            mesh_handle = client.scene.add_mesh_trimesh(
                                mesh_path,
                                mesh=mesh
                            )
                            cameras = self.get_cameras_from_client(client, self.render_wh)
                            depth = self.render(cameras, render_mode="exp_depth")[0, :, :, 0:1]
                            img = self.render(cameras, render_mode="rgb")[0]
                            client.scene.set_background_image(img, depth=depth)

                            self.running_idx = (self.running_idx + 1) % len(mesh_path_list)

                        time.sleep(1 / 30)  # 1초 간격
                        mesh_handle.remove()
                
                def start_render():
                    if not self.running_mesh:
                        self.running_mesh = True
                        timer_thread = threading.Thread(target=render_next_mesh, daemon=True)
                        timer_thread.start()
                start_render()

                print("Play SMPL-X")
            
            stop_playing_SMPL_button = self.server.gui.add_button(
                "Stop Playing SMPL",
                color="red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Stop Playing SMPL-X as mesh.",
            )

            @stop_playing_SMPL_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.running_mesh = False

            play_camera_button = self.server.gui.add_button(
                "Play Camera",
                color="white",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Play Camera as frustum.",
            )

            @play_camera_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path(self.current_data_type)
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)

                start_color = np.array([1.0, 0.0, 0.0])
                end_color = np.array([0.0, 0.0, 1.0])
                t = np.linspace(0, 1, len(cameras))[:, None]  # (N, 1)
                colors = (1 - t) * start_color + t * end_color  # (N, 3)

                camera_data_list = [self.camera_to_fov_quat_position(camera) for camera in cameras] # (fov, quat, position)
                camera_positions = np.array([data[-1] for data in camera_data_list])
                self.camera_pc_handle = client.scene.add_point_cloud(
                    name="camera_trajectory",
                    points=camera_positions,       # (N, 3)
                    colors=colors,     # (N, 3) in [0, 1]
                    point_size=0.01
                )

                self.update_camera(client, cameras[0])

                import threading
                import time
                lock = threading.Lock()
                
                def render_next_camera():
                    while self.running_camera:
                        with lock:
                            i = self.running_idx
                            fov, quat, position = camera_data_list[i]
                            self.camera_frustum_handle = client.scene.add_camera_frustum(
                                f"camera_{i}",
                                fov=fov,
                                aspect=self.render_wh[0] / self.render_wh[1],
                                scale=0.1,
                                position=position,
                                wxyz=quat,
                                visible=True,
                                color=colors[i],
                            )
                            """
                            cameras = self.get_cameras_from_client(client, self.render_wh)
                            depth = self.render(cameras, render_mode="exp_depth")[0, :, :, 0:1]
                            img = self.render(cameras, render_mode="rgb")[0]
                            client.scene.set_background_image(img, depth=depth)
                            """

                            self.running_idx = (self.running_idx + 1) % len(camera_data_list)

                        time.sleep(1 / 30)  # 1초 간격
                        self.camera_frustum_handle.remove()
                
                def start_render():
                    if not self.running_camera:
                        self.running_camera = True
                        timer_thread = threading.Thread(target=render_next_camera, daemon=True)
                        timer_thread.start()
                start_render()

                print("Play Camera")
            
            stop_playing_camera_button = self.server.gui.add_button(
                "Stop Playing Camera",
                color="red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Stop Playing Camera as frustum",
            )

            @stop_playing_camera_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.running_camera = False
                self.camera_pc_handle.remove()
                self.camera_frustum_handle.remove()

            play_optimized_depth_button = self.server.gui.add_button(
                "Play Optimized depth",
                color="white",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Play Optimized Depth as point clouds.",
            )

            @play_optimized_depth_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path(self.current_data_type)
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                self.update_camera(client, cameras[0])

                scene_render_path = self.scene_render_path
                video_frame_path = self.video_frame_path
                scene_depth_path = self.depth_dir_path
                optimized_depth_path = self.optimized_depth_path

                N = len(cam_params_path)
                H, W = cv2.imread(os.path.join(video_frame_path, '0.png')).shape[:2]

                scene_depths = np.array([np.load(os.path.join(scene_depth_path, f'{i}.npy'))[:, :, 0] for i in range(len(os.listdir(scene_depth_path)))])
                optimized_depths = np.array([cv2.resize(np.load(os.path.join(optimized_depth_path, f'{i}.npy')), (W, H)) for i in range(len(os.listdir(optimized_depth_path)))])
                scene_renders = np.array([cv2.cvtColor(cv2.imread(os.path.join(scene_render_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(scene_render_path)))])
                video_frames = np.array([cv2.cvtColor(cv2.imread(os.path.join(video_frame_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(video_frame_path)))])

                pc_worlds = []
                pc_rgbs = []
                for i in range(N):
                    # if i != 0:#  and i != 80:
                    #     continue
                    fx, fy = focal[i]
                    cx, cy = princpt[i]
                    
                    c2w_R = c2w[i, :3, :3]
                    c2w_t = c2w[i, :3, 3]

                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]
                    ])
                    pc_rgb = video_frames[i].reshape(-1, 3)
                    pc_world = depth_to_pc(optimized_depths[i], K, c2w_R, c2w_t)
                    pc_worlds.append(pc_world)
                    pc_rgbs.append(pc_rgb)

                self.update_camera(client, cameras[0])

                import threading
                import time
                lock = threading.Lock()
                
                def render_next_pcs():
                    while self.running_pcs:
                        with lock:
                            i = self.running_idx
                            self.optimized_pcs_handle = client.scene.add_point_cloud(
                                name=f"pc_world_{i}",
                                points=pc_worlds[i],
                                colors=pc_rgbs[i],
                                point_size=0.001
                            )
                            """
                            cameras = self.get_cameras_from_client(client, self.render_wh)
                            depth = self.render(cameras, render_mode="exp_depth")[0, :, :, 0:1]
                            img = self.render(cameras, render_mode="rgb")[0]
                            client.scene.set_background_image(img, depth=depth)
                            """

                            self.running_idx = (self.running_idx + 1) % len(pc_worlds)

                        time.sleep(1 / 16)  # 1초 간격
                        self.optimized_pcs_handle.remove()
                
                def start_render():
                    if not self.running_pcs:
                        self.running_pcs = True
                        timer_thread = threading.Thread(target=render_next_pcs, daemon=True)
                        timer_thread.start()
                start_render()

                print("Play Optimized Pcs")
            
            stop_playing_optimized_depth_button = self.server.gui.add_button(
                "Stop Playing Optimized Depth",
                color="red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Stop Playing Camera as frustum",
            )

            @stop_playing_optimized_depth_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.running_pcs = False
                self.optimized_pcs_handle.remove()

            self.show_optimized_depth_checkbox = self.server.gui.add_checkbox(
                "Show pc from optimized depth",
                initial_value=False,
            )

            vis_pc_from_depth_button = self.server.gui.add_button(
                "Visualize Point Cloud from Depth",
                color="orange",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Point Cloud from depth",
            )

            @vis_pc_from_depth_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                self.set_path(self.current_data_type)
                cam_param_dir_path = self.cam_param_path
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                self.update_camera(client, cameras[0])

                scene_render_path = self.scene_render_path
                video_frame_path = self.video_frame_path
                scene_depth_path = self.depth_dir_path
                optimized_depth_path = self.optimized_depth_path

                N = len(cam_params_path)
                H, W = cv2.imread(os.path.join(video_frame_path, '0.png')).shape[:2]

                scene_depths = np.array([np.load(os.path.join(scene_depth_path, f'{i}.npy'))[:, :, 0] for i in range(len(os.listdir(scene_depth_path)))])
                optimized_depths = np.array([cv2.resize(np.load(os.path.join(optimized_depth_path, f'{i}.npy')), (W, H)) for i in range(len(os.listdir(optimized_depth_path)))])
                scene_renders = np.array([cv2.cvtColor(cv2.imread(os.path.join(scene_render_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(scene_render_path)))])
                video_frames = np.array([cv2.cvtColor(cv2.imread(os.path.join(video_frame_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(video_frame_path)))])

                for i in range(N):
                    # if i != 0:#  and i != 80:
                    #     continue
                    fx, fy = focal[i]
                    cx, cy = princpt[i]
                    
                    c2w_R = c2w[i, :3, :3]
                    c2w_t = c2w[i, :3, 3]

                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]
                    ])
                    if self.show_optimized_depth_checkbox.value:
                        pc_rgb = video_frames[i].reshape(-1, 3)
                        pc_world = depth_to_pc(optimized_depths[i], K, c2w_R, c2w_t)
                    else:
                        pc_rgb = scene_renders[i].reshape(-1, 3)
                        pc_world = depth_to_pc(scene_depths[i], K, c2w_R, c2w_t)
                    
                    client.scene.add_point_cloud(
                        name=f"pc_world_{i}",
                        points=pc_world,
                        colors=pc_rgb,
                        point_size=0.001
                    )
            dump_n_contrib_button = self.server.gui.add_button(
                "Dump N_Contrib",
                color="blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize confidence.",
            )
            @dump_n_contrib_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                # Dump for static dir
                self.set_path('static')

                cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)
                self.update_camera(client, cameras[0])

                # Render Depth
                self.server.scene.set_global_visibility(False)
                alpha = self.render(cameras, render_mode="alpha")
                n_contribs = self.render(cameras, render_mode="n_contribs")
                self.server.scene.set_global_visibility(True)
                
                confidence = -np.log(1 - alpha + 1e-6) * n_contribs
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                confidence_vis = np.expand_dims(confidence * 255, axis=-1).astype(np.uint8)
                confidence_vis = np.repeat(confidence_vis, 3, axis=-1)

                for i in range(confidence.shape[0]):
                    confidence_path = os.path.join(self.scene_confidence_path, f"{i}.npy")
                    np.save(confidence_path, confidence[i])
                    confidence_vis_path = os.path.join(self.scene_confidence_vis_path, f"{i}.png")
                    imageio.imwrite(confidence_vis_path, confidence_vis[i])
                
                self.set_rgb_render_mode()

    def _build_data_dir(self):
        root_path = self.root_path
        if not os.path.exists(root_path):
            print("Root Path does not exist.")
            return
        
        for data_type in self.data_type_list:
            self.data_path = os.path.join(root_path, data_type)
            self.scene_render_path = os.path.join(self.data_path, 'scene_renders')
            self.depth_dir_path = os.path.join(self.data_path, 'depths_scene')
            self.depth_vis_dir_path = os.path.join(self.data_path, 'depths_scene_vis')
            self.cam_param_path = os.path.join(self.data_path, 'cam_params_scene')
            self.scene_confidence_path = os.path.join(self.data_path, 'scene_confidence')
            self.scene_confidence_vis_path = os.path.join(self.data_path, 'scene_confidence_vis')

            os.makedirs(self.data_path, exist_ok=True)
            os.makedirs(self.scene_render_path, exist_ok=True)
            os.makedirs(self.depth_dir_path, exist_ok=True)
            os.makedirs(self.depth_vis_dir_path, exist_ok=True)
            os.makedirs(self.cam_param_path, exist_ok=True)
            os.makedirs(self.scene_confidence_path, exist_ok=True)
            os.makedirs(self.scene_confidence_vis_path, exist_ok=True)

    def set_path(self, data_type):
        if data_type not in self.data_type_list:
            print("Unsupported data type!")
            return

        self.current_data_type = data_type
        self.data_path = os.path.join(self.root_path, data_type)
        self.scene_render_path = os.path.join(self.data_path, 'scene_renders')
        self.depth_dir_path = os.path.join(self.data_path, 'depths_scene')
        self.depth_vis_dir_path = os.path.join(self.data_path, 'depths_scene_vis')
        self.cam_param_path = os.path.join(self.data_path, 'cam_params_scene')
        self.scene_confidence_path = os.path.join(self.data_path, 'scene_confidence')
        self.scene_confidence_vis_path = os.path.join(self.data_path, 'scene_confidence_vis')
        self.mesh_dir_path = os.path.join(self.data_path, "smplx_optimized", "meshes_smoothed")
        self.char_traj_dir_path = os.path.join(self.data_path, "smplx_optimized", "traj_smoothed")
        self.optimized_depth_path = os.path.join(self.data_path, 'depths_vggt_optimized')
        self.video_frame_path = os.path.join(self.data_path, 'frames')
        self.smplx_scale_path = os.path.join(self.data_path, "smplx_optimized", "scale.json")
        self.smplx_shape_path = os.path.join(self.data_path, 'smplx_optimized', "shape_param.json")
        self.smplx_params_path = os.path.join(self.data_path, "smplx_optimized", "smplx_params_smoothed")

    def run_difx(self):
        print("Running Difix...")
        os.chdir('../Difix3D/src')
        cmd = f"python run_difix.py --root_path {self.data_path}"
        print(cmd)
        os.system(cmd)
        os.chdir('../../gaussian-splatting-lightning')
        print("Finish Difix...")
    
    def get_current_camera(self, client: viser.ClientHandle):
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vtf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        fov = camera.fov
        return c2w, fov
    
    def get_K(self, fov: float, img_wh: Tuple[int, int]):
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ) # (3, 3)
        return K
    
    def get_cam_params(self, client: viser.ClientHandle, img_wh: Tuple[int, int]):
        c2w, fov = self.get_current_camera(client) # (4, 4)
        K = self.get_K(fov, img_wh) # (3, 3)

        return c2w, K
    
    def get_cameras_from_client(self, client: viser.ClientHandle, img_wh: Tuple[int, int]):
        c2w, fov = self.get_current_camera(client) # (4, 4), float
        K = self.get_K(fov, img_wh) # (3, 3)

        R = torch.tensor(c2w[:3, :3]).T[None] # w2c
        t = torch.tensor(-c2w[:3, :3].T @ c2w[:3, 3])[None] # w2c
        fx = torch.tensor(K[0][0])[None]
        fy = torch.tensor(K[1][1])[None]
        cx = torch.tensor(K[0][2])[None]
        cy = torch.tensor(K[1][2])[None]
        width = torch.tensor(int(cx * 2), dtype=torch.int)[None]
        height = torch.tensor(int(cy * 2), dtype=torch.int)[None]
        appearance_id = torch.zeros(1, dtype=torch.int)
        normalized_appearance_id = torch.zeros(1, dtype=torch.int)
        distortion_params = torch.zeros((1, 4), dtype=torch.int)
        camera_type = torch.zeros(1, dtype=torch.int)

        camera = Cameras(
            R=R,
            T=t,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            appearance_id=appearance_id,
            normalized_appearance_id=normalized_appearance_id,
            distortion_params=distortion_params,
            camera_type=camera_type
        )
        return camera

    def get_cameras_from_json(self, cam_params_path):
        cam_params = []
        for i, cam_param_path in enumerate(cam_params_path):
            with open(cam_param_path, 'r') as f:
                cam_param = json.load(f)
            cam_params.append(cam_param)
        N = len(cam_params)
        R = torch.tensor([cam_param['R'] for cam_param in cam_params])
        t = torch.tensor([cam_param['t'] for cam_param in cam_params])
        fx = torch.tensor([cam_param['focal'][0] for cam_param in cam_params])
        fy = torch.tensor([cam_param['focal'][1] for cam_param in cam_params])
        cx = torch.tensor([cam_param['princpt'][0] for cam_param in cam_params])
        cy = torch.tensor([cam_param['princpt'][1] for cam_param in cam_params])
        width = torch.tensor(cx * 2, dtype=torch.int)
        height = torch.tensor(cy * 2, dtype=torch.int)
        appearance_id = torch.zeros(N, dtype=torch.int)
        normalized_appearance_id = torch.zeros(N, dtype=torch.int)
        distortion_params = torch.zeros((N, 4), dtype=torch.int)
        camera_type = torch.zeros(N, dtype=torch.int)

        camera = Cameras(
            R=R,
            T=t,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            appearance_id=appearance_id,
            normalized_appearance_id=normalized_appearance_id,
            distortion_params=distortion_params,
            camera_type=camera_type
        )
        return camera

    def camera_to_fov_quat_position(self, camera: Camera):
        c2w = torch.linalg.inv(camera.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        focal = [camera.fx, camera.fy]
        princpt = [camera.cx, camera.cy]
        r = Rotation.from_matrix(R.tolist())
        quat = r.as_quat() # (x, y, z, w)
        quat = np.array([quat[3], quat[0], quat[1], quat[2]]) # (w, x, y, z)
        fov_radians = 2 * np.arctan(2 * princpt[1].cpu() / (2 * focal[1].cpu()))
        return fov_radians, quat, t
    
    def update_camera(self, client, camera: Camera):
        # world_to_camera is transposed. To get c2w, apply transpose -> inverse
        c2w = torch.linalg.inv(camera.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        focal = [camera.fx, camera.fy]
        princpt = [camera.cx, camera.cy]
        r = Rotation.from_matrix(R.tolist())
        quat = r.as_quat() # (x, y, z, w)
        quat = [quat[3], quat[0], quat[1], quat[2]] # (w, x, y, z)
        fov_radians = 2 * np.arctan(2 * princpt[1].cpu() / (2 * focal[1].cpu()))
        client.camera.fov = fov_radians
        client.camera.wxyz = quat
        client.camera.position = t

    def set_rgb_render_mode(self):
        available_outputs = self.renderer.renderer.get_available_outputs()
        output_type_info = available_outputs.get("rgb", None)
        self.renderer._set_output_type("rgb", output_type_info)
        return

    def render(self, cameras: Cameras, render_mode="rgb", optimizable=False):
        available_outputs = self.renderer.renderer.get_available_outputs()

        output_type_info = available_outputs.get(render_mode, None)
        if output_type_info is None:
            return
        if render_mode == "exp_depth" or "n_contribs" or "alpha":
            output_type_info.visualizer = self.renderer.no_processing
        
        self.renderer._set_output_type(render_mode, output_type_info)
        renders = []
        for camera in cameras:
            if render_mode == "rgb":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).permute(1, 2, 0) # (H, W, 3)
                     # Clipping
                    render = torch.clamp(render, 0.0, 1.0)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
                    # Clipping
                    render = np.clip(render, 0.0, 1.0)
                    render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
            elif render_mode == "exp_depth":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).permute(1, 2, 0)[:, :, 0] # (H, W)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
                    depth = render.transpose(1, 2, 0) # (H, W, 3)
                    depth_map = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_map = cv2.applyColorMap((depth_map.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    render = np.concatenate([depth[:, :, 0:1], depth_map], axis=-1) # (H, W, 4)
            elif render_mode == "n_contribs" or render_mode == "alpha":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda"))[0, :, :] # (H, W)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda"))[0, :, :].detach().cpu().numpy()  # (H, W)
            renders.append(render)
        if optimizable is False:
            renders = np.array(renders)
        else:
            renders = torch.stack(renders)
        return renders

    def optimize_cam_params(self, smplx_scale=None, look_target=None, device="cuda:0"):
        data_type = self.current_data_type

        cameras = None
        img = None
        bbox_masks = None

        maximum_mask_ratio = 0.3
        minimum_mask_ratio = 0.15
        camera_human_elevation_min_degree = 50.0
        camera_human_elevation_max_degree = 60.0

        pixel_margin = 50.0

        start_point = torch.tensor(self.start_point.value).to(device)
        end_point = torch.tensor(self.end_point.value).to(device)
        points_world = torch.stack([start_point, end_point]).to(device)
        if look_target == 'start':
            points_world = points_world[0:1, :]
        elif look_target == 'end':
            points_world = points_world[1:2, :]

        N = points_world.shape[0]

        center_point = (start_point + end_point) / 2
        path_vector = end_point - start_point
        path_length = torch.linalg.norm(path_vector)
        
        # Get arbitrary point on plane
        if self.world_up_vector is None:
            vx, vy, vz = path_vector
            x, y = 1.0, 0.0
            if vz != 0:
                z = -(vx*x + vy*y) / vz
            else:
                z = 1.0
            up_vector = torch.tensor([x, y, z]).float().to(device)
            up_vector = up_vector / torch.linalg.norm(up_vector)
            initial_point = center_point
            initial_alpha = torch.zeros(initial_point.shape).to(initial_point) + up_vector
            lr = 1e-1
        else:
            up_vector = self.world_up_vector.float().to(device)
            up_vector = up_vector / torch.linalg.norm(up_vector)
            look_dir = torch.cross(path_vector, up_vector, dim=-1)
            look_dir = look_dir / torch.linalg.norm(look_dir)
            if look_target == 'start':
                initial_point = start_point
            elif look_target == 'end':
                initial_point = end_point
            else:
                initial_point = center_point
            initial_alpha = look_dir * self.ref_dist * 2.5 + up_vector
            lr = 1e-1

        alpha = torch.tensor(initial_alpha, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        pbar = trange(1000, desc="Training", dynamic_ncols=True)
        for step in pbar:
            optimizer.zero_grad()

            trans = initial_point + alpha

            forward = (initial_point - trans) / torch.linalg.norm(initial_point - trans)
            right = torch.cross(forward, up_vector, dim=-1) / torch.linalg.norm(torch.cross(forward, up_vector, dim=-1))
            camera_up = torch.cross(forward, right, dim=-1)
            rotation_matrix = torch.stack([right, camera_up, forward], axis=1)

            K = torch.tensor(self.get_K(np.radians(75), self.render_wh), device=device).float()

            R = rotation_matrix.T[None] # w2c
            t = (-rotation_matrix.T @ trans)[None] # w2c
            fx = K[0][0][None]
            fy = K[1][1][None]
            cx = K[0][2][None].float()
            cy = K[1][2][None].float()
            width = torch.tensor(int(cx * 2), device=device, dtype=torch.int)[None]
            height = torch.tensor(int(cy * 2), device=device, dtype=torch.int)[None]
            appearance_id = torch.zeros(1, dtype=torch.int).to(device)
            normalized_appearance_id = torch.zeros(1, dtype=torch.int).to(device)
            distortion_params = torch.zeros((1, 4), dtype=torch.int).to(device)
            camera_type = torch.zeros(1, dtype=torch.int).to(device)

            cameras = Cameras(
                R=R,
                T=t,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                appearance_id=appearance_id,
                normalized_appearance_id=normalized_appearance_id,
                distortion_params=distortion_params,
                camera_type=camera_type
            )
            self.update_camera(self.server.get_clients()[0], cameras[0])

            depth = self.render(cameras, render_mode='exp_depth', optimizable=True) # (N, H, W)
            pc_cam = project_points_to_image(points_world, K, rotation_matrix, trans, include_depth=True).to(depth.device)
            points_to_camera = trans.unsqueeze(0) - points_world
            points_to_camera = points_to_camera / torch.norm(points_to_camera, dim=1, keepdim=True)

            loss = 0
            loss_info = {}

            if data_type != 'static':
                loss_center = torch.abs(pc_cam[0, -1] - pc_cam[1, -1])
                loss += loss_center
                loss_info["loss_center"] = f"{loss_center.item():.2f}"

                R_batch = R.repeat(N, 1, 1)
                t_batch = t.repeat(N, 1)
                focal_batch = torch.cat((fx, fy), dim=-1).repeat(N, 1)
                princpt_batch = torch.cat((cx, cy), dim=-1).repeat(N, 1)
                img = self.render(cameras, render_mode="rgb", optimizable=True) * 255
                human_renders, human_depths, human_masks, bb = self.render_smplx_at_3d_position(points_world, R_batch, t_batch, focal_batch, princpt_batch, img, smplx_scale)
                bbox_masks, bboxes = mask_to_bbox_mask(human_masks)
                bbox_masks /= 255.0

                target_depth = depth.repeat(N, 1, 1).unsqueeze(-1)
                target_depth[human_masks <= 0] = 0
                human_depths[human_masks <= 0] = 0

                total_pixel = width * height

                x_min = bb[:, 0].min()
                y_min = bb[:, 1].min()
                x_max = bb[:, 2].max()
                y_max = bb[:, 3].max()
                whole_bb = torch.stack([x_min, y_min, x_max, y_max]).unsqueeze(0).to(device)

                bb_widths = (bb[:, 2].clamp(0, width[0]) - bb[:, 0].clamp(0, width[0]))
                bb_heights = (bb[:, 3].clamp(0, height[0]) - bb[:, 1].clamp(0, height[0]))
                bb_areas = (bb_widths * bb_heights).unsqueeze(-1)
                
                # angles between points to camera vec and human up vec >= thresh
                loss_camera_over_points = margin_range_loss(torch.sum(points_to_camera * up_vector.unsqueeze(0), dim=-1), \
                                                                lower=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_max_degree))).to(device), \
                                                                upper=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_min_degree))).to(device), \
                                                            margin=0).mean()

                loss_human_mask_in_image = margin_range_loss(whole_bb, \
                                                                lower=torch.zeros((1, 4)).to(device), \
                                                            upper=torch.stack([width, height, width, height], dim=-1).to(device), \
                                                            margin=0).mean()
                loss_human_mask_area = margin_range_loss(bb_areas, \
                                                            lower=torch.stack([total_pixel * minimum_mask_ratio])[None].to(device), \
                                                        upper=torch.stack([total_pixel * maximum_mask_ratio])[None].to(device), \
                                                        margin=0.0).mean()
                loss_human_mask_visible = margin_range_loss(human_depths, \
                                                            lower=torch.zeros(human_depths.shape).to(device), \
                                                        upper=target_depth, \
                                                        margin=0.0).mean()
                
                loss += 10 * loss_camera_over_points + loss_human_mask_in_image + loss_human_mask_area + 10 * loss_human_mask_visible
                loss_info["loss_camera_over_points"] = f"{loss_camera_over_points.item():.2f}"
                loss_info["loss_human_mask_area"] = f"{loss_human_mask_area.item():.2f}"
                loss_info["loss_human_mask_in_image"] = f"{loss_human_mask_in_image.item():.2f}"
                loss_info["loss_human_mask_visible"] = f"{loss_human_mask_visible.item():.5f}"

            else:
                self.set_rgb_render_mode()
                screen_coord_x, screen_coord_y = pc_cam[:, 0], pc_cam[:, 1]
                pc_in_image = (screen_coord_x >= 0).all() & (screen_coord_x < self.render_wh[0]).all() & (screen_coord_y >= 0).all() & (screen_coord_y < self.render_wh[1]).all() & (pc_cam[:, 2] >= 0).all()
                loss_pc_in_image = margin_range_loss(pc_cam[:, :2], lower=torch.zeros((1, 2), device=device), upper=torch.cat([width, height], dim=0)[None].to(device), margin=pixel_margin).mean() + \
                                    margin_range_loss(pc_cam[:, 2], lower=torch.zeros((1, 2), device=device)).mean()

                loss += loss_pc_in_image
                loss_info["loss_pc_in_image"] = f"{loss_pc_in_image.item():.2f}"
                loss_camera_scene = depth.mean()

                if pc_in_image:

                    if len(pc_cam) == 2:
                        loss_center = torch.abs(pc_cam[0, -1] - pc_cam[1, -1])
                        loss += 10 * loss_center
                        loss_info["loss_center"] = f"{loss_center.item():.2f}"

                    depth_from_pc_positions = depth[:, screen_coord_y.int(), screen_coord_x.int()]
                    loss_visible_pc = margin_range_loss(pc_cam[:, 2], lower=torch.zeros((1, 2)).to(device), upper=depth_from_pc_positions, margin=0.01).mean()

                    if (pc_cam[:, 2] - depth_from_pc_positions > 0).all():
                        loss += 100 * loss_camera_scene
                        loss_info["loss_camera_scene"] = f"{loss_camera_scene.item():.2f}"

                    loss += 100 * loss_visible_pc
                    loss_info["loss_visible_pc"] = f"{loss_visible_pc.item():.2f}"

                if self.world_up_vector is not None:
                    loss_camera_over_points = margin_range_loss(torch.sum(points_to_camera * up_vector.unsqueeze(0), dim=-1), \
                                                                    lower=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_max_degree))).to(device), \
                                                                    upper=torch.cos(torch.deg2rad(torch.tensor(camera_human_elevation_min_degree))).to(device), \
                                                                margin=0).mean()
                    loss += 100 * loss_camera_over_points
                    loss_info["loss_camera_over_points"] = f"{loss_camera_over_points.item():.2f}"

            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 0:
                print("Initial Loss:", loss.item())

            loss_info["loss"] = f"{loss.item():.2f}"
            pbar.set_postfix(loss_info)
        
        self.set_rgb_render_mode()

        return cameras, img, bbox_masks
    
    def render_smplx_at_3d_position(self, points_world, R_w2c, t_w2c, focal, princpt, img, smplx_scale=1.0):
        """
        points_world : (N,3)
        R_w2c       : (N,3,3) world-to-camera rotation
        t_w2c       : (N,3)  world-to-camera translation
        focal       : (N,2) focal length
        princpt     : (N,2)
        img         : (N,H,W,3)
        smplx_scale : scalar


        mask        : (N,H,W)
        """
        device = points_world.device
        N = points_world.shape[0]
        R_w2c = R_w2c.to(device)
        t_w2c = t_w2c.to(device)
        focal = focal.to(device)
        princpt = princpt.to(device)
        K = torch.zeros((N, 3, 3), device=device)
        K[:, 0, 0] = focal[:, 0]
        K[:, 1, 1] = focal[:, 1]
        K[:, 0, 2] = princpt[:, 0]
        K[:, 1, 2] = princpt[:, 1]
        world_up_vector = self.world_up_vector[None, :].to(device)
        smpl_x.layer['neutral'] = smpl_x.layer['neutral'].to(device)

        
        smplx_scale = torch.tensor([smplx_scale]).float().to(device)
        points_cam = torch.einsum('nij,nj->ni', R_w2c, points_world) + t_w2c

        axis = np.array([1, 0, 0])
        angle = np.pi
        yz_inv_axis_angle = torch.tensor((axis * angle))[None].float().to(device)

        output = smpl_x.layer['neutral'](
                                global_orient = torch.zeros((N, 3)).to(device), \
                                body_pose = torch.zeros((N, 21, 3)).to(device), \
                                jaw_pose = torch.zeros((N, 3)).to(device), \
                                leye_pose = torch.zeros((N, 3)).to(device), \
                                reye_pose = torch.zeros((N, 3)).to(device), \
                                left_hand_pose = torch.zeros((N, 15, 3)).to(device), \
                                right_hand_pose = torch.zeros((N, 15, 3)).to(device), \
                                expression = torch.zeros((N, 10)).to(device), \
                                betas = torch.zeros((N, 100)).to(device), \
                                )
        mesh_cam = output.vertices * smplx_scale[None, None, :]
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[None, None, :]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]

        mesh_cam = mesh_cam - root_cam[:, None, :]
        kpt_cam = kpt_cam - root_cam[:, None, :]

        up_vector_from_world = torch.einsum('nij,nj->ni', R_w2c, world_up_vector.repeat(N, 1))
        up_vector_from_world = torch.nn.functional.normalize(up_vector_from_world, dim=-1)
        cam_up_vector = torch.tensor([0., 1., 0.], device=device).expand(N, -1)

        R_up_align = rotation_from_a_to_b(cam_up_vector, up_vector_from_world)

        verts = mesh_cam
        verts = torch.einsum('nij,ntj->nti', R_up_align, verts)  # (N,T,3)
        verts = verts + points_cam[:, None, :]

        cam_param = {
            'R': R_w2c,
            't': t_w2c,
            'focal': focal,
            'princpt': princpt
        }
        face = torch.tensor(smpl_x.face[None]).repeat((N, 1, 1)).to(device)
        img = torch.ones(img.shape).to(img) * 255
        images, depth, mask = render_smplx(verts, face, cam_param, img) # (N, H, W, 3), (N, H, W, 1), (N, H, W, 1)

        # min, max value of vertex in image coordinates
        verts_image = project_points_to_image(verts.reshape(-1, 3), K[0]).reshape(N, -1, 2)
        min_vals, _ = verts_image.min(dim=1)
        max_vals, _ = verts_image.max(dim=1)
        x_min = min_vals[:, 0].unsqueeze(-1)
        y_min = min_vals[:, 1].unsqueeze(-1)
        x_max = max_vals[:, 0].unsqueeze(-1)
        y_max = max_vals[:, 1].unsqueeze(-1)
        bb = torch.cat([x_min, y_min, x_max, y_max], dim=-1)

        # Debugging
        # imageio.imwrite(os.path.join(self.data_path, 'start_smpl.png'), images[0].detach().cpu().numpy().astype(np.uint8))
        # imageio.imwrite(os.path.join(self.data_path, 'end_smpl.png'), images[1].detach().cpu().numpy().astype(np.uint8))
        # imageio.imwrite(os.path.join(self.static_dir_path, 'start_mask.png'), bbox_mask[0].repeat(1, 1, 3).detach().cpu().numpy().astype(np.uint8))
        # imageio.imwrite(os.path.join(self.static_dir_path, 'end_mask.png'), bbox_mask[1].repeat(1, 1, 3).detach().cpu().numpy().astype(np.uint8))

        return images, depth, mask, bb
    
    def get_priors(self):
        self.set_path('static')
        with open(os.path.join(self.cam_param_path, '0.json'), 'r') as f:
            cam_param = json.load(f)
            R_w2c = torch.tensor(cam_param['R'])
            t_w2c = torch.tensor(cam_param['t'])
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c
        with open(self.smplx_scale_path, 'r') as f:
            smplx_scale = torch.tensor(json.load(f))
        with open(os.path.join(self.smplx_params_path, '0.json'), 'r') as f:
            smplx_param = json.load(f)
        with open(self.smplx_shape_path, 'r') as f:
            smplx_shape = json.load(f)
        device = 'cpu'
        smpl_x.layer['neutral'] = smpl_x.layer['neutral'].to(device)
        output = smpl_x.layer['neutral'](
            global_orient = torch.tensor(smplx_param['root_pose'])[None].to(device), \
            body_pose = torch.tensor(smplx_param['body_pose'])[None].to(device), \
            jaw_pose = torch.tensor(smplx_param['jaw_pose'])[None].to(device), \
            leye_pose = torch.tensor(smplx_param['leye_pose'])[None].to(device), \
            reye_pose = torch.tensor(smplx_param['reye_pose'])[None].to(device), \
            left_hand_pose = torch.tensor(smplx_param['lhand_pose'])[None].to(device), \
            right_hand_pose = torch.tensor(smplx_param['rhand_pose'])[None].to(device), \
            expression = torch.tensor(smplx_param['expr'])[None].to(device), \
            betas = torch.tensor(smplx_shape)[None].to(device), \
        )
        mesh_cam = output.vertices * smplx_scale[None, None, :]
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[None, None, :]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:].clone()

        axis = np.array([1, 0, 0])
        angle = np.pi
        yz_inv_rot = axis_angle_to_matrix(torch.tensor((axis * angle))[None].float())
        
        kpt_cam -= root_cam[:, None, :]
        kpt_cam = kpt_cam @ yz_inv_rot @ R_c2w.T + t_c2w[None, None, :]
        
        pelvis_kpt = kpt_cam[:, smpl_x.kpt['idx'][smpl_x.kpt['name'].index('Pelvis')], :]
        left_shoulder_kpt = kpt_cam[:, smpl_x.kpt['idx'][smpl_x.kpt['name'].index('L_Shoulder')], :]
        right_shoulder_kpt = kpt_cam[:, smpl_x.kpt['idx'][smpl_x.kpt['name'].index('R_Shoulder')], :]
        
        up_vector = (left_shoulder_kpt + right_shoulder_kpt) / 2 - pelvis_kpt
        up_vector = torch.nn.functional.normalize(up_vector, dim=-1)[0]
        # Match camera convention
        up_vector = up_vector

        return smplx_scale.item(), up_vector


    def sample_gaussians_from_points(self, query_points, N=100): 
        """ 
        query_points: (N, 3) 
        N : int 
        """ 
        means = self.means.detach().cpu().numpy() 
        dists = np.linalg.norm( 
            query_points[:, np.newaxis, :] - means[np.newaxis, :, :], axis=-1)
        idx = np.argsort(dists, axis=1)[:, :N] 
        return idx

    def sample_gaussian(self, mu, Sigma, N=1000):
        return np.random.multivariate_normal(mu, Sigma, size=N)

    def nn_distance(self, mu1, S1, mu2, S2, N=1000):
        from scipy.spatial import cKDTree
        pts1 = self.sample_gaussian(mu1, S1, N)
        pts2 = self.sample_gaussian(mu2, S2, N)
        tree2 = cKDTree(pts2)
        d1, _ = tree2.query(pts1)  # pts1 -> pts2
        tree1 = cKDTree(pts1)
        d2, _ = tree1.query(pts2)  # pts2 -> pts1
        return 0.5*(np.mean(d1) + np.mean(d2))  # symmetric NN distance

    def retrieve_gaussian_distribution(self, device='cuda', random_samples=1):
        from scipy.spatial.transform import Rotation as R

        means = self.means.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        scales_mat = np.zeros((scales.shape[0], 3, 3))
        scales_mat[:, np.arange(3), np.arange(3)] = scales
        rotations = R.from_quat(self.rotations.detach().cpu().numpy()[:, [1, 2, 3, 0]]).as_matrix() # (w, x, y, z) --> (x, y, z, w)

        N = len(means)
        num_sample = int(random_samples * N)
        indices = torch.randperm(N)[:num_sample]

        means = torch.tensor(means[indices]).float().to(device)
        scales = torch.tensor(scales[indices]).float().to(device)
        scales = torch.diag_embed(scales)
        rotations = torch.tensor(rotations[indices]).float().to(device)

        s = rotations @ scales @ scales.transpose(1, 2) @ rotations.transpose(1, 2)
        return means, s

    def gaussian_knn_median_p3d(self, M=5, K=1, seed=42, device='cuda'):
        """
        Args:
            mu: (N, 3) tensor, Gaussian mean
            cov: (N, 3, 3) tensor, Gaussian covariance. None이면 identity
            M: int, 각 Gaussian에서 샘플링할 점 수
            K: int, KNN 개수
            seed: int, 랜덤 시드
            device: 'cpu' or 'cuda'
        
        Returns:
            median_dists: (N, N) tensor, 각 Gaussian pair의 MxM 거리 median
            samples: (N, M, 3) tensor, 샘플 점
        """

        from pytorch3d.ops import knn_points

        if seed is not None:
            torch.manual_seed(seed)

        N = len(self.means)
        """
        mu, cov = self.retrieve_gaussian_distribution(device=device)
        print(cov.shape)
        
        mu = mu.to(device)
        N, D = mu.shape
        
        if cov is None:
            cov = torch.eye(D, device=device).unsqueeze(0).repeat(N, 1, 1)
        else:
            cov = cov.to(device)
        cov = cov + 1e-6 * torch.eye(3, device=cov.device)

        batch_size = 500_000
        all_samples = []
        for start in range(0, M, batch_size):
            end = min(M, start + batch_size)
            samples_batch = dist.rsample((end - start,))
            all_samples.append(samples_batch)
        samples = torch.cat(all_samples, dim=0)

        median_dists = torch.zeros(N, N, device=device)
        """
        samples = self.means
        
        for i in range(N):
            # query: samples[i] -> (1, M, 3)
            query = samples[i].unsqueeze(0)
            for j in range(N):
                # reference: samples[j] -> (1, M, 3)
                reference = samples[j].unsqueeze(0)
                
                # knn_points 사용 (return_sorted=True 기본)
                knn = knn_points(query, reference, K=M)  # (1, M, M)
                # knn.dists: (1, M, M) 제곱거리
                median_dists[i, j] = torch.median(torch.sqrt(knn.dists))
        
        return median_dists
    
    def median_self_knn(self, K=1, device='cpu'):
        """
        points: (N, 3) tensor
        K: nearest neighbors
        """
        from pytorch3d.ops import knn_points

        indices = self.opacities > 0.1
        points = self.means[indices[:, 0].bool()]
        points = points.to(device)

        points_batched = points.unsqueeze(0)  # (1, N, 3)
        
        knn = knn_points(points_batched, points_batched, K=K+1)  # 자기 자신 포함
        dists = torch.sqrt(knn.dists[:, :, 1:])  # 자기 자신 제외 (첫 번째가 자기 자신)
        
        # median
        median_dist = torch.median(dists)
        return median_dist

    def calculate_ref_distance(self):
        center_point = (np.array(self.start_point.value) + np.array(self.end_point.value)) / 2
        self.ref_dist = point_to_plane_distance(center_point, np.concatenate([self.world_up_vector, [self.plane_offset]]))
        return self.ref_dist

def depth_to_pc(depth_map, K, R_c2w=None, t_c2w=None, screen_pos=None):
    """
    depth_map   : (H,W)
    K           : (3,3) intrinsic matrix
    R_c2w       : (3,3) camera-to-world rotation
    t_c2w       : (3,)  camera-to-world translation
    screen_pos  : (2,)  screen positions

    points      : (H x W, 3)
    """
    H, W = depth_map.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=-1)
    if screen_pos is not None:
        screen_pos = (int(screen_pos[0] * W), int(screen_pos[1] * H))
        points = points[screen_pos[1], screen_pos[0], :]
    points = np.reshape(points, (-1, 3))
    if R_c2w is not None and t_c2w is not None:
        points = (R_c2w @ points.T).T + t_c2w
    return points

def project_points_to_image(points_world, K, R_c2w=None, t_c2w=None, include_depth=False):
    """
    points_world : (N,3)
    K           : (3,3) intrinsic matrix
    R_c2w       : (3,3) camera-to-world rotation
    t_c2w       : (3,)  camera-to-world translation

    uv          : (N,2) or (N,3) if include_depth=True
    """
    # world -> camera
    if R_c2w is not None and t_c2w is not None:
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        X_c = (R_w2c @ points_world.T).T + t_w2c  # (N,3)
    else:
        X_c = points_world
    
    z = X_c[:, 2]

    # homogeneous projection
    uvw = (K @ X_c.T).T / z[:, None]  # (N,3)

    if include_depth:
        uvw[:, 2] *= z
        uv = uvw
    else:
        uv = uvw[:, :2]
    
    return uv

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

def mask_to_bbox_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (N, H, W, 1) torch.uint8, binary human mask (0 or 255)

    return
    (N, H, W, 1) torch.uint8, bounding box mask
    (N, 4) torch.uint8, bboxes
    """
    N, H, W, _ = mask.shape
    bbox_masks = torch.zeros(mask.shape, device=mask.device).unsqueeze(-1)
    bboxes = []
    for i in range(N):
        ys, xs = torch.nonzero(mask[i, :, :, 0] > 0, as_tuple=True)
        if ys.numel() == 0 or xs.numel() == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        bboxes.append(torch.stack([x_min, y_min, x_max, y_max]))

        bbox_masks[i, y_min:y_max+1, x_min:x_max+1] = 255
    
    bboxes = torch.stack(bboxes).to(mask.device)

    return bbox_masks, bboxes

def render_smplx(mesh, face, cam_param, bkg, blend_ratio=1.0):
    render_shape = (int(cam_param['princpt'][0][1] * 2), int(cam_param['princpt'][0][0] * 2)) # height, width
    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )

    # render
    renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
    images, fragments = renderer(mesh, materials=materials)

    # background masking
    depth = fragments.zbuf
    is_fg = torch.relu(depth)
    images = images[:,:,:,:3] * 255 # (N, H, W, 3)
    mask = (is_fg * 255) # (N, H, W, 1)
    images = images * blend_ratio + bkg / 255 * (1 - blend_ratio)
    images = images * is_fg + bkg * torch.logical_not(is_fg)
    return images, depth, mask

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return (x1, y1, x2, y2)

def interpolate_bbox(bbox1, bbox2, num_steps):
    bboxes = []
    for t in range(num_steps + 1):
        alpha = t / num_steps
        interp = (1 - alpha) * np.array(bbox1) + alpha * np.array(bbox2)
        bboxes.append(interp.astype(int))
    return bboxes

def bbox_to_mask(bbox, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def rotation_for_billboarding(obj_pos, cam_pos):
    forward = obj_pos - cam_pos
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)

    up = torch.tensor([0.0, 1.0, 0.0])[None].float().to(forward.device)

    right = torch.cross(up, forward, dim=-1)
    right = right / torch.norm(right, dim=-1, keepdim=True)

    true_up = torch.cross(forward, right, dim=-1)

    R = torch.stack([right, true_up, forward], dim=1)  # 3x3
    return R

def rotation_from_a_to_b(a: torch.Tensor, b: torch.Tensor, eps=1e-6) -> torch.Tensor:

    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)

    v = torch.cross(a, b, dim=-1)  # 회전축
    c = torch.sum(a * b, dim=-1)   # 코사인 각
    s = v.norm(dim=-1)             # 사인 각

    # cross-product matrix
    def skew(v):
        zero = torch.zeros_like(v[..., 0])
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        return torch.stack([
            torch.stack([zero, -vz, vy], dim=-1),
            torch.stack([vz, zero, -vx], dim=-1),
            torch.stack([-vy, vx, zero], dim=-1)
        ], dim=-2)
    
    K = skew(v)
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[:-1] + (3,3))

    # Rodrigues formula
    R = I + K + K @ K * ((1 - c) / (s**2 + eps))[..., None, None]
    return R

def fit_plane_ransac(points, threshold=0.01, min_inliers=100):
    """
    points  : (N, 3)
    """
    # (x, y, z)
    X = points[:, :2]
    y = points[:, 2]

    ransac = RANSACRegressor(residual_threshold=threshold)
    ransac.fit(X, y)
    
    # z = ax + by + d → ax + by - z + d = 0
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_
    normal = np.array([a, b, -1.0])
    
    inlier_mask = ransac.inlier_mask_
    return normal, inlier_mask, d


def iterative_ransac_planes(points, ref_point, threshold=0.1, min_inliers=100, max_planes=1):
    """
    points  : (N, 3)
    """
    normals = []
    intercepts = []
    remaining = points.copy()

    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        normal, mask, intercept = fit_plane_ransac(remaining, threshold, min_inliers)
        inliers = remaining[mask]
        if len(inliers) < min_inliers:
            break

        # Direct ref point
        v = ref_point - inliers.mean(0)
        if np.dot(v, normal) < 0:
            normal = -normal

        normals.append(normal)
        intercepts.append(intercept)

        remaining = remaining[~mask]

    return np.array(normals), np.array(intercepts)

def extract_ground_normals(P, M, ref_point, threshold=0.1, min_inliers=100, max_planes=1):
    """
    P: (N, H, W, 3)
    M: (N, H, W, 1)
    ref_point: (2,)
    """
    all_normals = []
    all_intercepts = []

    for n in range(len(P)):
        points = P[n][M[n, ..., 0] > 0.5]
        if len(points) < min_inliers:
            all_normals.append([])
            continue

        normals, intercepts = iterative_ransac_planes(points, ref_point, threshold, min_inliers, max_planes)
        all_normals.append(normals)
        all_intercepts.append(intercepts)
    return all_normals, all_intercepts

def get_cam_params_on_arc(P1, P2, r_dist=1.0, bin=12):
    """
    P1      : (3,)
    P2      : (3,)
    r_dist  : int
    bin     : int

    c2w     : (bin - 1, 4, 4)
    """
    M = (P1 + P2) / 2

    v = P2 - P1
    v_norm = v / np.linalg.norm(v)

    vx, vy, vz = v_norm

    x, y = 1.0, 0.0
    if vz != 0:
        z = -(vx*x + vy*y) / vz
    else:
        z = 1.0

    u1 = np.array([x, y, z])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v_norm, u1)
    u2 /= np.linalg.norm(u2)

    theta = np.linspace(0, 2*np.pi, bin)[:-1]
    camera_positions = M + r_dist * (np.outer(np.cos(theta), u1) + np.outer(np.sin(theta), u2))

    rotation_matrices = []
    for C in camera_positions:
        forward = M - C
        forward /= np.linalg.norm(forward)

        right = v_norm
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        R = np.column_stack((right, up, forward))
        rotation_matrices.append(R)

    rotation_matrices = np.stack(rotation_matrices)

    c2w = np.zeros((bin - 1, 4, 4))
    c2w[:, :3, :3] = rotation_matrices
    c2w[:, :3, 3] = camera_positions
    return c2w

def spherical_kmeans(points, K=10, n_iter=1000):
    """
    points  : (N, 3)
    K       : int
    n_iter  : int

    centroids   : (K, 3)
    labels      : (M,)
    counts      : (M,)
    """
    N, D = points.shape
    centroids = points[np.random.choice(N, K, replace=False)].copy()

    for _ in range(n_iter):
        similarity = points @ centroids.T  # (N, K)
        labels = np.argmax(similarity, axis=1)  # (N,)

        for k in range(K):
            cluster_points = points[labels == k]
            if len(cluster_points) > 0:
                c = cluster_points.mean(axis=0)
                centroids[k] = c / np.linalg.norm(c)

    counts = np.bincount(labels)
    dominant_idx = np.argmax(counts)
    dominant_vector = centroids[dominant_idx]

    return dominant_vector, labels, counts, dominant_idx

def get_dominant_vector_from_dbscan(X, deg=10):
    """
    X   : (N, 3)

    dominant_vector : (3,)
    num_clusters    : int
    """
    db = DBSCAN(eps=1 - np.cos(np.deg2rad(deg)), min_samples=1, metric='cosine')
    db.fit(X)
    labels = db.labels_
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)

    largest_cluster = None
    largest_size = 0

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > largest_size:
            largest_size = len(cluster_points)
            largest_cluster = cluster_points
    
    dominant_vector = largest_cluster.mean(axis=0)
    dominant_vector /= np.linalg.norm(dominant_vector)

    return dominant_vector, num_clusters

def get_cam_params_by_rolling(c2w, bin=12):
    """
    c2w     : (4, 4)
    bin     : int

    new_c2w : (bin, 4, 4)
    """
    forward = c2w[:3, 2]
    right = c2w[:3, 0]
    up = c2w[:3, 1]

    new_c2w = torch.tensor(c2w)[None].repeat(bin, 1, 1)

    for i in range(bin):
        theta = np.deg2rad(360 // bin * i)
        right_new = np.cos(theta) * right + np.sin(theta) * up
        up_new    = -np.sin(theta) * right + np.cos(theta) * up
        
        rot = np.stack([right_new, up_new, forward], axis=1)
        rot = torch.tensor(rot)
        new_c2w[i, :3, :3] = rot
    return new_c2w

def check_pc_visibility(points, image_width, image_height, scene_depth):
    """
    points      : (N, 3)
    image_width : int
    image_height: int
    depth       : (H, W)

    visibility  : bool
    """
    N = points.shape[0]

    for i in range(N):
        x, y, z = points[i]
        ix, iy = int(round(x)), int(round(y))
        
        # Check whether points are in image boundary
        if not (0 <= ix < image_width and 0 <= iy < image_height):
            return False

        # Check occlusion with scene
        if z > scene_depth[iy, ix]:
            return False

    return True

def generate_cluster_colors(N):
    import matplotlib.colors as mcolors
    hues = np.linspace(0, 1, N, endpoint=False)
    saturation = 0.9
    value = 0.9
    colors = np.array([mcolors.hsv_to_rgb([h, saturation, value]) for h in hues])
    return colors

def find_dominant_up_vector(normals, top_n=5):
    """
    normals : (N, 3)
    """
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    cos_sim = normals @ normals.T  # (N, N)
    mean_sim = np.mean(cos_sim, axis=1)

    top_indices = np.argsort(mean_sim)[-top_n:]
    dominant_vec = normals[top_indices].mean(axis=0)
    
    return dominant_vec

def point_to_plane_distance(point, plane):
    """
    point: (3,) array_like, [x0, y0, z0]
    plane: (4,) array_like, [a, b, c, d]  -> ax + by + cz + d = 0
    
    return: float, distance between point and plane
    """
    point = np.asarray(point)
    plane = np.asarray(plane)
    a, b, c, d = plane
    x0, y0, z0 = point
    
    numerator = abs(a*x0 + b*y0 + c*z0 + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    return numerator / denominator

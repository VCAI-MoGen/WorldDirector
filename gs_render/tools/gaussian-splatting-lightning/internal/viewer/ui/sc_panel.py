import traceback
import datetime
import os.path
from typing import Tuple
from internal.cameras.cameras import Cameras

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re
import imageio
import json
import cv2

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

        self.render_wh = (720, 480)
        self.frame_num = 81
        self.render_mode = "rgb"
        self.root_path = self.viewer.root_path

        self._build_data_dir()

        self.start_point = (0.0, 0.0, 0.0)
        self.end_point = (0.0, 0.0, 0.0)

        self._setup_root_path()
        self._setup_traj_folder()
        self._setup_refine_folder()
        self._setup_vis_folder()

    def _setup_root_path(self):
        root_path_input = self.server.gui.add_text(
            "Root Path",
            initial_value=self.root_path,
            hint="Root Path for data"
        )
        @root_path_input.on_update
        def _(_) -> None:
            self.root_path = root_path_input.value

    def _setup_traj_folder(self):
        with self.server.gui.add_folder("Traj"):
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

                    pc = depth_to_world_pc(depth_value[:, :, 0], K, R, t, screen_pos[0]) # (N=1, 3)
                    self.start_point = pc[0].tolist()
                    print(f"start point: {self.start_point}")
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

                    pc = depth_to_world_pc(depth_value[:, :, 0], K, R, t, screen_pos[0]) # (N=1, 3)
                    self.end_point = pc[0].tolist()
                    print(f"end point: {self.end_point}")
                    scene.add_point_cloud(
                        name='end_point',
                        points=pc,
                        colors=(0.0, 0.0, 1.0),
                        point_size=0.01,
                        point_shape='circle'
                    )

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
                cameras = self.get_cameras_from_client(client, img_wh=self.render_wh)
                render_wh = self.render_wh

                # Render Depth
                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")[0]
                depth_value = depth[:, :, 0:1] # (H, W, 1)
                depth_map = depth[:, :, 1:].astype(np.uint8) # (H, W, 3)
                self.server.scene.set_global_visibility(True)
                
                depth_path = os.path.join(self.depth_vis_dir_static, '0.png')
                imageio.imwrite(depth_path, depth_map)
                depth_npy_path = os.path.join(self.depth_dir_static, '0.npy')
                np.save(depth_npy_path, depth_value)
                print(f"Save depth render in {depth_path}\nRender resolution is {depth_map.shape}")
                print(f"Save depth in {depth_npy_path}\nDepth resolution is {depth_value.shape}")
                print(f"Max depth: {depth_value.max()}")
                print(f"Min depth: {depth_value.min()}")
                
                for i in range(self.frame_num):
                    depth_path = os.path.join(self.depth_vis_dir_traj, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map)
                    depth_npy_path = os.path.join(self.depth_dir_traj, f'{i}.npy')
                    np.save(depth_npy_path, depth_value)

                # Render Image
                self.server.scene.set_global_visibility(False)
                image = self.render(cameras, render_mode="rgb")[0]
                self.server.scene.set_global_visibility(True)
                image_path = os.path.join(self.image_dir, 'render_image.png')
                imageio.imwrite(image_path, image)
                print(f"Save image in {image_path}\nRender resolution is {image.shape}")

                c2w, K = self.get_cam_params(client, render_wh)
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                points = np.array([self.start_point, self.end_point])
                uv = project_points_to_image(points, K, R, t)
                start_uv = uv[0].tolist()
                end_uv = uv[1].tolist()

                image_with_start_point = image.copy()
                cv2.circle(image_with_start_point, (int(start_uv[0]), int(start_uv[1])), 10, (255, 0, 0), -1)
                image_path = os.path.join(self.image_dir, 'start_point.png')
                imageio.imwrite(image_path, image_with_start_point)

                image_with_end_point = image.copy()
                cv2.circle(image_with_end_point, (int(end_uv[0]), int(end_uv[1])), 10, (0, 0, 255), -1)
                image_path = os.path.join(self.image_dir, 'end_point.png')
                imageio.imwrite(image_path, image_with_end_point)

                # Save start/end point
                points_path = os.path.join(self.image_dir, 'points.json')
                points = {
                    "start_point": self.start_point,
                    "end_point": self.end_point
                }
                with open(points_path, 'w') as f:
                    json.dump(points, f)
                print(f"Save points in {points_path}")
                print(f"Start point: {points["start_point"]}")
                print(f"End point: {points["end_point"]}")

                # Save camera parameters
                R = c2w[:3, :3].tolist()
                t = c2w[:3, 3].tolist()
                focal = [K[0, 0], K[1, 1]]
                princpt = [render_wh[0] // 2, render_wh[1] // 2]

                cam_param = {
                    "R": R,
                    "t": t,
                    "focal": focal,
                    "princpt": princpt
                }
                json_path = os.path.join(self.cam_param_path_static, "0.json")
                with open(json_path, 'w') as f:
                    json.dump(cam_param, f)
                print(f"Save cam_param in {json_path}")

                for i in range(self.frame_num):
                    json_path = os.path.join(self.cam_param_path_traj, f"{i}.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)
                
                print("Finish Dumping!")
    
    def _setup_refine_folder(self):
        with self.server.gui.add_folder("Refine"):
            dummy_button = self.server.gui.add_button("Dummy")

    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Visualization"):
            dummy_button = self.server.gui.add_button("Dummy")

    def _build_data_dir(self):
        root_path = self.root_path
        if not os.path.exists(root_path):
            print("Root Path does not exist.")
            return
        
        self.static_dir_path = os.path.join(root_path, 'static')
        self.traj_dir_path = os.path.join(root_path, 'traj')

        self.image_dir = os.path.join(self.static_dir_path, "scene_render_image")
        self.depth_dir_static = os.path.join(self.static_dir_path, "depth_scene")
        self.depth_vis_dir_static = os.path.join(self.static_dir_path, "depth_scene_vis")
        self.depth_dir_traj = os.path.join(self.traj_dir_path, "depth_scene")
        self.depth_vis_dir_traj = os.path.join(self.traj_dir_path, "depth_scene_vis")

        self.cam_param_path_static = os.path.join(self.static_dir_path, 'cam_params_scene')
        self.cam_param_path_traj = os.path.join(self.traj_dir_path, 'cam_params_scene')

        os.makedirs(self.static_dir_path, exist_ok=True)
        os.makedirs(self.traj_dir_path, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.depth_dir_static, exist_ok=True)
        os.makedirs(self.depth_vis_dir_static, exist_ok=True)
        os.makedirs(self.depth_dir_traj, exist_ok=True)
        os.makedirs(self.depth_vis_dir_traj, exist_ok=True)     
        os.makedirs(self.cam_param_path_static, exist_ok=True)
        os.makedirs(self.cam_param_path_traj, exist_ok=True)

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

        R = torch.tensor(c2w[:3, :3]).T[None].cuda() # w2c
        t = torch.tensor(-c2w[:3, :3].T @ c2w[:3, 3])[None].cuda() # w2c
        fx = torch.tensor(K[0][0])[None].cuda()
        fy = torch.tensor(K[1][1])[None].cuda()
        cx = torch.tensor(K[0][2])[None].cuda()
        cy = torch.tensor(K[1][2])[None].cuda()
        width = torch.tensor(int(cx * 2), dtype=torch.int)[None].cuda()
        height = torch.tensor(int(cy * 2), dtype=torch.int)[None].cuda()
        appearance_id = torch.zeros(1, dtype=torch.int).cuda(),
        normalized_appearance_id = torch.zeros(1, dtype=torch.int).cuda(),
        distortion_params = torch.zeros((1, 4), dtype=torch.int).cuda(),
        camera_type = torch.zeros(1, dtype=torch.int).cuda()

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
    
    def render(self, cameras: Cameras, render_mode="rgb"):
        available_outputs = self.renderer.renderer.get_available_outputs()

        output_type_info = available_outputs.get(render_mode, None)
        if output_type_info is None:
            return
        if render_mode == "exp_depth":
            output_type_info.visualizer = self.renderer.no_processing

        self.renderer._set_output_type(render_mode, output_type_info)
        renders = []
        for camera in cameras:
            render = self.renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
            if render_mode == "rgb":
                render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
            elif render_mode == "exp_depth":
                depth = render.transpose(1, 2, 0) # (H, W, 3)
                depth_map = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_map = cv2.applyColorMap((depth_map.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                render = np.concatenate([depth[:, :, 0:1], depth_map], axis=-1) # (H, W, 4)
            renders.append(render)
        return renders

def depth_to_world_pc(depth_map, K, R_c2w, t_c2w, screen_pos=None):
    """
    depth_map   : (H,W)
    K           : (3,3) intrinsic matrix
    R_c2w       : (3,3) camera-to-world rotation
    t_c2w       : (3,)  camera-to-world translation

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
    points = (R_c2w @ points.T).T + t_c2w
    return points

def project_points_to_image(points_world, K, R_c2w, t_c2w):
    """
    points_world : (N,3)
    K           : (3,3) intrinsic matrix
    R_c2w       : (3,3) camera-to-world rotation
    t_c2w       : (3,)  camera-to-world translation

    uv          : (N,2)
    """

    # world -> camera
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    X_c = (R_w2c @ points_world.T).T + t_w2c  # (N,3)

    z = X_c[:, 2]
    mask = z > 0

    # homogeneous projection
    uv = (K @ (X_c.T / z)).T  # (N,3)
    uv = uv[:, :2]

    uv[~mask] = np.nan

    return uv
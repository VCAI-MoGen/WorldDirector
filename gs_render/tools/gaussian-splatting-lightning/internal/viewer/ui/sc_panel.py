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
        self.data_type_list = ["static", "traj", "detailed"]

        self.current_data_type = self.data_type_list[0]

        self.running_mesh = False
        self.running_mesh_idx = 0
        
        self._build_data_dir()
        self._set_vis_path()

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
            self._build_data_dir()
            self._set_vis_path()

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

                cam_param_dir_path = self.cam_param_path_detailed
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)

                # Render Video
                self.server.scene.set_global_visibility(False)
                image = self.render(cameras, render_mode="rgb") # (N, H, W, 3)
                self.server.scene.set_global_visibility(True)
                render_video_path = os.path.join(self.detailed_dir_path, 'scene_render_video.mp4')
                imageio.mimsave(render_video_path, image, fps=16)
                print(f"Video saved in {render_video_path}")

                self.update_camera(client, cameras[0])

            dump_depth_video_button = self.server.gui.add_button(
                "Dump Depth Video",
                color="black",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Dump the trajectory as a depth video.",
            )
            @dump_depth_video_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                cam_param_dir_path = self.cam_param_path_detailed
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)

                # Render Depth
                self.server.scene.set_global_visibility(False)
                depth = self.render(cameras, render_mode="exp_depth")
                depth_value = depth[:, :, :, 0:1] # (N, H, W, 1)
                depth_map = depth[:, :, :, 1:].astype(np.uint8) # (N, H, W, 3)
                self.server.scene.set_global_visibility(True)

                self.update_camera(client, cameras[0])
                
                for i in range(depth.shape[0]):
                    depth_path = os.path.join(self.depth_vis_dir_detailed, f'{i}.png')
                    imageio.imwrite(depth_path, depth_map[i])
                    depth_npy_path = os.path.join(self.depth_dir_detailed, f'{i}.npy')
                    np.save(depth_npy_path, depth_value[i])
                print("Depths are saved...")

    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Visualization"):
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
                self._set_vis_path()

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

                cam_param_dir_path = self.cam_param_path_vis
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                traj_path_list = [os.path.join(self.char_traj_dir_path_vis, f'{i}.json') for i in range(len(os.listdir(self.char_traj_dir_path_vis)))]

                traj_list = []
                for traj_path in traj_path_list:
                    with open(traj_path, 'r') as f:
                        traj_cam = json.load(f)['traj']
                    traj_list.append(traj_cam)
                
                # traj_list = np.load(traj_path)
                traj = np.array(traj_list)
                traj = np.einsum('nij,nj->ni', c2w[:, :3, :3], traj) + c2w[:, :3, 3]
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
                
                cam_param_dir_path = self.cam_param_path_vis
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                mesh_path_list = [os.path.join(self.mesh_dir_path_vis, f'{i}.ply') for i in range(len(os.listdir(self.mesh_dir_path_vis)))]
                mesh_list = [trimesh.load_mesh(mesh_path).apply_transform(c2w[i]) for i, mesh_path in enumerate(mesh_path_list)]

                self.update_camera(client, cameras[0])

                import threading
                import time
                lock = threading.Lock()
                
                def render_next_mesh():
                    while self.running_mesh:
                        with lock:
                            i = self.running_mesh_idx
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

                            self.running_mesh_idx = (self.running_mesh_idx + 1) % len(mesh_path_list)

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

            vis_pc_from_depth_button = self.server.gui.add_button(
                "Visualize Point Cloud from Depth",
                color="orange",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Point Cloud from depth",
            )
            # TODO: add scene depth to pc with rgb
            @vis_pc_from_depth_button.on_click
            def _(event: viser.GuiEvent) -> None: 
                client = event.client
                assert client is not None

                cam_param_dir_path = self.cam_param_path_vis
                cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                cameras = self.get_cameras_from_json(cam_params_path)
                c2w = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
                focal = torch.stack([cameras.fx, cameras.fy], dim=1).detach().cpu().numpy()
                princpt = torch.stack([cameras.cx, cameras.cy], dim=1).detach().cpu().numpy()

                self.update_camera(client, cameras[0])

                video_frame_path = self.video_frame_path
                scene_depth_path = self.scene_depth_path
                optimized_depth_path = self.optimized_depth_path

                N = len(os.listdir(cam_params_path))
                H, W = cv2.imread(os.path.join(video_frame_path, '0.png')).shape[:2]

                scene_depths = np.array([np.load(os.path.join(scene_depth_path, f'{i}.npy'))[:, :, 0] for i in range(len(os.listdir(scene_depth_path)))])
                optimized_depths = np.array([cv2.resize(np.load(os.path.join(optimized_depth_path, f'{i}.npy')), (W, H)) for i in range(len(os.listdir(optimized_depth_path)))])
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

                    pc_rgb = video_frames[i].reshape(-1, 3)
                    pc_world = depth_to_world_pc(optimized_depths[i], K, c2w_R, c2w_t)
                    
                    client.scene.add_point_cloud(
                        name=f"pc_world_{i}",
                        points=pc_world,
                        colors=pc_rgb,
                        point_size=0.001
                    )

    def _build_data_dir(self):
        root_path = self.root_path
        if not os.path.exists(root_path):
            print("Root Path does not exist.")
            return
        
        self.static_dir_path = os.path.join(root_path, 'static')
        self.traj_dir_path = os.path.join(root_path, 'traj')
        self.detailed_dir_path = os.path.join(root_path, 'detailed')

        self.image_dir = os.path.join(self.static_dir_path, "scene_render_image")

        self.depth_dir_static = os.path.join(self.static_dir_path, "depths_scene")
        self.depth_vis_dir_static = os.path.join(self.static_dir_path, "depths_scene_vis")
        self.depth_dir_traj = os.path.join(self.traj_dir_path, "depths_scene")
        self.depth_vis_dir_traj = os.path.join(self.traj_dir_path, "depths_scene_vis")
        self.depth_dir_detailed = os.path.join(self.detailed_dir_path, "depths_scene")
        self.depth_vis_dir_detailed = os.path.join(self.detailed_dir_path, "depths_scene_vis")

        self.cam_param_path_static = os.path.join(self.static_dir_path, 'cam_params_scene')
        self.cam_param_path_traj = os.path.join(self.traj_dir_path, 'cam_params_scene')
        self.cam_param_path_detailed = os.path.join(self.detailed_dir_path, 'cam_params_scene')

        os.makedirs(self.static_dir_path, exist_ok=True)
        os.makedirs(self.traj_dir_path, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.depth_dir_static, exist_ok=True)
        os.makedirs(self.depth_vis_dir_static, exist_ok=True)
        os.makedirs(self.depth_dir_traj, exist_ok=True)
        os.makedirs(self.depth_vis_dir_traj, exist_ok=True)
        os.makedirs(self.depth_dir_detailed, exist_ok=True)
        os.makedirs(self.depth_vis_dir_detailed, exist_ok=True)
        os.makedirs(self.cam_param_path_static, exist_ok=True)
        os.makedirs(self.cam_param_path_traj, exist_ok=True)
        os.makedirs(self.cam_param_path_detailed, exist_ok=True)

    def _set_vis_path(self):
        self.cam_param_path_vis = os.path.join(self.root_path, self.current_data_type, 'cam_params_scene')
        self.mesh_dir_path_vis = os.path.join(self.root_path, self.current_data_type, "smplx_optimized", "meshes_smoothed")
        self.char_traj_dir_path_vis = os.path.join(self.root_path, self.current_data_type, "smplx_optimized", "traj_smoothed")
        self.scene_depth_path = os.path.join(self.root_path, self.current_data_type, 'depths_scene')
        self.optimized_depth_path = os.path.join(self.root_path, self.current_data_type, 'depths_vggt_optimized')
        # frame1_path = '/data/ckd248/SCVideo/HSI/fitting/data/Custom/data/bicycle3/scene_gs/test/frames_pure'
        self.video_frame_path = os.path.join(self.root_path, self.current_data_type, 'frames')
    
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
    
    def update_camera(self, client, camera: Camera):
        # world_to_camera is transposed. To get c2w, apply transpose -> inverse
        c2w = torch.linalg.inv(camera.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        print(c2w)
        focal = [camera.fx, camera.fy]
        princpt = [camera.cx, camera.cy]
        r = Rotation.from_matrix(R.tolist())
        quat = r.as_quat() # (x, y, z, w)
        quat = [quat[3], quat[0], quat[1], quat[2]] # (w, x, y, z)
        fov_radians = 2 * np.arctan(2 * princpt[1] / (2 * focal[1]))
        client.camera.fov = fov_radians
        client.camera.wxyz = quat
        client.camera.position = t

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
        renders = np.array(renders)
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
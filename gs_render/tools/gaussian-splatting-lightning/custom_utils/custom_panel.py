import os

import pycolmap
import numpy as np
import torch
import viser
import imageio
import json
import trimesh

from tqdm import tqdm
from .cameras import Camera, Cameras, build_cameras
from scipy.spatial.transform import Rotation

from .render_frustum import add_frustum_spline

class CustomPanel:
    def __init__(self, server, root_path, renderer=None, recon=None):
        self.server = server
        self.root_path = root_path
        self.renderer = renderer
        self.recon = recon

        self.is_frustum_visualized = {"colmap": False, "GS": False, "npz": False, "npz_w2c": False, "vae": False, "npy": False}
        self.frustum_handles = {"colmap": [], "GS": [], "npz": [], "npz_w2c": [], "vae": [], "npy": []}

        self.pcd = None

        self._setup_vis_folder()
    
    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Viz"):
            self.camera_type = self.server.gui.add_dropdown(
                "Camera Type",
                ["colmap", "GS", "npz", "npz_w2c", "vae", "npy"],
                initial_value="colmap",
            )

            self.fps = self.server.gui.add_number(
                "FPS",
                min=1,
                max=60,
                step=1,
                initial_value=5,
            )

            render_by_cam_button = self.server.gui.add_button(
                    "Render by Cam",
                    color="purple",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_by_cam_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    camera_type = self.camera_type.value
                    if camera_type == "colmap":
                        cameras = self.get_cameras_from_colmap()
                    elif camera_type == "GS":
                        cameras = self.get_cameras_from_json()
                    elif camera_type == "npz":
                        cameras = self.get_cameras_from_npz()
                    elif camera_type == "npz_w2c":
                        cameras = self.get_cameras_from_npz(is_w2c=True)
                    elif camera_type == "vae":
                        cameras = self.get_cameras_from_vae()
                    elif camera_type == "npy":
                        cameras = self.get_cameras_from_npy()
                    else:
                        raise ValueError("Camera type invalid.", camera_type)
                    renders = self.render(cameras, fps=self.fps.value)

                except Exception as e:
                    print("Error:", e)

            render_by_client_button = self.server.gui.add_button(
                    "Render by Client",
                    color="Brown",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_by_client_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    camera_type = self.camera_type.value
                    if camera_type == "colmap":
                        cameras = self.get_cameras_from_colmap()
                    elif camera_type == "GS":
                        cameras = self.get_cameras_from_json()
                    elif camera_type == "npz":
                        cameras = self.get_cameras_from_npz()
                    elif camera_type == "npz_w2c":
                        cameras = self.get_cameras_from_npz(is_w2c=True)
                    elif camera_type == "vae":
                        cameras = self.get_cameras_from_vae()
                    elif camera_type == "npy":
                        cameras = self.get_cameras_from_npy()
                    else:
                        raise ValueError("Camera type invalid.", camera_type)
                    self.render_with_client(event.client, cameras, self.root_path)
                except Exception as e:
                    print("Error:", e)

            self.frustum_range = self.server.gui.add_vector2(
                    "Frustum Range",
                    min=(0, 1),
                    max=(1000, 1000),
                    step=1,
                    initial_value=(0, 100),
                )

            self.frustum_interval = self.server.gui.add_number(
                    "Frustum Interval",
                    min=1,
                    max=50,
                    step=1,
                    initial_value=5,
                )

            self.frustum_scale = self.server.gui.add_number(
                    "Frustum Scale",
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    initial_value=0.1,
                )

            self.frustum_thickness = self.server.gui.add_number(
                    "Frustum Thickness",
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    initial_value=3,
                )

            self.frustum_start_color = self.server.gui.add_vector3(
                    "Frustum Start Color",
                    min=(0, 0, 0),
                    max=(1, 1, 1),
                    step=0.01,
                    initial_value=(1.0, 0.0, 0.0),
                )

            self.frustum_end_color = self.server.gui.add_vector3(
                    "Frustum End Color",
                    min=(0, 0, 0),
                    max=(1, 1, 1),
                    step=0.01,
                    initial_value=(0.0, 0.0, 1.0),
                )

            vis_camera_frustum_button = self.server.gui.add_button(
                    "Visualize Camera Frustum",
                    color="Green",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Visualize camera trajectory as point clouds.",
                )
            @vis_camera_frustum_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                
                
                camera_type = self.camera_type.value
                self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]
                cameras = None
                if camera_type == "colmap":
                    cameras = self.get_cameras_from_colmap()
                elif camera_type == "GS":
                    cameras = self.get_cameras_from_json()
                elif camera_type == "npz":
                    cameras = self.get_cameras_from_npz()
                elif camera_type == "npz_w2c":
                    cameras = self.get_cameras_from_npz(is_w2c=True)
                elif camera_type == "vae":
                    cameras = self.get_cameras_from_vae()
                elif camera_type == "npy":
                    cameras = self.get_cameras_from_npy()
                else:
                    raise ValueError("Camera type invalid.", camera_type)
            
                if cameras is None:
                    raise ValueError("Camera Not Found")

                fov, quat, position = camera_to_fov_quat_position(cameras[0])

                for handle_dict in self.frustum_handles[camera_type]:
                    for key, handle in handle_dict.items():
                        handle.remove()
                self.frustum_handles[camera_type].clear()

                if not self.is_frustum_visualized[camera_type]:
                    return

                f_min, f_max = self.frustum_range.value
                f_min = int(f_min)
                f_max = int(f_max)
                cam_min = f_min
                cam_max = min(f_max, len(cameras))

                start_color = np.array(self.frustum_start_color.value)
                end_color = np.array(self.frustum_end_color.value)
                t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (N, 1)
                colors = (1 - t) * start_color + t * end_color  # (N, 3)

                for i in range(cam_min, cam_max, self.frustum_interval.value):
                    camera = cameras[i]
                    fov, quat, position = camera_to_fov_quat_position(camera)

                    handle = add_frustum_spline(
                        scene=client.scene,
                        name=f"camera_{i}_frustum_{camera_type}",
                        fov_y=fov,
                        aspect=camera.cx / camera.cy,
                        wxyz=quat,
                        position=position,
                        far=1.0,
                        scale=self.frustum_scale.value,
                        thickness=self.frustum_thickness.value,
                        color=colors[i],
                    )

                    self.frustum_handles[camera_type].append(handle)

            self.pcd_type = self.server.gui.add_dropdown(
                "Point Cloud Type",
                ["colmap", "ply", "npy", "pt", "pt_input"],
                initial_value="pt",
            )
            self.pcd_pt_path = self.server.gui.add_text(
                "Point Cloud pt path",
                initial_value="Hello world",
            )

            vis_point_cloud_button = self.server.gui.add_button(
                "Visualize Point Cloud",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Colmap.",
            )
            @vis_point_cloud_button.on_click
            def _(event: viser.GuiEvent) -> None:
                if self.pcd is not None:
                    self.pcd.remove()
                    self.pcd = None
                    return
                
                if self.pcd_type.value == "colmap":
                    points, colors = self.get_pcd_from_colmap()
                elif self.pcd_type.value == "ply":
                    points, colors = self.get_pcd_from_ply()
                elif self.pcd_type.value == "npy":
                    points, colors = self.get_pcd_from_npy()
                elif self.pcd_type.value == "pt":
                    points, colors = self.get_pcd_from_pt()
                elif self.pcd_type.value == "pt_input":
                    points, colors = self.get_pcd_from_pt(self.pcd_pt_path.value)

                self.pcd = self.server.scene.add_point_cloud(
                    name="/point_cloud",
                    points=points,
                    colors=colors,
                    point_size=0.01,
                )
            self.traj_path = self.server.gui.add_text(
                "Trajectory path",
                initial_value="Hello world",
            )

            vis_camera_frustums_button = self.server.gui.add_button(
                    "Visualize Camera Frustums",
                    color="Green",
                    icon=viser.Icon.PLAYER_PLAY,
                )
            @vis_camera_frustums_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                camera_type = self.camera_type.value
                self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]
                camera_dir_path = "/data2/ckd248/SCVideo/camera_generation/infer_vae/20260204_043019"
                camera_paths = [os.path.join(camera_dir_path, camera_path) for camera_path in os.listdir(camera_dir_path)]
                camera_paths = sorted(
                    camera_paths,
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-2])
                )
                # camera_paths = camera_paths[::2]
                print(camera_paths)
                for j, camera_path in enumerate(camera_paths):
                    self.traj_path.value = camera_path 
                    print(self.traj_path)
                    cameras = self.get_cameras_from_npy()

                    fov, quat, position = camera_to_fov_quat_position(cameras[0])
                    self.frustum_handles[camera_type].clear()

                    if not self.is_frustum_visualized[camera_type]:
                        return

                    f_min, f_max = self.frustum_range.value
                    f_min = int(f_min)
                    f_max = int(f_max)
                    cam_min = f_min
                    cam_max = min(f_max, len(cameras))

                    start_color = np.array([1.0, 0.0, 0.0])
                    end_color = np.array([0.0, 0.0, 1.0])
                    t = np.linspace(0, 1, len(camera_paths))[:, None]  # (N, 1)
                    colors = (1 - t) * start_color + t * end_color  # (N, 3)

                    for i in range(cam_min, cam_max, self.frustum_interval.value):
                        camera = cameras[i]
                        fov, quat, position = camera_to_fov_quat_position(camera)

                        handle = add_frustum_spline(
                            scene=client.scene,
                            name=f"camera_{i}_{j}_frustum_{camera_type}",
                            fov_y=fov,
                            aspect=camera.cx / camera.cy,
                            wxyz=quat,
                            position=position,
                            far=1.0,
                            scale=self.frustum_scale.value,
                            thickness=self.frustum_thickness.value,
                            color=colors[j],
                        )

                        self.frustum_handles[camera_type].append(handle)
            capture_camerea_button = self.server.gui.add_button(
                    "Capture Current Camera as First Frame",
                    color="Black",
                    icon=viser.Icon.PLAYER_PLAY,
                )
            @capture_camerea_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                camera = client.camera
                wxyz = camera.wxyz
                t = camera.position
                R = Rotation.from_quat(wxyz[[[1, 2, 3, 0]]]).as_matrix()[0]
                E0 = np.eye(4)
                E0[:3, :3] = R.T
                E0[:3, 3] = -R.T @ t

                save_path = os.path.join(self.root_path, "e0.json")
                with open(save_path, 'w') as f:
                    json.dump(E0.tolist(), f)
                
                print(E0)
                print(f"Capture saved in : {save_path}")
                

    def get_cameras_from_colmap(self):
        if self.recon is None:
            return None
        extrinsics, intrinsics = get_colmap_camera_params(self.recon)
        cameras = build_cameras(extrinsics, intrinsics)
        return cameras

    def get_cameras_from_json(self):
        w2c_ext, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

    def get_cameras_from_npz(self, is_w2c=False):
        cam_params = np.load(self.traj_path.value, allow_pickle=True)
        
        intrinsics = cam_params['intrinsics']

        if is_w2c:
            w2c = cam_params['poses']
        else:
            c2w = cam_params['poses']
            N = c2w.shape[0]

            R_c2w = c2w[:, :3, :3]
            t_c2w = c2w[:, :3, 3:]

            R_w2c = np.transpose(R_c2w, (0, 2, 1))
            t_w2c = (-R_w2c @ t_c2w).squeeze(-1)

            w2c = np.tile(np.eye(4), (N, 1, 1))
            w2c[:, :3, :3] = R_w2c
            w2c[:, :3, 3] = t_w2c

        # cameras = build_cameras(extrinsics, intrinsics)
        cameras = build_cameras(w2c, intrinsics)
        return cameras
    
    def get_cameras_from_vae(self):
        _, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))
        w2c_ext = np.load(f"{self.root_path}/vae4.npy", allow_pickle=True)
        N = w2c_ext.shape[0]
        intrinsics = intrinsics[:N]
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras
    
    def get_cameras_from_npy(self):
        _, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))
        w2c_ext = np.load(self.traj_path.value, allow_pickle=True) # (N, 4, 4)
        print(w2c_ext.shape)
        N = w2c_ext.shape[0]
        intrinsics = intrinsics[:N]
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

    def get_pcd_from_colmap(self):
        points = []
        colors = []
        for p in self.recon.points3D.values():
            points.append(p.xyz)
            colors.append(p.color / 255.0)
        points = np.asarray(points)
        colors = np.asarray(colors)

        return points, colors

    def get_pcd_from_ply(self):
        ply_path = os.path.join(self.root_path, "scene.ply")
        g = trimesh.load(ply_path, process=False)
        if isinstance(g, trimesh.PointCloud):
            points = np.asarray(g.vertices, dtype=np.float32)
            colors = getattr(g, "colors", None)
            if colors is not None:
                colors = np.asarray(colors, dtype=np.uint8)[:, :3]
        else:
            points = np.asarray(g.vertices, dtype=np.float32)
            colors = None
        
        return points, colors

    def get_pcd_from_npy(self):
        point_cloud_path = os.path.join(self.root_path, "point_cloud.npy")
        points = np.load(point_cloud_path)
        colors = np.repeat([[0., 0., 1.0]], len(points), axis=0)

        return points, colors
    
    def get_pcd_from_pt(self, point_cloud_path=None):
        if point_cloud_path is None:
            point_cloud_path = os.path.join(self.root_path, "scene.pt")
        point_clouds = torch.load(point_cloud_path).numpy()
        points = point_clouds[:, :3]
        colors = point_clouds[:, 3:]

        return points, colors

    def render_with_client(self, client, cameras: Cameras, out_dir):
        images = []
        print("Rendering...")
        for camera in tqdm(cameras):
            fov, quat, t = camera_to_fov_quat_position(camera)
            img = client.get_render(
                height=camera.height.item(),
                width=camera.width.item(),
                wxyz=quat,
                position=t,
                fov=fov,
            )
            images.append(img)

        save_path = os.path.join(out_dir, "colmap_scene_viz.mp4")
        imageio.mimsave(save_path, images, fps=10)
        print("Rendering done. Saved to", save_path)
    
    def render(self, cameras: Cameras, render_mode="rgb", optimizable=False, fps=10):
        print("Setting Render")
        print(self.renderer.renderer)
        available_outputs = self.renderer.renderer.get_available_outputs()

        output_type_info = available_outputs.get(render_mode, None)
        if output_type_info is None:
            return
        if render_mode == "exp_depth" or "n_contribs" or "alpha":
            output_type_info.visualizer = self.renderer.no_processing
        
        self.renderer._set_output_type(render_mode, output_type_info)
        renders = []
        print("Rendering...")
        for camera in tqdm(cameras):
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

        save_path = os.path.join(self.root_path, "scene_viz.mp4")
        imageio.mimsave(save_path, renders, fps=fps)

        if optimizable is False:
            renders = np.array(renders)
        else:
            renders = torch.stack(renders)
        
        print("Rendering done. Saved to", save_path)

        return renders

def camera_to_fov_quat_position(camera: Camera):
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

def get_colmap_camera_params(recon: pycolmap.Reconstruction):
    """
    Output:
        extrinsics: [N, 4, 4] numpy array (w2c)
        intrinsics: [N, 3, 3] numpy array (w2c)
    """
    sorted_images = sorted(
        recon.images.values(),
        key=lambda img: img.name
    )

    extrinsic_list = []
    intrinsic_list = []
    for image in sorted_images:
        extrinsic_dict = image.cam_from_world().todict()

        xyzw_w2c, t_w2c = extrinsic_dict['rotation']['quat'], extrinsic_dict['translation']
        R_w2c = Rotation.from_quat(xyzw_w2c).as_matrix()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = image.camera.params[:4]
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_camera_params_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    extrinsic_list = []
    intrinsic_list = []
    for params in data:
        R_w2c, t_w2c = params['rotation'], params['position']

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_camera_params_from_json_deprecated(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    sorted_params = sorted(
        data,
        key=lambda x: x['img_name']
    )

    extrinsic_list = []
    intrinsic_list = []
    for params in sorted_params:
        R_c2w, t_c2w = params['rotation'], params['position']

        R_w2c = np.array(R_c2w).T
        t_w2c = (- R_w2c @ np.array(t_c2w).reshape(-1, 1)).squeeze(-1)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy = params['fx'], params['fy']

        if 'cx' not in params.keys():
            scale = 4
            fx /= scale
            fy /= scale
            width, height = params['width'], params['height']
            width /= scale
            height /= scale
            cx, cy = width/2.0, height/2.0
        else:
            cx, cy = params['cx'], params['cy']

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics


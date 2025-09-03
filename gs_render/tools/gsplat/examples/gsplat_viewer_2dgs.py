import viser
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable
from nerfview import Viewer, RenderTabState
import os
import numpy as np
import imageio
import json
import trimesh
import cv2


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    # LCY
    root_path: str = "/data/ckd248/SCVideo/data/room"
    running_mesh: bool = False
    running_mesh_idx: int = 0
    start_point: Tuple[float, float] = (0.0, 0.0)
    end_point: Tuple[float, float] = (0.0, 0.0)


class GsplatViewer(Viewer):
    """
    Viewer for gsplat 2dgs.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        render_res: Tuple = (480, 720)
    ):
        self.render_res = render_res
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat 2dgs viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")
        self.render_tab_state.render_height, self.render_tab_state.render_width = self.render_res

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth", "normal", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)
                
                # LCY
                root_path_input = server.gui.add_text(
                    "Root Path",
                    initial_value=self.render_tab_state.root_path,
                    hint="Root Path for data"
                )
                @root_path_input.on_update
                def _(_) -> None:
                    self.render_tab_state.root_path = root_path_input.value

                pick_start_point_button = server.gui.add_button(
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
                        screen_pos = event.screen_pos
                        self.render_tab_state.start_point = screen_pos[0]
                        print(f"start point: {self.render_tab_state.start_point}")
                        scene.remove_pointer_callback()
                    
                
                pick_end_point_button = server.gui.add_button(
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
                        screen_pos = event.screen_pos
                        self.render_tab_state.end_point = screen_pos[0]
                        print(f"end point: {self.render_tab_state.end_point}")
                        scene.remove_pointer_callback()

                dump_image_with_start_end_button = server.gui.add_button(
                    "Dump Image (Start, End Points)",
                    color="green",
                    icon=viser.Icon.LIVE_PHOTO,
                    hint="Dump the current render image as an image with start point and end point.",
                )
                @dump_image_with_start_end_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    client = event.client
                    assert client is not None
                    root_path = self.render_tab_state.root_path
                    self.render_tab_state.render_mode = "rgb"
                    camera_state = self.get_camera_state(client) # c2w
                    image_dir = os.path.join(root_path, "scene_render_image")
                    os.makedirs(image_dir, exist_ok=True)
                    # server.scene.set_global_visibility(False)
                    image = client.get_render(
                        height=self.render_tab_state.render_height,
                        width=self.render_tab_state.render_width,
                    )
                    image_path = os.path.join(image_dir, 'render_image.png')
                    imageio.imwrite(image_path, image)
                    print(f"Save image in {image_path}\nRender resolution is {image.shape}")

                    H, W, _ = image.shape
                    start_point = self.render_tab_state.start_point
                    end_point = self.render_tab_state.end_point

                    image_with_start_point = image.copy()
                    cv2.circle(image_with_start_point, (int(W * start_point[0]), int(H * start_point[1])), 10, (255, 0, 0), -1)
                    image_path = os.path.join(image_dir, 'start_point.png')
                    imageio.imwrite(image_path, image_with_start_point)

                    image_with_end_point = image.copy()
                    cv2.circle(image_with_end_point, (int(W * end_point[0]), int(H * end_point[1])), 10, (255, 0, 0), -1)
                    image_path = os.path.join(image_dir, 'end_point.png')
                    imageio.imwrite(image_path, image_with_end_point)

                    cam_param_path = os.path.join(root_path, 'cam_params_scene')
                    os.makedirs(cam_param_path, exist_ok=True)

                    height = self.render_tab_state.render_height
                    width = self.render_tab_state.render_width
                    c2w = camera_state.c2w
                    K = camera_state.get_K((width, height))

                    R = c2w[:3, :3].tolist()
                    t = c2w[:3, 3].tolist()
                    focal = [K[0, 0], K[1, 1]]
                    princpt = [width // 2, height // 2]

                    cam_param = {
                        "R": R,
                        "t": t,
                        "focal": focal,
                        "princpt": princpt
                    }
                    for i in range(81):
                        json_path = os.path.join(cam_param_path, f"{i}.json")
                        with open(json_path, 'w') as f:
                            json.dump(cam_param, f)
                    print(f"Save cam_param in {cam_param_path}")
                    
                load_cam_params_button = server.gui.add_button(
                    "Load Camera Parameters",
                    color="yellow",
                    icon=viser.Icon.FOLDER_OPEN,
                    hint="Load a camera parameter",
                )

                @load_cam_params_button.on_click
                def _(event: viser.GuiEvent) -> None: 
                    client = event.client
                    assert client is not None

                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, "cam_params_scene")
                    cam_param_path_list = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    cam_params_list = []
                    for cam_param_path in cam_param_path_list:
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        cam_params_list.append(cam_param)
                    R_c2w = np.array([cam_param['R'] for cam_param in cam_params_list]) # c2w
                    t_c2w = np.array([cam_param['t'] for cam_param in cam_params_list]) # c2w
                    focal = np.array([cam_param['focal'] for cam_param in cam_params_list])
                    princpt = np.array([cam_param['princpt'] for cam_param in cam_params_list])

                    self.update_camera(client, R_c2w[0], t_c2w[0], focal[0], princpt[0])

                dump_image_button = server.gui.add_button(
                    "Dump Image",
                    color="green",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Dump the current render image as an image.",
                )
                @dump_image_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    client = event.client
                    assert client is not None
                    self.render_tab_state.render_mode = "rgb"
                    camera_state = self.get_camera_state(client) # c2w
                    image_dir = os.path.join(self.output_dir, "image")
                    os.makedirs(image_dir, exist_ok=True)
                    # server.scene.set_global_visibility(False)
                    image = client.get_render(
                        height=self.render_tab_state.render_height,
                        width=self.render_tab_state.render_width,
                    )
                    image_path = os.path.join(image_dir, 'render_image.png')
                    imageio.imwrite(image_path, image)
                    print(f"Save image in {image_path}\nRender resolution is {image.shape}")

                    height = self.render_tab_state.render_height
                    width = self.render_tab_state.render_width
                    c2w = camera_state.c2w
                    K = camera_state.get_K((width, height))

                    R = c2w[:3, :3].tolist()
                    t = c2w[:3, 3].tolist()
                    focal = [K[0, 0], K[1, 1]]
                    princpt = [width // 2, height // 2]

                    cam_param = {
                        "R": R,
                        "t": t,
                        "focal": focal,
                        "princpt": princpt
                    }
                    json_path = os.path.join(image_dir, "cam_param.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)
                    print(f"Save cam_param in {json_path}")
                    
                dump_depth_button = server.gui.add_button(
                    "Dump depth",
                    color="red",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Dump the current render depth as an image.",
                )
                @dump_depth_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    self.render_tab_state.render_mode = "depth(precise)"
                    
                    client = event.client
                    assert client is not None
                    camera_state = self.get_camera_state(client) # c2w
                    depth_dir = os.path.join(self.output_dir, "depth")
                    os.makedirs(depth_dir, exist_ok=True)
                    server.scene.set_global_visibility(False)
                    self.render_tab_state.preview_render = True
                    depth = self.render_fn(camera_state, self.render_tab_state)
                    self.render_tab_state.preview_render = False
                    depth_value = depth[:, :, 3:4]
                    depth = depth[:, :, 0:3]
                    
                    depth_path = os.path.join(depth_dir, 'render_depth.png')
                    depth = (depth * 255).astype(np.uint8)
                    imageio.imwrite(depth_path, depth)
                    print(f"Save depth render in {depth_path}\nRender resolution is {depth.shape}")

                    depth_npy_path = os.path.join(depth_dir, 'depth.npy')
                    np.save(depth_npy_path, depth_value)
                    print(f"Save depth in {depth_npy_path}\nDepth resolution is {depth_value.shape}")
                    print(f"Max depth: {depth_value.max()}")
                    print(f"Min depth: {depth_value.min()}")


                    height = self.render_tab_state.render_height
                    width = self.render_tab_state.render_width
                    c2w = camera_state.c2w
                    K = camera_state.get_K((width, height))

                    R = c2w[:3, :3].tolist()
                    t = c2w[:3, 3].tolist()
                    focal = [K[0, 0], K[1, 1]]
                    princpt = [width // 2, height // 2]

                    cam_param = {
                        "R": R,
                        "t": t,
                        "focal": focal,
                        "princpt": princpt
                    }
                    json_path = os.path.join(depth_dir, "cam_param.json")
                    with open(json_path, 'w') as f:
                        json.dump(cam_param, f)
                    print(f"Save cam_param in {json_path}")

                    self.render_tab_state.render_mode = "rgb"

                vis_traj_button = server.gui.add_button(
                    "Visualize Trajectory",
                    color="blue",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Visualize trajectory as point clouds.",
                )
                @vis_traj_button.on_click
                def _(event: viser.GuiEvent) -> None: 
                    client = event.client
                    assert client is not None

                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, "cam_params_scene")
                    cam_param_path_list = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    cam_params_list = []
                    for cam_param_path in cam_param_path_list:
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        cam_params_list.append(cam_param)
                    R_c2w = np.array([cam_param['R'] for cam_param in cam_params_list]) # c2w
                    t_c2w = np.array([cam_param['t'] for cam_param in cam_params_list]) # c2w
                    focal = np.array([cam_param['focal'] for cam_param in cam_params_list])
                    princpt = np.array([cam_param['princpt'] for cam_param in cam_params_list])

                    self.update_camera(client, R_c2w[0], t_c2w[0], focal[0], princpt[0])

                    traj_dir_path = os.path.join(self.render_tab_state.root_path, "smplx_optimized/traj_smoothed")
                    traj_path_list = [os.path.join(traj_dir_path, f'{i}.json') for i in range(len(os.listdir(traj_dir_path)))]

                    traj_list = []
                    for traj_path in traj_path_list:
                        with open(traj_path, 'r') as f:
                            traj_cam = json.load(f)['traj']
                        traj_list.append(traj_cam)
                    """
                    traj_dir_path = "/home/ckd248/data/HSI/fitting/data/Custom/data/bicycle3/smplx_optimized/traj"
                    traj_path_list = [os.path.join(traj_dir_path, f'{i}.json') for i in range(len(os.listdir(traj_dir_path)))]

                    for traj_path in traj_path_list:
                        with open(traj_path, 'r') as f:
                            traj_cam = json.load(f)['traj']
                        traj_list.append(traj_cam)
                    """
                    # traj_list = np.load(traj_path)
                    traj = np.array(traj_list)
                    traj = np.einsum('nij,nj->ni', R_c2w, traj) + t_c2w
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
                    """
                    from plyfile import PlyData

                    plydata = PlyData.read('/home/ckd248/data/gs_render/tools/3DGS-to-PC/3dgs_pc.ply')
                    vertex = plydata['vertex']

                    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T  # (N,3)

                    N = points.shape[0]
                    sample_size = int(N * 0.05)
                    indices = np.random.choice(N, size=sample_size, replace=False)

                    if 'red' in vertex.data.dtype.names:
                        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0  # (N,3), 0~1 float
                    else:
                        colors = None
                    client.scene.add_point_cloud(
                        name="gs pc",
                        points=points[indices],       # (N, 3)
                        colors=colors[indices],     # (N, 3) in [0, 1]
                        point_size=0.005
                    )
                    print("Visualize Point Clouds")
                    """
                play_SMPL_button = server.gui.add_button(
                    "Play SMPL",
                    color="white",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Play SMPL-X as mesh.",
                )

                @play_SMPL_button.on_click
                def _(event: viser.GuiEvent) -> None: 
                    client = event.client
                    assert client is not None
                    
                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, "cam_params_scene")
                    cam_param_path_list = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    cam_params_list = []
                    for cam_param_path in cam_param_path_list:
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        cam_params_list.append(cam_param)
                    R_c2w = np.array([cam_param['R'] for cam_param in cam_params_list]) # c2w
                    t_c2w = np.array([cam_param['t'] for cam_param in cam_params_list]) # c2w
                    focal = np.array([cam_param['focal'] for cam_param in cam_params_list])
                    princpt = np.array([cam_param['princpt'] for cam_param in cam_params_list])

                    self.update_camera(client, R_c2w[0], t_c2w[0], focal[0], princpt[0])

                    mesh_dir_path = os.path.join(self.render_tab_state.root_path, "smplx_optimized/meshes_smoothed")
                    mesh_path_list = [os.path.join(mesh_dir_path, f'{i}.ply') for i in range(len(os.listdir(mesh_dir_path)))]

                    import threading
                    import time
                    lock = threading.Lock()
                    
                    def render_next_mesh():
                        while self.render_tab_state.running_mesh:
                            with lock:
                                i = self.render_tab_state.running_mesh_idx
                                mesh_path = mesh_path_list[i]
                                mesh = trimesh.load_mesh(mesh_path)

                                c2w = np.eye(4)
                                c2w[:3, :3] = R_c2w[i]
                                c2w[:3, 3] = t_c2w[i]

                                mesh.apply_transform(c2w)
                                mesh_handle = client.scene.add_mesh_trimesh(
                                    mesh_path,
                                    mesh=mesh
                                )

                                self.render_tab_state.running_mesh_idx = (self.render_tab_state.running_mesh_idx + 1) % len(mesh_path_list)

                            time.sleep(1 / 16)  # 1초 간격
                            mesh_handle.remove()
                    
                    def start_render():
                        if not self.render_tab_state.running_mesh:
                            self.render_tab_state.running_mesh = True
                            timer_thread = threading.Thread(target=render_next_mesh, daemon=True)
                            timer_thread.start()
                    start_render()

                    print("Play SMPL-X")
                
                stop_playing_SMPL_button = server.gui.add_button(
                    "Stop Playing SMPL",
                    color="red",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Stop Playing SMPL-X as mesh.",
                )

                @stop_playing_SMPL_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    self.render_tab_state.running_mesh = False
                
                vis_SMPL_button = server.gui.add_button(
                    "Visualize SMPL",
                    color="white",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Visualize SMPL-X as mesh.",
                )

                @vis_SMPL_button.on_click
                def _(event: viser.GuiEvent) -> None: 
                    client = event.client
                    assert client is not None
                    
                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, "cam_params_scene")
                    cam_param_path_list = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    cam_params_list = []
                    for cam_param_path in cam_param_path_list:
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        cam_params_list.append(cam_param)
                    R_c2w = np.array([cam_param['R'] for cam_param in cam_params_list]) # c2w
                    t_c2w = np.array([cam_param['t'] for cam_param in cam_params_list]) # c2w
                    focal = np.array([cam_param['focal'] for cam_param in cam_params_list])
                    princpt = np.array([cam_param['princpt'] for cam_param in cam_params_list])

                    self.update_camera(client, R_c2w[0], t_c2w[0], focal[0], princpt[0])

                    mesh_dir_path = os.path.join(self.render_tab_state.root_path, "smplx_optimized/meshes_smoothed")
                    mesh_path_list = [os.path.join(mesh_dir_path, f'{i}.ply') for i in range(len(os.listdir(mesh_dir_path)))]

                    indices_to_show = [0, 26, 42, 80]
                    # indices_to_show = [i for i in range(len(mesh_path_list))]

                    for i in indices_to_show:
                        mesh_path = mesh_path_list[i]
                        mesh = trimesh.load_mesh(mesh_path)

                        c2w = np.eye(4)
                        c2w[:3, :3] = R_c2w[i]
                        c2w[:3, 3] = t_c2w[i]

                        mesh.apply_transform(c2w)
                        mesh_handle = client.scene.add_mesh_trimesh(
                            mesh_path,
                            mesh=mesh
                        )
                        # mesh_handle.remove()

                    print("Visualize SMPL-X")

                vis_pc_from_depth_button = server.gui.add_button(
                    "Visualize Point Cloud from Depth",
                    color="blue",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Visualize Point Cloud from depth",
                )

                @vis_pc_from_depth_button.on_click
                def _(event: viser.GuiEvent) -> None: 
                    client = event.client
                    assert client is not None
                    
                    def depth_to_point_cloud(depth_map, intrinsics):
                        H, W = depth_map.shape
                        i, j = np.meshgrid(np.arange(W), np.arange(H))
                        z = depth_map
                        x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
                        y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
                        points = np.stack((x, y, z), axis=-1)
                        return points.reshape(-1, 3)

                    def apply_c2w(points, R, t):
                        
                        return (R @ points.T).T + t


                    root_path = self.render_tab_state.root_path
                    depth1_dir_path = os.path.join(root_path, 'scene_gs/depths_npy')
                    # depth1_dir_path = os.path.join(self.render_tab_state.root_path, 'scene_gs/depths_npy')
                    depth2_dir_path = os.path.join(root_path, 'depths_vggt_optimized')
                    frame1_path = '/data/ckd248/SCVideo/HSI/fitting/data/Custom/data/bicycle3/scene_gs/test/frames_pure'
                    frame2_path = os.path.join(root_path, 'frames')
                    cam_param_path = os.path.join(root_path, 'cam_params_scene')

                    import cv2
                    N = len(os.listdir(cam_param_path))
                    H, W = cv2.imread(os.path.join(frame2_path, '0.png')).shape[:2]

                    cam_param_list = [os.path.join(cam_param_path, f'{i}.json') for i in range(len(os.listdir(cam_param_path)))]
                    cam_params = []
                    for cam_param_path in cam_param_list:
                        with open(cam_param_path, 'r') as f:
                            cam_params.append(json.load(f))
                    depth1 = np.array([np.load(os.path.join(depth1_dir_path, f'{i}.npy'))[:, :, 0] for i in range(len(os.listdir(depth1_dir_path)))])
                    depth2 = np.array([cv2.resize(np.load(os.path.join(depth2_dir_path, f'{i}.npy')), (W, H)) for i in range(len(os.listdir(depth2_dir_path)))])
                    rgb1 = np.array([cv2.cvtColor(cv2.imread(os.path.join(frame1_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(frame1_path)))])
                    rgb1 = np.array([frame for frame in imageio.get_reader('/data/ckd248/SCVideo/HSI/fitting/data/Custom/data/bicycle_dataset_2d_detailed/scene_render_video.mp4')]) # (N, H, W, 3)
                    rgb2 = np.array([cv2.cvtColor(cv2.imread(os.path.join(frame2_path, f'{i}.png')), cv2.COLOR_BGR2RGB) for i in range(len(os.listdir(frame2_path)))])

                    R_c2w = np.array([cam_param['R'] for cam_param in cam_params]) # c2w
                    t_c2w = np.array([cam_param['t'] for cam_param in cam_params]) # c2w
                    focal = np.array([cam_param['focal'] for cam_param in cam_params])
                    princpt = np.array([cam_param['princpt'] for cam_param in cam_params])

                    self.update_camera(client, R_c2w[0], t_c2w[0], focal[0], princpt[0])

                    # ---------------- 시각화 -----------------------

                    for i in range(len(cam_params)):
                        # if i != 0:#  and i != 80:
                        #     continue
                        fx, fy = cam_params[i]['focal']
                        cx, cy = cam_params[i]['princpt']
                        
                        c2w_R = np.array(cam_params[i]["R"])
                        c2w_t = np.array(cam_params[i]["t"])

                        # c2w_R = np.eye(3)
                        # c2w_t = np.zeros((3))
                        intrinsics = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]
                        ])
                        pc_cam = depth_to_point_cloud(depth1[i], intrinsics)
                        pc_rgb = rgb1[i].reshape(-1, 3)

                        # c2w 변환
                        pc_world = apply_c2w(pc_cam, c2w_R, c2w_t)
                        
                        client.scene.add_point_cloud(
                            name=f"pc_world_{i}",
                            points=pc_world,
                            colors=pc_rgb,
                            point_size=0.001
                        )
                        
                        pc_cam = depth_to_point_cloud(depth2[i], intrinsics)
                        pc_rgb = rgb2[i].reshape(-1, 3)

                        # c2w 변환
                        pc_world = apply_c2w(pc_cam, c2w_R, c2w_t)
                        """
                        client.scene.add_point_cloud(
                            name=f"pc_world2_{i}",
                            points=pc_world,
                            colors=pc_rgb,
                            point_size=0.001
                        )
                        """
                        
                    
                dump_depth_video_button = server.gui.add_button(
                    "Dump Depth Video",
                    color="red",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Dump the trajectory as a depth video.",
                )
                @dump_depth_video_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    self.render_tab_state.render_mode = "depth(precise)"
                    
                    client = event.client
                    assert client is not None

                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, 'cam_params_scene')
                    cam_params_path = [os.path.join(cam_param_dir_path, f'{i}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    scene_info_path = os.path.join(self.render_tab_state.root_path, "scene_gs")
                    os.makedirs(scene_info_path, exist_ok=True)
                    depth_npy_dir = os.path.join(scene_info_path, "depths_npy")
                    os.makedirs(depth_npy_dir, exist_ok=True)
                    depth_dir = os.path.join(scene_info_path, "depths")
                    os.makedirs(depth_dir, exist_ok=True)
                    for i, cam_param_path in enumerate(cam_params_path):
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        R = np.array(cam_param['R']) # c2w
                        t = np.array(cam_param['t']) # c2w
                        focal = np.array(cam_param['focal'])
                        princpt = np.array(cam_param['princpt'])

                        self.update_camera(client, R, t, focal, princpt)

                        camera_state = self.get_camera_state(client)
                        server.scene.set_global_visibility(False)
                        self.render_tab_state.preview_render = True
                        depth = self.render_fn(camera_state, self.render_tab_state)
                        self.render_tab_state.preview_render = False
                        depth_value = depth[:, :, 3:4]
                        depth = depth[:, :, 0:3]
                        
                        depth_path = os.path.join(depth_dir, f'{i}.png')
                        depth = (depth * 255).astype(np.uint8)
                        imageio.imwrite(depth_path, depth)
                        print(f"Save depth render in {depth_path}\nRender resolution is {depth.shape}")

                        depth_npy_path = os.path.join(depth_npy_dir, f'{i}.npy')
                        np.save(depth_npy_path, depth_value)
                        print(f"Save depth in {depth_npy_path}\nDepth resolution is {depth_value.shape}")
                        print(f"Max depth: {depth_value.max()}")
                        print(f"Min depth: {depth_value.min()}")

                    self.render_tab_state.render_mode = "rgb"

                dump_video_wild_button = server.gui.add_button(
                    "Dump Video from Wild Traj",
                    color="green",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Dump the trajectory as a video.",
                )
                @dump_video_wild_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    client = event.client
                    assert client is not None

                    cam_param_dir_path = os.path.join(self.render_tab_state.root_path, 'While the character moves the camera remains static and pushes in')
                    cam_params_path = [os.path.join(cam_param_dir_path, f'{str(i).zfill(6)}.json') for i in range(len(os.listdir(cam_param_dir_path)))]
                    render_dir_path = os.path.join(self.render_tab_state.root_path, "render_et")
                    os.makedirs(render_dir_path, exist_ok=True)
                    render_video_path = os.path.join(self.render_tab_state.root_path, "render_et.mp4")
                    scale_path = os.path.join(self.render_tab_state.root_path, "smplx_optimized", "scale.json")
                    with open(scale_path, 'r') as f:
                        scale = json.load(f)[0]

                    c2w_dir_path = os.path.join(self.render_tab_state.root_path, "cam_params_scene")
                    c2w_path_list = [os.path.join(c2w_dir_path, f'{i}.json') for i in range(len(os.listdir(c2w_dir_path)))]
                    c2w_list = []
                    for c2w_path_list in c2w_path_list:
                        with open(c2w_path_list, 'r') as f:
                            c2w = json.load(f)
                        c2w_list.append(c2w)
                    R_c2w = np.array([c2w['R'] for c2w in c2w_list]) # c2w
                    t_c2w = np.array([c2w['t'] for c2w in c2w_list]) # c2w
                    focal_c2w = np.array([c2w['focal'] for c2w in c2w_list])
                    princpt_c2w = np.array([c2w['princpt'] for c2w in c2w_list])
                    
                    render_frames = []
                    Rs = []
                    ts = []
                    for i, cam_param_path in enumerate(cam_params_path):
                        with open(cam_param_path, 'r') as f:
                            cam_param = json.load(f)
                        R = np.array(cam_param['R']) # c2w
                        t = np.array(cam_param['t']) # c2w

                        R = R_c2w[i] @ R
                        t = R_c2w[i] @ t + t_c2w[i]
                        # t = t * scale

                        Rs.append(R)
                        ts.append(t)
                    
                    from scipy.signal import savgol_filter
                    from scipy.spatial.transform import Rotation

                    def smooth_camera_trajectory_savgol(rot_mats, translations, window=11, poly=3):
                        """
                        rot_mats: (N, 3, 3) numpy array
                        translations: (N, 3) numpy array
                        window: Savitzky–Golay window size (odd)
                        poly: Savitzky–Golay polynomial order
                        """
                        N = len(rot_mats)
                        
                        # --- translation smoothing ---
                        smoothed_trans = np.zeros_like(translations)
                        for i in range(3):  # x, y, z 각각
                            smoothed_trans[:, i] = savgol_filter(translations[:, i], window_length=window, polyorder=poly)
                        
                        # --- rotation smoothing ---
                        quats = Rotation.from_matrix(rot_mats).as_quat()  # (N, 4)
                        smoothed_quats = np.zeros_like(quats)
                        
                        for i in range(4):  # x, y, z, w 각각 component
                            smoothed_quats[:, i] = savgol_filter(quats[:, i], window_length=window, polyorder=poly)
                        
                        # normalize quaternion (중요!)
                        smoothed_quats /= np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
                        
                        smoothed_rotations = Rotation.from_quat(smoothed_quats).as_matrix()  # (N, 3, 3)
                        
                        return smoothed_rotations, smoothed_trans
                    Rs, ts = smooth_camera_trajectory_savgol(np.array(Rs), np.array(ts))

                    for i, (R, t) in enumerate(zip(Rs, ts)):

                        focal = focal_c2w[i]
                        princpt = princpt_c2w[i]

                        self.update_camera(client, R, t, focal, princpt)

                        camera_state = self.get_camera_state(client)
                        server.scene.set_global_visibility(False)
                        self.render_tab_state.preview_render = True
                        render = self.render_fn(camera_state, self.render_tab_state)
                        self.render_tab_state.preview_render = False
                        
                        render_path = os.path.join(render_dir_path, f'{i}.png')
                        render = (render * 255).astype(np.uint8)
                        imageio.imwrite(render_path, render)
                        print(f"Save render in {render_path}\nRender resolution is {render.shape}")

                        render_frames.append(render)

                    imageio.mimsave(render_video_path, render_frames, fps=16)  # fps=16

                    traj = np.array(ts)
                    N = traj.shape[0]
                    colors = np.stack([
                        np.linspace(1.0, 0.0, N),  # R
                        np.zeros(N),              # G
                        np.linspace(0.0, 1.0, N)  # B
                    ], axis=1)  # shape: (N, 3)

                    client.scene.add_point_cloud(
                        name="cam_trajectory_cloud",
                        points=traj,       # (N, 3)
                        colors=colors,     # (N, 3) in [0, 1]
                        point_size=0.01
                    )

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
            }
        )
        super()._populate_rendering_tab()
    def update_camera(self, client, R, t, focal, princpt):
        """
        client: Client object in viser.
        R: c2w Rotation Matrix, shape: (3, 3)
        t: c2w Translation, shape: (3)
        focal: Focal Length, shape: (2) [fx, fy]
        princpt: Principal Point, shape: (2) [cx, cy]
        """

        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R.tolist())
        quat = r.as_quat() # (x, y, z, w)
        quat = [quat[3], quat[0], quat[1], quat[2]] # (w, x, y, z)
        fov_radians = 2 * np.arctan(2 * princpt[1] / (2 * focal[1]))
        client.camera.fov = fov_radians
        client.camera.wxyz = quat
        client.camera.position = t
        self.rerender(_=None)

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count

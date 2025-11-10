import os
from pathlib import Path
import time
import json
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
import yaml
from internal.sc_panel import ScPanel
import trimesh
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.transforms import Transform3d
from scipy.spatial.transform import Rotation
import random


class Viewer:
    def __init__(
            self,
            model_paths: list[str],
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            root_path: str = "Default",
            perturb: bool = False
    ):
        self.device = torch.device("cuda")

        self.model_paths = model_paths
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.extra_video_render_args = []

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.root_path = root_path
        self.perturb = perturb

    def _load_model_from_file(self, load_from: str):

        print("load model from {}".format(load_from))
        self.mesh_vis = trimesh.load_mesh(load_from)

        ext = os.path.splitext(load_from)[-1]
        if ext == '.obj':
            self.mesh_model = load_objs_as_meshes([load_from], load_textures=True, create_texture_atlas=True, texture_wrap='repeat', texture_atlas_size=8, device=self.device)
        elif ext == '.glb':
            mesh_path = load_from.split('.')[0] + '.obj'
            self.mesh_model = load_objs_as_meshes([mesh_path], load_textures=True, create_texture_atlas=True, texture_wrap='repeat', texture_atlas_size=8, device=self.device)
            """
            io = IO()
            io.register_meshes_format(MeshGlbFormat())
            self.mesh_model = io.load_mesh(load_from, include_textures=True, device=self.device)
            """
        
        if self.perturb:
            random_perturbation_path = os.path.join(self.root_path, 'rand_perturb.json')
            if os.path.exists(random_perturbation_path):
                with open(random_perturbation_path, 'r') as f:
                    data = json.load(f)
                    R_perturb = data['R']
                    scale_perturb = data['scale']
            else:
                R_perturb = Rotation.random().as_matrix()
                scale_perturb = random.uniform(0.5, 5)
                data = {
                    "R": R_perturb.tolist(),
                    "scale": scale_perturb
                }
                with open(random_perturbation_path, 'w') as f:
                    json.dump(data, f)

            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] *= scale_perturb
            transform_matrix[:3, :3] = transform_matrix[:3, :3] @ R_perturb
            self.mesh_vis.apply_transform(transform_matrix)
            T = Transform3d(matrix=torch.tensor(transform_matrix.T).to(self.mesh_model.verts_padded()))
            transformed_verts = T.transform_points(self.mesh_model.verts_padded()).to(self.mesh_model.verts_padded())
            
            self.transform_perturb = transform_matrix

            self.mesh_model = self.mesh_model.update_padded(transformed_verts)

        self.scene_mesh_handle = self._server.scene.add_mesh_trimesh(
            "/scene_mesh",
            mesh=self.mesh_vis,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )

    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None, enable_renderer_options: bool = True):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.scene.set_up_direction(self.up_direction)
        self._server = server
        self._load_model_from_file(self.model_paths[0])
        server.gui.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )

        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.gui.add_tab_group()

        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        with tabs.add_tab("SC") as sc_tab:
            self.sc_panel = ScPanel(server, self, sc_tab)

        if block is True:
            while True:
                time.sleep(999)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

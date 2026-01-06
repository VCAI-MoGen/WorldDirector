import os
import json

import numpy as np
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes, IO, save_obj
from pytorch3d.transforms import Transform3d
from scipy.spatial.transform import Rotation
import random
import shutil

data_path = '/home/ckd248/data/Scene2VidBench/Mesh'
target_path = '/home/ckd248/data/Scene2VidBench/Mesh_perturbed'
os.makedirs(target_path, exist_ok=True)
root_path_list = sorted(os.listdir(data_path))

for root_path_name in root_path_list:
    root_path = os.path.join(data_path, root_path_name)
    target_root_path = os.path.join(target_path, root_path_name)
    shutil.copytree(root_path, target_root_path, dirs_exist_ok=True)

    data_json_path = os.path.join(root_path, 'data.json')
    with open(data_json_path, 'r') as f:
        data_json = json.load(f)
    load_from = data_json['scene']
    mesh_vis = trimesh.load_mesh(load_from)

    ext = os.path.splitext(load_from)[-1]
    if ext == '.obj':
        mesh_path = load_from.copy()
        mesh_model = load_objs_as_meshes([mesh_path], load_textures=True, create_texture_atlas=True, texture_wrap='repeat', texture_atlas_size=8, device='cuda:0')
    elif ext == '.glb':
        mesh_path = load_from.split('.')[0] + '.obj'
        mesh_model = load_objs_as_meshes([mesh_path], load_textures=True, create_texture_atlas=True, texture_wrap='repeat', texture_atlas_size=8, device='cuda:0')

    random_perturbation_path = os.path.join(root_path, 'rand_perturb.json')
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
    mesh_vis.apply_transform(transform_matrix)
    T = Transform3d(matrix=torch.tensor(transform_matrix.T).to(mesh_model.verts_padded()))
    transformed_verts = T.transform_points(mesh_model.verts_padded()).to(mesh_model.verts_padded())
    
    transform_perturb = transform_matrix

    mesh_model = mesh_model.update_padded(transformed_verts)

    target_mesh_vis_path = os.path.join(target_root_path, os.path.basename(load_from))
    target_mesh_model_path = os.path.join(target_root_path, os.path.basename(mesh_path))
    
    mesh_vis.export(target_mesh_vis_path)
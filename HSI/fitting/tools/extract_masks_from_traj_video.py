import os
from glob import glob
import argparse
import torch
from tqdm import tqdm
import numpy as np
import json
import imageio
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
target_root_path = root_path + '_detailed'

def mask_to_bbox_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H, W, 1) np.uint8, binary human mask (0 or 255)
    return: (H, W, 1) np.uint8, bounding box mask
    """

    mask = mask.squeeze(-1)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[y_min:y_max+1, x_min:x_max+1] = 255

    return bbox_mask[..., None]

def render_mesh_and_mask(mesh, face, cam_param, bkg, blend_ratio=1.0):
    mesh = torch.FloatTensor(mesh).cuda()[None,:,:]
    face = torch.LongTensor(face).cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}
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
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
    
    # background masking
    is_fg = (fragments.zbuf > 0).float().cpu().numpy()[0]
    fg = images[0,:,:,:3].cpu().numpy() * 255 # (H, W, 3)
    mask = is_fg * 255 # (H, W, 1)
    fg = fg * blend_ratio + bkg / 255 * (1 - blend_ratio)
    render = fg * is_fg + bkg * (1 - is_fg)
    return render.astype(np.uint8), mask.astype(np.uint8)

def load_ply_mesh(ply_path):
    verts, faces = load_ply(ply_path)
    return verts, faces

cam_params_path = os.path.join(root_path, 'cam_params_scene')
target_cam_params_path = os.path.join(target_root_path, 'cam_params_scene')
smplx_ply_path = os.path.join(root_path, 'smplx_optimized', 'meshes_smoothed')
scene_render_video_path = os.path.join(target_root_path, 'scene_render_video.mp4')
scene_render_frames = np.array([frame for frame in imageio.get_reader(scene_render_video_path)]) # (N, H, W, 3)
mask_video_path = os.path.join(target_root_path, 'mask_video.mp4')
human_masked_video_path = os.path.join(target_root_path, 'human_masked_video.mp4')
human_render_video_path = os.path.join(target_root_path, 'human_render_video.mp4')

cam_params_list = glob(os.path.join(cam_params_path, '*.json'))
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in cam_params_list])
mask_frames = []
human_render_frames = []
human_masked_frames = []
for frame_idx in tqdm(frame_idx_list):
    ply_path = os.path.join(smplx_ply_path, f'{frame_idx}.ply')

    # traj video cam_param
    cam_param_path = os.path.join(cam_params_path, f'{frame_idx}.json')
    with open(cam_param_path, 'r') as f:
        cam_param = json.load(f)
    
    # motion video cam_param
    target_cam_param_path = os.path.join(target_cam_params_path, f'{frame_idx}.json')
    with open(target_cam_param_path, 'r') as f:
        target_cam_param = json.load(f)
    
    # Load verts, faces from ply (camera coordinate in traj video)
    verts, faces = load_ply_mesh(ply_path)
    
    R_c2w = torch.tensor(cam_param['R']).to(verts.device) # (3, 3)
    t_c2w = torch.tensor(cam_param['t']).to(verts.device) # (3,)

    R_w2c = torch.tensor(target_cam_param['R']).to(verts.device).T # (3, 3)
    t_w2c = -R_w2c @ torch.tensor(target_cam_param['t']).to(verts.device) # (3,)

    # 1. initial verts are in camera coordinate
    # 2. transform to world coordinate
    # 3. transform to camera coordinate in motion video
    verts = (verts @ R_c2w.T + t_c2w) @ R_w2c.T + t_w2c

    image, mask = render_mesh_and_mask(verts, faces, target_cam_param, bkg=scene_render_frames[frame_idx]) # (H, W, 3), (H, W, 1)
    bb_mask = mask_to_bbox_mask(mask) # (H, W, 1)
    bb_mask = np.repeat(bb_mask, 3, axis=-1)
    human_masked = image.copy()
    human_masked[bb_mask > 0] = 127

    mask_frames.append(bb_mask)
    human_render_frames.append(image)
    human_masked_frames.append(human_masked)
    
imageio.mimsave(mask_video_path, mask_frames, fps=16)
imageio.mimsave(human_render_video_path, human_render_frames, fps=16)
imageio.mimsave(human_masked_video_path, human_masked_frames, fps=16)
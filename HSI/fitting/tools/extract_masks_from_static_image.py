import os
from glob import glob
import argparse
import torch
from tqdm import tqdm
import numpy as np
import json
import imageio
import cv2
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
from pytorch3d.transforms import matrix_to_axis_angle
import sys
sys.path.insert(0, os.path.join('..', 'main'))
sys.path.insert(0, os.path.join('..', 'common'))
from utils.smpl_x import smpl_x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--interpolate', action='store_true', dest='interpolate')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
interpolate_flag = args.interpolate

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
    obj_pos = torch.tensor(obj_pos)
    cam_pos = torch.tensor(cam_pos)

    forward = obj_pos - cam_pos
    forward = forward / torch.norm(forward)
    forward = forward.float()

    up = torch.tensor([0.0, 1.0, 0.0]).float()

    right = torch.cross(up, forward)
    right = right / torch.norm(right)

    true_up = torch.cross(forward, right)

    R = torch.stack([right, true_up, forward], dim=1).float()  # 3x3
    return R

device = "cuda:0"
static_root_path = os.path.join(root_path, 'static')
traj_root_path = os.path.join(root_path, 'traj')
cam_params_path = os.path.join(static_root_path, 'cam_params_scene')
point_path = os.path.join(static_root_path, 'points.json')
with open(point_path, 'r') as f:
    points = json.load(f)
smplx_scale = torch.tensor([1.0])
with open(os.path.join(static_root_path, 'smplx_optimized', 'scale.json'), 'r') as f:
    smplx_scale = torch.tensor(json.load(f))
with open(os.path.join(static_root_path, 'smplx_optimized', 'shape_param.json')) as f:
    shape_param = torch.FloatTensor(json.load(f))

mask_video_path = os.path.join(traj_root_path, 'mask_video.mp4')
human_masked_video_path = os.path.join(traj_root_path, 'human_masked_video.mp4')
human_render_image_path = os.path.join(traj_root_path, 'human_render_image.png')

cam_params_list = glob(os.path.join(cam_params_path, '*.json'))

# traj video cam_param
cam_param_path = os.path.join(cam_params_path, '0.json')
with open(cam_param_path, 'r') as f:
    cam_param = json.load(f)

R_w2c = np.array(cam_param['R']) # (3, 3)
t_w2c = np.array(cam_param['t']) # (3,)

start_point = np.array(points["start_point"])
end_point = np.array(points["end_point"])
start_point_cam = R_w2c @ start_point + t_w2c
end_point_cam = R_w2c @ end_point + t_w2c
print(f"Start point position in camera coordinate system: {start_point_cam}")
print(f"End point position in camera coordinate system: {end_point_cam}")

axis = np.array([1, 0, 0]).astype(np.float32)
angle = np.pi
yz_inv_axis_angle = torch.tensor((axis * angle))[None]

with torch.no_grad():
    output = smpl_x.layer['neutral'](
                            global_orient = torch.stack([yz_inv_axis_angle, yz_inv_axis_angle]), \
                            body_pose = torch.zeros((2, 21, 3)), \
                            jaw_pose = torch.zeros((2, 3)), \
                            leye_pose = torch.zeros((2, 3)), \
                            reye_pose = torch.zeros((2, 3)), \
                            left_hand_pose = torch.zeros((2, 15, 3)), \
                            right_hand_pose = torch.zeros((2, 15, 3)), \
                            expression = torch.zeros((2, 10)), \
                            # transl = torch.FloatTensor(smplx_param['trans']).view(1,-1), \
                            betas = shape_param.view(1, -1), \
                            )
mesh_cam = output.vertices * smplx_scale[None, None, :]
kpt_cam = output.joints[:,smpl_x.kpt['idx'],:] * smplx_scale[None, None, :]
root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]

mesh_cam = mesh_cam - root_cam[:, None, :]
kpt_cam = kpt_cam - root_cam[:, None, :]

R_start = rotation_for_billboarding(start_point_cam, [0.0, 0.0, 0.0])

start_point_cam = torch.tensor(start_point_cam)
verts = mesh_cam[0]
verts = (R_start @ verts.T).T
verts = verts + start_point_cam
verts = verts.float()

img_path = os.path.join(static_root_path, 'scene_renders', '0.png')
img = imageio.imread(img_path)

combined_start_point, mask_start = render_mesh_and_mask(verts, smpl_x.face, cam_param, img)
imageio.imwrite(os.path.join(traj_root_path, 'start_smpl.png'), combined_start_point)

R_end = rotation_for_billboarding(end_point_cam, [0.0, 0.0, 0.0])

end_point_cam = torch.tensor(end_point_cam)
verts = mesh_cam[1]
verts = (R_end @ verts.T).T
verts = verts + end_point_cam
verts = verts.float()

combined_end_point, mask_end = render_mesh_and_mask(verts, smpl_x.face, cam_param, img)
imageio.imwrite(os.path.join(traj_root_path, 'end_smpl.png'), combined_end_point)

num_frames = 81
mask_frames = []
human_masked_frames = []

first_frame = mask_to_bbox_mask(mask_start[:, :,])
last_frame = mask_to_bbox_mask(mask_end)

bbox_start = mask_to_bbox(first_frame[:, :, 0])
bbox_end = mask_to_bbox(last_frame[:, :, 0])

bboxes = interpolate_bbox(bbox_start, bbox_end, num_frames)

interpolated_mask = np.zeros_like(first_frame, dtype=np.uint8)
for b in bboxes:
    m = bbox_to_mask(b, first_frame.shape)
    interpolated_mask = cv2.bitwise_or(interpolated_mask, m)
interpolated_mask = interpolated_mask[:, :, None]

mask_frames.append(first_frame)
for i in range(num_frames - 2):
    if interpolate_flag:
        mask_frames.append(interpolated_mask)
    else:
        whole_mask = (np.ones(first_frame.shape) * 255).astype(np.uint8)
        mask_frames.append(whole_mask)
mask_frames.append(last_frame)

for mask in mask_frames:
    human_masked_frame = img.copy()
    mask = np.repeat(mask, repeats=3, axis=-1)
    human_masked_frame[mask > 0] = 127
    human_masked_frames.append(human_masked_frame)

imageio.mimsave(mask_video_path, mask_frames, fps=16)
imageio.mimsave(human_masked_video_path, human_masked_frames, fps=16)
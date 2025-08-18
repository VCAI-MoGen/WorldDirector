import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import sys
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import predictions_to_glb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--make_depth_video', dest='make_depth_video', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
root_path = args.root_path
make_depth_video = args.make_depth_video

save_path = osp.join(root_path, 'scene_info')
os.makedirs(save_path, exist_ok=True)

cam_path = osp.join(root_path, 'cam_params')
os.makedirs(cam_path, exist_ok=True)

depth_path = osp.join(root_path, 'depths_vggt')
os.makedirs(depth_path, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load and preprocess example images
frame_list = os.listdir(osp.join(root_path, 'frames'))
frame_list = sorted([int(x.split('/')[-1][:-4]) for x in frame_list])
image_names = [osp.join(root_path, 'frames', f'{i}.png') for i in frame_list]
target_h, target_w, _ = cv2.imread(image_names[0]).shape
images = load_and_preprocess_images(image_names).to(device)

if make_depth_video:
    cap = cv2.VideoCapture(os.path.join(root_path, 'video.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)

    masks = np.stack([np.load(os.path.join(root_path, 'depths', f'{frame_idx}_mask.npy')).astype(bool) for frame_idx in frame_list])
    depth = np.stack([cv2.resize(np.load(os.path.join(depth_path, f'{frame_idx}.npy')).astype(np.float32), (target_w, target_h)) for frame_idx in frame_list])
    depth[~masks] = 0
    min_depth = np.min(depth)
    max_depth = np.max(depth)

    # 비디오 작성 준비
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(root_path, 'depths_vggt.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))

    for i, frame_idx in enumerate(frame_list):
        depth_map = ((depth[i, :, :] - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # 컬러맵 적용 (원한다면)
        color_depth = cv2.applyColorMap(depth_map[:, :, None], colormap=cv2.COLORMAP_JET)

        # 영상에 프레임 추가
        out.write(color_depth)

    out.release()
    print(f"Saved depth video to {out_path}")
    sys.exit()

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Save scene info
        for key in predictions.keys():
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            file_name = f'{key}.npy'
            np.save(osp.join(save_path, file_name), predictions[key])

        _, source_h, source_w = predictions['images'][0].shape

        # Save cam_params
        intrinsic = predictions['intrinsic'].astype(np.float32)
        extrinsic = predictions['extrinsic'].astype(np.float32)
        for i in range(intrinsic.shape[0]):
            R = extrinsic[i][:3, :3].tolist()
            t = extrinsic[i][:, 3].tolist()
            focal = [intrinsic[i][0][0] / source_w * target_w, intrinsic[i][1][1] / source_h * target_h]
            princpt = [target_w // 2, target_h // 2]

            with open(osp.join(cam_path, str(frame_list[i]) + '.json'), 'w') as f:
                json.dump({'R': R, 't': t, 'focal': focal, 'princpt': princpt}, f)

        # Save depth
        depth = predictions['depth'].astype(np.float32)
        for i in range(depth.shape[0]):
            np.save(osp.join(depth_path, str(frame_list[i]) + '.npy'), depth[i][:, :, 0])

        # Save GLB scene
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=1e-4,
            prediction_mode="Predicted Pointmap"
        )
        glbscene.export(file_obj=osp.join(save_path, 'scene.glb'))

import argparse
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get arguments
args = parse_args()
root_path = args.root_path

video_path = os.path.join(root_path, 'video.mp4')
keypoint_path = os.path.join(root_path, 'keypoints_whole_body.mp4')
depth_path = os.path.join(root_path, 'depths_vggt.mp4')
smplx_init_path = os.path.join(root_path, 'smplx_init.mp4')
image_optimized_path = os.path.join(root_path, 'smplx_optimized_smoothed.mp4')
scene_optimized_path = os.path.join(root_path, 'smplx_optimized_smoothed_3d.mp4')

import cv2
import numpy as np

video_paths = [
    video_path,
    keypoint_path,
    depth_path,
    smplx_init_path,
    image_optimized_path,
    scene_optimized_path
]

caps = [cv2.VideoCapture(p) for p in video_paths]

frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS)

grid_cols = 3
grid_rows = 2
out_width = frame_width * grid_cols
out_height = frame_height * grid_rows

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(root_path, 'output_grid.mp4')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

while True:
    frames = []
    ret_all = True

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            ret_all = False
            break
        if i < 3:
            frames.append(frame)
        elif i == 3 or i == 4:
            frame = frame[: , frame_width:, :]
            frames.append(frame)
        else:
            frame = cv2.resize(frame, (frame_width, frame_height))
            frames.append(frame)

    if not ret_all:
        break

    row1 = np.hstack(frames[0:3])
    row2 = np.hstack(frames[3:6])
    grid_frame = np.vstack([row1, row2])

    out.write(grid_frame)

for cap in caps:
    cap.release()
out.release()
print(f"Saved in {output_path}")
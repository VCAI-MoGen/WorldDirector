import os
import json
import shutil

dataset_path = '/home/ckd248/data/Scene2VidBench/Mesh'
target_dir = os.listdir(dataset_path)

eval_target_path = './data'

lingo_target_path = os.path.join(eval_target_path, 'lingo')
ours_target_path = os.path.join(eval_target_path, 'ours_mesh')

for target in target_dir:
    meta_data_path = os.path.join(dataset_path, target, 'data.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    text = meta_data['text']

    ours_video_path = os.path.join(dataset_path, target, 'traj', 'video.mp4')
    lingo_video_path = os.path.join(dataset_path, target, 'lingo.mp4')

    src = ours_video_path
    dst = os.path.join(ours_target_path, f'{text}.mp4')
    shutil.copy(src, dst)

    src = lingo_video_path
    dst = os.path.join(lingo_target_path, f'{text}.mp4')
    shutil.copy(src, dst)
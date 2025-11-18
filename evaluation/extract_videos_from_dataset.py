import os
import json
import shutil

target_dir = ['0000', '0005', '0009', '0016', '0017', '0019', '0028', '0032', '0034', '0037', '0042', '0045', '0048', '0049']

dataset_path = '/home/ckd248/data/Scene2VidBench/GS'
trajcrafter_path = '/data1/mks0601/TrajectoryCrafter/experiments'
gen3c_path = '/data1/mks0601/GEN3C/outputs'
eval_target_path = './data'

scene_render_target_path = os.path.join(eval_target_path, 'scene_render')
ours_target_path = os.path.join(eval_target_path, 'ours')
trajcrafter_target_path = os.path.join(eval_target_path, 'trajectorycrafter')
gen3c_target_path = os.path.join(eval_target_path, 'gen3c')


for target in target_dir:
    meta_data_path = os.path.join(dataset_path, target, 'data.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    text = meta_data['text']
        
    scene_render_path = os.path.join(dataset_path, target, 'detailed', 'scene_render_difix_enhanced.mp4')
    ours_video_path = os.path.join(dataset_path, target, 'detailed', 'video.mp4')
    trajcrafter_video_path = os.path.join(trajcrafter_path, target, 'gen.mp4')
    gen3c_video_path = os.path.join(gen3c_path, f'{target}.mp4')

    src = scene_render_path
    dst = os.path.join(scene_render_target_path, f'{text}.mp4')
    shutil.copy(src, dst)

    src = ours_video_path
    dst = os.path.join(ours_target_path, f'{text}.mp4')
    shutil.copy(src, dst)

    src = trajcrafter_video_path
    dst = os.path.join(trajcrafter_target_path, f'{text}.mp4')
    shutil.copy(src, dst)

    src = gen3c_video_path
    dst = os.path.join(gen3c_target_path, f'{text}.mp4')
    shutil.copy(src, dst)


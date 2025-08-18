import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--video_dir_path', type=str, dest='video_dir_path')
    parser.add_argument('--step', type=int, dest='step')
    parser.add_argument('--text_idx', type=str, dest='text_idx')
    parser.add_argument('--mask_threshold', type=float, dest='mask_threshold', default=0.5)
    parser.add_argument('--mask_norm', dest='mask_norm', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
video_dir_path = args.video_dir_path
step = args.step
text_idx = args.text_idx
mask_norm = args.mask_norm
mask_threshold = args.mask_threshold

cmd = f'python vis_attention.py --root_path {root_path} --video_dir_path {video_dir_path} --step {step} --text_idx {text_idx} --mask_threshold {mask_threshold}'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened. terminate the script.')
    sys.exit()

cmd = f'python vis_attention_vace.py --root_path {root_path} --video_dir_path {video_dir_path} --step {step} --text_idx {text_idx} --mask_threshold {mask_threshold}'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened. terminate the script.')
    sys.exit()

if mask_norm:
    cmd = f'python vis_attention_vace_mask_norm.py --root_path {root_path} --video_dir_path {video_dir_path} --step {step} --text_idx {text_idx} --mask_threshold {mask_threshold}'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()
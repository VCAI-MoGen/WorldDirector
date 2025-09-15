import os
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
num_frames = len(os.listdir(osp.join(root_path, 'frames')))

with open(osp.join(root_path, 'frame_list_all.txt'), 'a') as f:
    for i in range(0, num_frames, 1):
        f.writelines(f'{i}\n')
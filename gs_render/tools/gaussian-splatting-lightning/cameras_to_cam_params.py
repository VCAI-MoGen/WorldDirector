import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--cameras_path', type=str, dest='cameras_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
cameras_path = args.cameras_path
cam_params_path = os.path.join(root_path, 'cam_params_scene')

with open(cameras_path, 'r') as f:
    cameras = json.load(f)
for i, camera in enumerate(cameras):
    c2w_t = np.array(camera["position"])
    c2w_R = np.array(camera["rotation"])
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    w2c_R = c2w_R.T
    w2c_t = -w2c_R @ c2w_t
    cam_param = {
        "R": w2c_R.tolist(),
        "t": w2c_t.tolist(),
        "focal": [fx, fy],
        "princpt": [cx, cy]
    }
    with open(os.path.join(cam_params_path, f'{i}.json'), 'w') as f:
        json.dump(cam_param, f)

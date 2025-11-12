import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--target_object', type=str, dest='target_object', default=None)
    parser.add_argument('--vis_all', dest='vis_all', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get arguments
args = parse_args()
root_path = args.root_path

if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]

target_object = args.target_object
vis_all = args.vis_all

cmd = 'python extract_frames.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when extracting frames. terminate the script.')
    sys.exit()

cmd = 'python generate_frame_list_all.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when generating frame list. terminate the script.')
    sys.exit()

# remove unnecessary frames
with open(osp.join(root_path, 'frame_list_all.txt')) as f:
    frame_idx_list = [int(x) for x in f.readlines()]
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
for img_path in img_path_list:
    frame_idx = int(img_path.split('/')[-1][:-4])
    if frame_idx not in frame_idx_list:
        cmd = 'rm ' + img_path
        result = os.system(cmd)
        if (result != 0):
            print('something bad happened when removing unnecessary frames. terminate the script.')
            sys.exit()

# make camera parameters
os.chdir('./vggt')
cmd = 'python run_vggt.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running vggt to get camera parameters. terminate the script.')
    sys.exit()

os.chdir('..')

os.chdir('./DECA')
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running DECA. terminate the script.')
    sys.exit()
os.chdir('..')

# SMPLest-X (get initial SMPLX parameters)
os.chdir('./SMPLest-X')
cmd = "python run_smplestx.py \
    --num_gpus 1 \
    --ckpt_name smplest_x_h \
    --root_path " + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running SMPLest-X. terminate the script.')
    sys.exit()
os.chdir('..')

# sapiens (get 2D whole-body keypoints)
os.chdir('./sapiens/lite/scripts/demo/torchscript')
cmd = 'python run_sapiens_kpt.py --root_path ' + root_path + ' --vis_format video'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens pose. terminate the script.')
    sys.exit()
os.chdir('../../../../../')

# segment-anything (get binary mask using keypoints from sapiens)
os.chdir('./segment-anything')
cmd = 'python run_sam.py --root_path ' + root_path + ' --vis_format video'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running segment-anything. terminate the script.')
    sys.exit()
os.chdir('../')

# fit depth
os.chdir('../main')
cmd = 'python fit_depth.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting. terminate the script.')
    sys.exit()
os.chdir('../tools')

# fit SMPLX
os.chdir('../main')
cmd = 'python fit.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting. terminate the script.')
    sys.exit()
os.chdir('../tools')

# smooth SMPLX
cmd = 'python smooth_smplx_params.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when smoothing smplx parameters. terminate the script.')
    sys.exit()

    
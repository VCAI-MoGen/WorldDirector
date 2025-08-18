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

"""
# make camera parameters for gs scene
if os.path.exists(osp.join(root_path, 'scene_gs')):
    cam_param_dir = osp.join(root_path, 'cam_params_scene')
    os.makedirs(cam_param_dir, exist_ok=True)
    cam_param_path = osp.join(root_path, 'scene_gs', 'cam_param.json')
    image_len = len(img_path_list)
    for i in range(image_len):
        shutil.copy(cam_param_path, osp.join(cam_param_dir, f'{i}.json'))
"""
"""
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
"""
"""
os.chdir('./LHM')
MODEL_NAME='LHM-1B'
MOTION_IMG_DIR=None
VIS_MOTION='true'
MOTION_IMG_NEED_MASK='true'
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=True

cmd = f"python -m LHM.launch infer.human_lrm model_name={MODEL_NAME} \
        image_input={os.path.join(root_path, 'frames', '0.png')} \
        export_video={EXPORT_VIDEO} \
        only_gs={True} \
        root_path={root_path} \
        motion_seqs_dir={os.path.join(root_path, 'smplx_init_lhm')} motion_img_dir={MOTION_IMG_DIR}  \
        vis_motion={VIS_MOTION} motion_img_need_mask={MOTION_IMG_NEED_MASK} \
        render_fps={RENDER_FPS} motion_video_read_fps={MOTION_VIDEO_READ_FPS}"

print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running LHM. terminate the script.')
    sys.exit()
os.chdir('..')
"""
"""
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

# sapiens (get depth using binary masks from Segment Anything)
os.chdir('./sapiens/lite/scripts/demo/torchscript')
cmd = 'python run_sapiens_depth.py --root_path ' + root_path + ' --vis_format video'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens depth. terminate the script.')
    sys.exit()
os.chdir('../../../../../')

# mask objects in video
if target_object is not None:
    os.chdir('./Grounded-SAM-2')
    cmd = 'python run_grounded_sam2.py --root_path ' + root_path + ' --text_prompt ' + target_object
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running grounded-sam-2. terminate the script.')
        sys.exit()
    os.chdir('..')

# reconstruct 3D scene from vggt
os.chdir('./vggt')
cmd = 'python reconstruct_scene.py --root_path ' + root_path + ' --remove_human'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running scene reconstruction. terminate the script.')
    sys.exit()
os.chdir('..')
"""

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

# visualize human-scene aligned motion
cmd = 'python vis_motion.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when visualizing. terminate the script.')
    sys.exit()

"""
# make human trajectory aligned with 3D scene
os.chdir('../main')
cmd = 'python get_human_traj.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting. terminate the script.')
    sys.exit()
os.chdir('../tools')
"""
if vis_all:
    # make vggt depth video
    os.chdir('./vggt')
    cmd = 'python run_vggt.py --root_path ' + root_path + ' --make_depth_video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running vggt to get camera parameters. terminate the script.')
        sys.exit()
    os.chdir('../')

    # make output_grid.mp4
    cmd = 'python vis_all.py --root_path ' + root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when visualizing. terminate the script.')
        sys.exit()
    
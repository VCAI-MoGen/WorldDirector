import os
import os.path as osp
from glob import glob
import sys
import argparse
import cv2
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, dest='video_path')
    parser.add_argument('--remove_human', dest='remove_human', action='store_true')
    args = parser.parse_args()
    assert args.video_path, "Please set video_path."
    return args

def resize_fixed_resolution(img, target_height=480, target_width=720):

    original_height, original_width = img.shape[:2]

    scale = target_height / original_height
    new_width = int(original_width * scale)
    new_height = target_height

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    if new_width < target_width:
        pad_w = target_width - new_width
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = cv2.copyMakeBorder(resized_img, 0, 0, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded
    elif new_width > target_width:
        start_x = (new_width - target_width) // 2
        cropped = resized_img[:, start_x:start_x + target_width]
        return cropped
    else:
        return resized_img

# get arguments
args = parse_args()
video_path = args.video_path
remove_human = args.remove_human
video_name = video_path.split('/')[-1]
root_path = osp.join(os.path.dirname(__file__), '../data/', video_name[:-4])
os.makedirs(root_path, exist_ok=True)

frame_path = osp.join(root_path, 'frames')
os.makedirs(frame_path, exist_ok=True)

scene_path = osp.join(root_path, 'scene')
os.makedirs(scene_path, exist_ok=True)

gs_path = osp.join(root_path, 'gs')
os.makedirs(gs_path, exist_ok=True)

scene_image_path = osp.join(scene_path, 'images')
os.makedirs(scene_image_path, exist_ok=True)

target_height, target_width = (480, 720)

# Extract frames
vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter(f"{osp.join(root_path, 'video.mp4')}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))
frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
success, frame = vidcap.read()
while success:
    print(str(frame_idx) + '/' + str(frame_num), end='\r')
    frame = resize_fixed_resolution(frame, target_height, target_width)
    cv2.imwrite(osp.join(frame_path, str(frame_idx).zfill(4) + '.png'), frame)
    writer.write(frame)
    success, frame = vidcap.read()
    frame_idx += 1
writer.release()
print(f"Finish extracting clips: {frame_num}, {frame_idx}")

if remove_human:
    # Extract human masks using LangSAM
    cmd = 'python extract_masks.py --root_path ' + root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when extracting masks. terminate the script.')
        sys.exit()

    # Video Inpainting
    os.chdir('./STTN')
    cmd = 'python run_STTN.py --root_path ' + root_path + ' --ckpt checkpoints/sttn.pth'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running STTN to inpaint video. terminate the script.')
        sys.exit()
    os.chdir('..')

    frame_path = osp.join(root_path, 'inpainted_frames')

# Move inpainted frames into scene path
stride = max(1, frame_num // 200 + 1)
print(stride)
print(f"Move images \nfrom {frame_path} \nto {scene_image_path}")
filtered_frame_cnt = 0
for i, image_name in tqdm(enumerate(sorted(os.listdir(frame_path)))):
    if i % stride != 0:
        continue
    filtered_frame_cnt += 1
    image_path = osp.join(frame_path, image_name)
    shutil.copy(image_path, osp.join(scene_image_path, image_name))
print(f"Filtered_frame_cnt: {filtered_frame_cnt}")

# Reconstruct 3D point clouds
os.chdir('./vggt')
cmd = 'python demo_colmap.py --scene_dir ' + scene_path + ' --use_ba' # ' --query_frame_num 3'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running vggt. terminate the script.')
    sys.exit()
os.chdir('..')

# 3D GS training
os.chdir('./gsplat')
cmd = f'''bash -c '. ~/data/anaconda3/etc/profile.d/conda.sh && conda activate gsplat && python examples/simple_trainer.py  default --data_factor 1 --data_dir {scene_path} --result_dir {gs_path}' '''
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running vggt. terminate the script.')
    sys.exit()
os.chdir('..')
import argparse
import os
import sys
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--width', type=int, default=720, dest='width')
    parser.add_argument('--height', type=int, default=480, dest='height')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
width = args.width
height = args.height
root_path = args.root_path
static_path = os.path.join(root_path, 'static')
scene_render_image_dir = os.path.join(static_path, 'scene_renders')
scene_render_image_path = os.path.join(scene_render_image_dir, '0.png')
output_path = os.path.join(static_path, 'frames')
os.makedirs(output_path, exist_ok=True)

text_prompt = "Add a standing human figure that looks proportionate to the scene, seamlessly integrated into the existing environment, facing the camera."

os.chdir('./flux')
cmd = f'python -m flux kontext --img_cond_path {scene_render_image_path} --prompt "{text_prompt}"  --num_steps 30 --aspect_ratio "3:2" --guidance 2.5 --seed 1 --output_dir {output_path}'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running flux. terminate the script.')
    sys.exit()

# Rename
zero_path = os.path.join(output_path, "0.png")
if os.path.exists(zero_path):
    os.remove(zero_path)

files = os.listdir(output_path)
img_files = [f for f in files if "img" in f]

old_path = os.path.join(output_path, img_files[0])
new_path = zero_path
os.rename(old_path, new_path)
print(f"Renaming {old_path} to {new_path}")

# Resize
image = cv2.imread(new_path)
resized = cv2.resize(image, (width, height))
cv2.imwrite(new_path, resized)

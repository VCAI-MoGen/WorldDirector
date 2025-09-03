import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
scene_render_image_path = os.path.join(root_path, 'scene_render_image')
start_point_image_path = os.path.join(scene_render_image_path, 'start_point.png')
end_point_image_path = os.path.join(scene_render_image_path, 'end_point.png')
output_path = os.path.join(root_path, 'flux')
os.makedirs(output_path, exist_ok=True)

text_prompt = "Insert a human figure at the red dot, with realistic proportions matching the surroundings"

os.chdir('./flux')
cmd = f'python -m flux kontext --img_cond_path {start_point_image_path} --prompt "{text_prompt}"  --num_steps 30 --aspect_ratio "3:2" --guidance 2.5 --seed 1 --output_dir {output_path}'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running flux. terminate the script.')
    sys.exit()

cmd = f'python -m flux kontext --img_cond_path {end_point_image_path} --prompt "{text_prompt}"  --num_steps 30 --aspect_ratio "3:2" --guidance 2.5 --seed 1 --output_dir {output_path}'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running flux. terminate the script.')
    sys.exit()

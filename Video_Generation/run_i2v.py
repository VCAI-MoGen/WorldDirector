import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, dest='image_path')
    parser.add_argument('--prompt', type=str, dest='prompt')
    parser.add_argument('--model', type=str, dest='model')
    args = parser.parse_args()
    assert args.image_path, "Please set image_path."
    assert args.prompt, "Please set prompt."
    assert args.model, "Please set model."
    return args

# get path
args = parse_args()
image_path = args.image_path
prompt = args.prompt
model = args.model

# Open-Sora I2V
if model == 'Open-Sora':
    os.chdir('./Open-Sora')
    cmd = f"torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/768px.py --cond_type i2v_head --prompt '{prompt}' --ref {image_path}"
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()

    output_path = './samples/video_768px/prompt_0000.mp4'
    result_path = osp.join('../results', f'opensora_{prompt}.mp4')
    shutil.copy(output_path, result_path)
    print(f'Video saved in {result_path}')

# CogVideoX I2V
if model == 'CogVideo':
    os.chdir('./CogVideo/inference')
    cmd = f'python cli_demo.py --prompt "{prompt}" --model_path THUDM/CogVideoX1.5-5b-I2V --generate_type "i2v" --image_or_video_path="{image_path}"'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()

    output_path = './output.mp4'
    result_path = osp.join('../../results', f'cogvideo_{prompt}.mp4')
    shutil.copy(output_path, result_path)
    print(f'Video saved in {result_path}')

# Wan2.1 I2V (prompt extension)
if model == 'Wan2.1':
    os.chdir('./Wan2.1')
    # cmd = f'python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image {image_path} --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "{prompt}" --save_file output.mp4'
    cmd = f'python generate.py --task i2v-14B --size 832*480 --ckpt_dir ./Wan2.1-I2V-14B-480P --image {image_path} --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "{prompt}" --save_file output.mp4'

    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()

    output_path = './output.mp4'
    result_path = osp.join('../results', f'wan_{prompt}.mp4')
    shutil.copy(output_path, result_path)
    print(f'Video saved in {result_path}')

# HunyuanVideo I2V
if model == 'HunyuanVideo-I2V':
    os.chdir('./HunyuanVideo-I2V')
    cmd = f'python3 sample_image2video_modified.py \
    --model HYVideo-T/2 \
    --prompt "{prompt}" \
    --i2v-mode \
    --i2v-image-path {image_path} \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()

    output_path = './results/output.mp4'
    result_path = osp.join('../results', f'hunyuan_{prompt}.mp4')
    shutil.copy(output_path, result_path)
    print(f'Video saved in {result_path}')
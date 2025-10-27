import argparse
from datetime import datetime
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import imageio.v3 as iio
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_video_dir', type=str, dest='generated_video_dir', default='./data/generated_video')
    parser.add_argument('--scene_render_dir', type=str, dest='scene_render_dir', default='./data/scene_render')
    parser.add_argument('--result_dir', type=str, dest='result_dir', default='evaluation_results')
    # parser.add_argument('--text_dir', type=str, dest='text_dir', default='./data/text_prompt')
    parser.add_argument('--name', type=str, dest='name')
    args = parser.parse_args()
    return args

def load_video_as_tensor(video_path: str) -> torch.Tensor:

    video = iio.imread(video_path)  # (N, H, W, C), dtype=uint8
    if video.ndim != 4:
        raise ValueError(f"Expected video of shape (N, H, W, C), got {video.shape}")

    # numpy → torch
    tensor = torch.from_numpy(video)  # (N, H, W, C)

    # (N, H, W, C) → (N, C, H, W)
    tensor = tensor.permute(0, 3, 1, 2).contiguous().float()

    return tensor

device = "cuda:0"
args = parse_args()
generated_video_dir = args.generated_video_dir
scene_render_dir = args.scene_render_dir
name = args.name
result_dir = args.result_dir

if name is not None:
    video_name_list = [name + '.mp4']
else:
    video_name_list = os.listdir(generated_video_dir)

generated_video_path_list = [os.path.join(generated_video_dir, video_name) for video_name in video_name_list]
scene_render_path_list = [os.path.join(scene_render_dir, video_name) for video_name in video_name_list]
texts = [video_name[:-4] for video_name in video_name_list]
N = len(generated_video_path_list)

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

eval_results = {}
psnr_list = []
ssim_list = []
clip_i_temporal_list = []
clip_score_list = []

with torch.no_grad():
    for idx, video_name in tqdm(enumerate(video_name_list), total=len(video_name_list)):
        generated_video_path = generated_video_path_list[idx]
        scene_render_video_path = scene_render_path_list[idx]
        text = texts[idx]

        generated_video = load_video_as_tensor(generated_video_path).to(device) # (N, C, H, W)
        scene_render_video = load_video_as_tensor(scene_render_video_path).to(device) # (N, C, H, W)
        generated_video_norm = generated_video / 255
        scene_render_video_norm = scene_render_video / 255
        B = generated_video.shape[0]

        # Scene Consistency between pred and target
        psnr_value = psnr(generated_video_norm, scene_render_video_norm).item()
        ssim_value = ssim(generated_video_norm, scene_render_video_norm).item()

        # Temporal Consistency between consecutive frames
        clip_i_temporal = clip(generated_video[:-1], generated_video[1:]).item()

        # Text-Motion Alignment
        clip_score = clip(generated_video, [text] * B).item()

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        clip_i_temporal_list.append(clip_i_temporal)
        clip_score_list.append(clip_score)

eval_results['psnr'] = [sum(psnr_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, psnr_list)]]
eval_results['ssim'] = [sum(ssim_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, ssim_list)]]
eval_results['clip_i_temporal'] = [sum(clip_i_temporal_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, clip_i_temporal_list)]]
eval_results['clip_score'] = [sum(clip_score_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, clip_score_list)]]

json_name = f'results_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json'
json_path = os.path.join(result_dir, json_name)
with open(json_path, 'w') as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

# vbench evaluate --dimension "overall_consistency motion_smoothness dynamic_degree imaging_quality aesthetic_quality" --videos_path /home/ckd248/data/SCVideo/evaluation/data/generated_video --mode=custom_input
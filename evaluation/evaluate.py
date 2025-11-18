import argparse
from datetime import datetime
import json
import os
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import imageio.v3 as iio
import torch
import torchvision.transforms as T
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from lang_sam import LangSAM
import imageio

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

def resize_video(video: torch.Tensor, target_t: int, target_h: int, target_w: int):
    """
    video: (T, C, H, W)
    return: (target_t, C, target_h, target_w)
    """
    v = video.unsqueeze(0).permute(0, 2, 1, 3, 4)

    out = F.interpolate(
        v,
        size=(target_t, target_h, target_w),
        mode="trilinear",
        align_corners=False
    )
    
    return out.permute(0, 2, 1, 3, 4)[0].int()

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

generated_video_path_list = sorted([os.path.join(generated_video_dir, video_name) for video_name in video_name_list])
scene_render_path_list = sorted([os.path.join(scene_render_dir, video_name) for video_name in video_name_list])
texts = [video_name[:-4] for video_name in video_name_list]
N = len(generated_video_path_list)

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
langsam = LangSAM(sam_type="sam2.1_hiera_large")
to_pil = T.ToPILImage()

def predict_human_masks(video_tensor, batch_size=41):
    """
    video_tensor    : (N, C, H, W)
    """
    N, C, H, W = video_tensor.shape
    video_pil = [to_pil(frame) for frame in video_tensor]
    
    human_masks = []
    for idx in range(0, len(video_pil), batch_size):
        start_idx = idx
        end_idx = idx + batch_size if idx + batch_size < len(video_pil) else len(video_pil)
        input_image_pil = video_pil[start_idx:end_idx]
        results = langsam.predict(input_image_pil, ["human. man. woman"] * len(input_image_pil))
        
        for result in results:
            mask = torch.zeros((H, W, 1)).bool()
            for pred in result['masks']:
                pred_mask = torch.tensor(pred[:, :, None]).to(mask)
                mask = torch.logical_or(mask, pred_mask)
            human_masks.append(mask.int())
    human_masks = torch.stack(human_masks).permute(0, 3, 1, 2).repeat(1, 3, 1, 1).to(device)
    return human_masks

eval_results = {}
psnr_list = []
ssim_list = []
lpips_list = []
clip_i_temporal_list = []
clip_score_list = []

with torch.no_grad():
    for idx, video_name in tqdm(enumerate(video_name_list), total=len(video_name_list)):
        generated_video_path = generated_video_path_list[idx]
        scene_render_video_path = scene_render_path_list[idx]
        text = texts[idx]

        generated_video = load_video_as_tensor(generated_video_path).to(device) # (N, C, H, W)
        scene_render_video = load_video_as_tensor(scene_render_video_path).to(device) # (N, C, H, W)

        T_g, C_g, H_g, W_g = generated_video.shape
        T_s, C_s, H_s, W_s = scene_render_video.shape
        assert C_g == C_s

        # generated_video = resize_video(generated_video, T_g, H_s, W_s)
        # scene_render_video = resize_video(scene_render_video, T_g, H_s, W_s)

        generated_video_norm = generated_video / 255.0
        scene_render_video_norm = scene_render_video / 255.0
        B = generated_video.shape[0]

        # Scene Consistency between pred and target
        generated_video_human_masks = predict_human_masks(generated_video_norm)
        generated_video_masked = generated_video_norm * (1 - generated_video_human_masks) + generated_video_human_masks
        scene_render_video_masked = scene_render_video_norm * (1 - generated_video_human_masks) + generated_video_human_masks

        """
        import numpy as np
        # float -> uint8 변환
        video_np = (generated_video_masked * 255).byte().cpu().numpy()  # (N, C, H, W)

        # shape 변환: imageio는 (H, W, C)
        video_np = np.transpose(video_np, (0, 2, 3, 1))  # (N, H, W, C)

        # 비디오 저장
        writer = imageio.get_writer(os.path.join('masked_video', f'generated_{text}.mp4'), fps=16)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        # float -> uint8 변환
        video_np = (scene_render_video_masked * 255).byte().cpu().numpy()  # (N, C, H, W)

        # shape 변환: imageio는 (H, W, C)
        video_np = np.transpose(video_np, (0, 2, 3, 1))  # (N, H, W, C)

        # 비디오 저장
        writer = imageio.get_writer(os.path.join('masked_video', f'scene_{text}.mp4'), fps=16)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
        """


        psnr_value = psnr(generated_video_masked, scene_render_video_masked).item()
        ssim_value = ssim(generated_video_masked, scene_render_video_masked).item()
        lpips_value = lpips(generated_video_masked * 2 - 1, scene_render_video_masked * 2 - 1).item()

        # Temporal Consistency between consecutive frames
        clip_i_temporal = clip(generated_video[:-1], generated_video[1:]).item()

        # Text-Motion Alignment
        clip_score = clip(generated_video, [text] * B).item()

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        lpips_list.append(lpips_value)
        clip_i_temporal_list.append(clip_i_temporal)
        clip_score_list.append(clip_score)

eval_results['psnr'] = [sum(psnr_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, psnr_list)]]
eval_results['ssim'] = [sum(ssim_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, ssim_list)]]
eval_results['lpips'] = [sum(lpips_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, lpips_list)]]
eval_results['clip_i_temporal'] = [sum(clip_i_temporal_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, clip_i_temporal_list)]]
eval_results['clip_score'] = [sum(clip_score_list) / N, [{"video_path": video_path, "video_results": video_results} for video_path, video_results in zip(generated_video_path_list, clip_score_list)]]

json_name = f'results_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}_exculding_vbench.json'
json_path = os.path.join(result_dir, json_name)
with open(json_path, 'w') as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

# vbench evaluate --dimension "overall_consistency motion_smoothness dynamic_degree imaging_quality aesthetic_quality" --videos_path /home/ckd248/data/SCVideo/evaluation/data/generated_video --mode=custom_input
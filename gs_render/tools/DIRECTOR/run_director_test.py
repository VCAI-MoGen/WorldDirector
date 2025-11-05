import datetime
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from utils.random_utils import set_random_seed
from visualization.common_viz import get_batch, init, encode_text
import argparse

"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='')
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    return args

args = parse_args
root_path = args.root_path
PROMPT = args.text
"""

root_path = 'vcai_char.npy'

SEED = 33
W_GUIDANCE = 1.4
PROMPT = "While the character moves right, the camera performs a boom bottom."

def get_batch_custom(char_feature, padding_mask, prompt, clip_model, seq_feat, device):
    # generate dummy batch for inference

    # Encode text
    caption_seq, caption_tokens = encode_text([prompt], clip_model, None, device)

    if seq_feat:
        caption_feat = caption_seq[0]
        caption_feat = F.pad(caption_feat, (0, 0, 0, 77 - caption_feat.shape[0]))
        caption_feat = caption_feat.unsqueeze(0).permute(0, 2, 1)
    else:
        caption_feat = caption_tokens
    
    dummy_batch = {}
    dummy_batch['traj_feat'] = torch.zeros((1, 9, 300)) # (1, 9, 300)
    dummy_batch['padding_mask'] = padding_mask[None] # (1, 300)
    dummy_batch['intrinsics'] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    dummy_batch['char_filename'] = ["dummy"]
    dummy_batch['char_feat'] = char_feature[None] # (1, 3, 300)
    dummy_batch['char_padding_mask'] = padding_mask[None] # (1, 300)
    dummy_batch['caption_feat'] = caption_feat
    dummy_batch['caption_raw'] = [prompt]
    dummy_batch['char_raw'] = {}
    dummy_batch['char_raw']['char_centers'] = torch.zeros((1, 3, 300))
    dummy_batch['char_raw']['char_vertices'] = torch.zeros((1, 300, 10475, 3))
    dummy_batch['char_raw']['char_vertices'] = torch.zeros((1, 300, 20908, 3))
    dummy_batch['char_raw']['char_joints'] = torch.zeros((1, 300, 127, 3))

    return dummy_batch

def encode_char_traj(char_dataset, char_traj):
    """
    char_dataset: CharacterDataset
    char_traj: (N, 3) numpy array
    """
    raw_char_feature = torch.from_numpy(char_traj).to(torch.float32)
    padding_size = char_dataset.num_cams - raw_char_feature.shape[0]

    char_feature = raw_char_feature.clone()
    if char_dataset.velocity:
        velocity = char_feature[1:].clone() - char_feature[:-1].clone()
        char_feature = torch.cat([raw_char_feature[0][None], velocity])

    if char_dataset.standardize:
        # Normalize the first frame (orgin) and the rest (velocity) separately
        if len(char_dataset.norm_mean) == 6:
            char_feature[0] -= char_dataset.norm_mean[:3, 0].to(raw_char_feature.device)
            char_feature[0] /= char_dataset.norm_std[:3, 0].to(raw_char_feature.device)
            char_feature[1:] -= char_dataset.norm_mean[3:, 0].to(raw_char_feature.device)
            char_feature[1:] /= char_dataset.norm_std[3:, 0].to(raw_char_feature.device)
        # Normalize all in one
        else:
            char_feature -= char_dataset.norm_mean[:, 0].to(raw_char_feature.device)
            char_feature /= char_dataset.norm_std[:, 0].to(raw_char_feature.device)
    padded_char_feature = F.pad(
        char_feature,
        (0, 0, 0, padding_size),
    )
    padding_mask = torch.ones((char_dataset.num_cams))
    padding_mask[char_feature.shape[0] :] = 0

    if char_dataset.sequential:
        padded_char_feature = padded_char_feature.permute(1, 0)
    else:
        padded_char_feature = padded_char_feature.reshape(-1)

    return padded_char_feature, padding_mask

if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_viz")

    seed = SEED
    prompt = PROMPT
    guidance = W_GUIDANCE

    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance
    char_dataset = dataset.modality_datasets['char']

    seq_feat = diffuser.net.model.clip_sequential
    char_traj = np.load(root_path)
    text = PROMPT
    char_feature, padding_mask = encode_char_traj(char_dataset, char_traj)
    batch = get_batch_custom(char_feature, padding_mask, text, clip_model, seq_feat, device)

    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)
    padding_mask = out["padding_mask"][0].to(bool).cpu()
    valid_len = int(padding_mask.sum().item())
    if valid_len == 0:
        raise RuntimeError("No valid frames found in the generated trajectory.")

    traj_matrices = out["gen_samples"][0].cpu() # (num_cams, 4, 4) (Padded)
    generated_cam_traj = traj_matrices[padding_mask] # (N, 4, 4)
    generated_cam_traj_npy = generated_cam_traj.detach().cpu().numpy()
    target = np.load('vcai_cam.npy')
    print(np.array_equal(generated_cam_traj_npy, target))
    print(target.shape)
    print(generated_cam_traj_npy.shape)
    print(target)
    print(generated_cam_traj_npy)
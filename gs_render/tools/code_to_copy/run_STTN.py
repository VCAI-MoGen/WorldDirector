# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import os.path as osp
import argparse
import copy
import datetime
import random
import sys
import json
from tqdm import tqdm

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor


parser = argparse.ArgumentParser(description="STTN")
parser.add_argument("--root_path", type=str, required=True)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model",   type=str, default='sttn')
args = parser.parse_args()


w, h = 432, 240
ref_length = 10
neighbor_stride = 5
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
        success, image = vidcap.read()
        count += 1
    return frames       

def read_frame_from_dir(frame_dir):
    frames = []
    frame_names = sorted(os.listdir(frame_dir))
    for frame_name in frame_names:
        frame = Image.open(os.path.join(frame_dir, frame_name))
        frames.append(frame.resize((w, h)))
    return frames


def main_worker():
    # set up models 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    root_path = args.root_path

    frame_path = osp.join(root_path, 'frames')
    mask_path = osp.join(root_path, 'masks')
    frame_name_list = sorted(os.listdir(frame_path))

    # prepare datset, encode all frames into deep space

    target_h, target_w, _ = cv2.imread(os.path.join(frame_path, frame_name_list[0])).shape

    frames = read_frame_from_dir(frame_path)
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    

    masks = read_mask(mask_path)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length

    with torch.no_grad():
        feats = model.encoder((feats*(1-masks).float()).view(video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    # print('loading videos and masks from: {}'.format(frame_path))

    # completing holes by spatial-temporal transformers
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
        

    inpainted_frame_path = osp.join(root_path, 'inpainted_frames')
    os.makedirs(inpainted_frame_path, exist_ok=True)

    writer = cv2.VideoWriter(f"{osp.join(root_path, 'inpainting_result.mp4')}", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (target_w, target_h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        image = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (target_w, target_h))
        cv2.imwrite(osp.join(inpainted_frame_path, frame_name_list[f]), image)
        writer.write(image)
    writer.release()
    print('Finish in {}'.format(f"{osp.join(root_path, 'inpainting_result.mp4')}"))

if __name__ == '__main__':
    main_worker()

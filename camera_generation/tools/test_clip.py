import torch
import clip
from PIL import Image
from typing import List
import argparse
from pathlib import Path
from typing import List
import glob
import os
import json
import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType
from clip.simple_tokenizer import SimpleTokenizer
import torch.nn.functional as F

tokenizer = SimpleTokenizer()

def encode_text(
    caption_raws: List[str],  # batch_size
    clip_model: clip.model.CLIP,
    max_token_length: int,
    device: str,
) -> TensorType["batch_size", "context_length"]:
    if max_token_length is not None:
        default_context_length = 77
        context_length = max_token_length + 2  # start_token + 20 + end_token
        assert context_length < default_context_length
        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            caption_raws, context_length=context_length, truncate=True
        )
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        # [bs, context_length] # if n_tokens > 77 -> will truncate
        texts = clip.tokenize(caption_raws, truncate=True)
        print(texts)
        decoded_tokens = [tokenizer.decoder[int(t)] for t in texts[0][texts[0] != 0]]
        print("Decoded tokens:", decoded_tokens)

    # [batch_size, n_ctx, d_model]
    x = clip_model.token_embedding(texts.to(device)).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest in each sequence)
    x_tokens = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)].float()
    x_seq = [x[k, : (m + 1)].float() for k, m in enumerate(texts.argmax(dim=-1))]

    return x_seq, x_tokens, decoded_tokens

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = '/data1/cympyc1785/SceneData/DL3DV/scenes/1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed/images/frame_00001.png'
image_path2 = '/data1/cympyc1785/SceneData/DL3DV/scenes/1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed/images/frame_00002.png'
text_prompt = "The camera pans left and moves forward, capturing a wide view of the basketball court and surrounding trees."
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
x_seq, x_tokens, decoded_tokens = encode_text([text_prompt], model, max_token_length=None, device=device)
print(len(x_seq))
print(x_seq[0].shape)
print(x_tokens.shape)

with torch.no_grad():
    image_features = model.encode_image(image)
    image_text_sim = F.cosine_similarity(x_tokens, image_features, dim=1)
    print(image_text_sim)
    for idx, tok in enumerate(x_seq[0]):
        image_tok_sim = F.cosine_similarity(x_seq[0][idx:idx+1].norm(dim=-1, keepdim=True), image_features.norm(dim=-1, keepdim=True), dim=-1)
        print(decoded_tokens[idx], image_tok_sim)
    
    text = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image)
    print(logits_per_text)

    decoded_text = [t[:-4] for t in decoded_tokens if '</w>' in t]
    text = clip.tokenize(decoded_text).to(device)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    for idx, data in enumerate(logits_per_image[0]):
        print(decoded_text[idx], data)

    image_feature1 = model.encode_image(image)
    image_feature2 = model.encode_image(image2)
    f1 = F.normalize(image_feature1, dim=1)
    f2 = F.normalize(image_feature2, dim=1)

    cos_sim = (f1 * f2).sum(dim=1)   # 또는 torch.matmul(f1, f2.T)

    print("Cosine similarity:", cos_sim.item())
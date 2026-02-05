import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half) / half
    ).to(t)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def positional_encoding(seq_len, dim, device='cpu'):
    pos = torch.arange(seq_len, device=device).float()[:, None]  # (N, 1)
    i = torch.arange(dim, device=device).float()[None, :]  # (1, D)
    angle_rates = 1 / (10000 ** (2 * (i//2) / dim))
    angle_rads = pos * angle_rates
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe  # (N, D)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        t = timestep_embedding(t, self.dim)
        x = self.lin1(t)
        x = F.silu(x)
        x = self.lin2(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weight = None

    def forward(self, x, context, key_padding_mask=None):
        a, attn_weight = self.attn(x, context, context, key_padding_mask=key_padding_mask)
        if self.training is False:
            self.attn_weight = attn_weight
        return self.norm(x + a)


class CameraDiffusionModel(nn.Module):
    def __init__(
        self,
        cam_dim=32,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        point_dim=768,
        text_dim=4096,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.cam_in = nn.Linear(cam_dim, hidden_dim)

        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(point_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    batch_first=True
                ),
                CrossAttention(hidden_dim, num_heads),
            ])
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, cam_dim)
        self.cross_attn_weight = None

    def forward(self, x_t, t, text_emb, text_mask, point_emb=None, point_mask=None):
        B, T, _ = x_t.shape

        h = self.cam_in(x_t)

        pos_emb = positional_encoding(T, h.shape[-1], device=h.device).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb

        t_embed = self.time_mlp(t)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed.unsqueeze(1).repeat(1, T, 1)

        h = h + t_embed
        
        text_tok = self.text_proj(text_emb)
        if point_emb is not None and point_mask is not None:
            point_tok = self.point_proj(point_emb)
            context = torch.cat([text_tok, point_tok], dim=1)
            mask = torch.cat([text_mask, point_mask], dim=1)
        else:
            context = text_tok
            mask = text_mask
        context_pos_emb = positional_encoding(context.shape[-2], context.shape[-1], device=context.device).unsqueeze(0).expand(B, -1, -1)
        context += context_pos_emb
        
        for self_attn, cross_attn in self.layers:
            h = self_attn(h)
            h = cross_attn(h, context, key_padding_mask=~mask) + h
            if self.training is False:
                if self.cross_attn_weight is None:
                    self.cross_attn_weight = cross_attn.attn_weight
                else:
                    self.cross_attn_weight += cross_attn.attn_weight
        if self.training is False:
            self.cross_attn_weight /= self.num_layers

        return self.out(h)   # (B, T, 9)
    
def get_model():
    model = CameraDiffusionModel()
    return model
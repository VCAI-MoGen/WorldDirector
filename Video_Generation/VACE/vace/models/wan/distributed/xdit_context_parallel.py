# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape # (seq_len, 1, 64)
    pad_size = target_len - seq_len # (0)
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device) # (0, 1, 64)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0) # (seq_len, 1, 64)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, mul=1):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        f = f * mul
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2)) # (seq_len / world_size, 40, 64)
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1) # (seq_len, 1, 64)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size) # (seq_len, 1, 64)
        s_per_rank = s
        if mul == 1:
            freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                        s_per_rank), :, :]
        elif mul == 2:
            s_per_rank_mul = s_per_rank // mul
            freqs_i_rank_a = freqs_i[(sp_rank * s_per_rank_mul):((sp_rank + 1) *
                                                       s_per_rank_mul), :, :]
            freqs_i_rank_b = freqs_i[(((sp_rank + mul) * s_per_rank_mul)):((sp_rank + mul + 1) *
                                                       s_per_rank_mul), :, :]
            freqs_i_rank = torch.cat([freqs_i_rank_a, freqs_i_rank_b], dim=0)  
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2) # (seq_len / world_size, 40, 128)
        x_i = torch.cat([x_i, x[i, s:]]) # (seq_len / world_size, 40, 128)

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def usp_dit_forward_vace(
    self,
    x,
    vace_context,
    seq_len,
    kwargs
):
    # embeddings
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in c
    ])

    # arguments
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)

    # Context Parallel
    c = torch.chunk(
        c, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block in self.vace_blocks:
        c = block(c, **new_kwargs)
    hints = torch.unbind(c)[:-1]
    return hints


def usp_dit_forward(
    self,
    x,
    t,
    vace_context,
    context,
    seq_len,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
    scene_context=None
):
    """
    x:              A list of videos each with shape [C, T, H, W]. ex. (16, 22, 60, 90)
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # if y is not None:
    #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # list of (1, 5120, 22, 30, 45)
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x] # list of (1, 29700, 5120)
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ]) # (B, 29700, 5120)

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # if clip_fea is not None:
    #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
    #     context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]
    
    kwargs['enrich'] = False

    hints = self.forward_vace(x, vace_context, seq_len, kwargs)

    kwargs['enrich'] = False

    kwargs['hints'] = hints
    kwargs['context_scale'] = vace_context_scale
    # if t[0] < 799:
    #     kwargs['enrich'] = True
    # else:
    #     kwargs['enrich'] = False
    # kwargs['enrich'] = True

    for idx, block in enumerate(self.blocks):
        # if t[0] in [999, 989, 245]:
        #     kwargs['idx'] = f"{t[0]}_{get_sequence_parallel_rank()}_{idx}"
        x = block(x, **kwargs)
        if 'idx' in kwargs:
            del kwargs['idx']

    del kwargs['enrich']

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)
    
    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     enrich=False,
                     idx=None,
                     dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    if enrich:

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        q_recon = q[2:]
        k_recon = k[2:]
        v_recon = v[2:]
        
        k_enrich_pos = torch.cat([k[0:1, ...], k_recon], dim=1)
        v_enrich_pos = torch.cat([v[0:1, ...], v_recon], dim=1)
        k_enrich_neg = torch.cat([k[1:2, ...], k_recon], dim=1)
        v_enrich_neg = torch.cat([v[1:2, ...], v_recon], dim=1)
        
        q_enrich = q[:2]
        k_enrich = torch.cat([k_enrich_pos, k_enrich_neg], dim=0)
        v_enrich = torch.cat([v_enrich_pos, v_enrich_neg], dim=0)

        # Hard Replacement
        k_enrich = torch.cat([k_recon, k_recon], dim=0)
        v_enrich = torch.cat([v_recon, v_recon], dim=0)

        # k_recon = rope_apply(k_recon, grid_sizes[2:], freqs)
        # k_enrich = rope_apply(k_enrich, grid_sizes[:2], freqs)

        
        # ROPE manipulation
        """
        q_enrich = rope_apply(torch.cat([q[:2], q[:2]], dim=1), grid_sizes[:2], freqs, mul=2)[:, :s, ...]
        k_enrich = rope_apply(k_enrich, grid_sizes[:2], freqs, mul=2)

        q_recon = rope_apply(q[2:], grid_sizes[2:], freqs)
        k_recon = rope_apply(k_recon, grid_sizes[2:], freqs)
        """
        

        # ROPE manipulation (Query)
        """
        q_enrich = rope_apply(torch.cat([q[:2], torch.cat([q[2:], q[2:]], dim=0)], dim=1), grid_sizes[:2], freqs, mul=2)[:, :s, ...]
        k_enrich = rope_apply(k_enrich, grid_sizes[:2], freqs, mul=2)

        q_recon = rope_apply(q[2:], grid_sizes[2:], freqs)
        k_recon = rope_apply(k_recon, grid_sizes[2:], freqs)
        """
        
        x_enrich = xFuserLongContextAttention()(
            None,
            query=half(q_enrich),
            key=half(k_enrich),
            value=half(v_enrich),
            window_size=self.window_size)
        x_recon = xFuserLongContextAttention()(
            None,
            query=half(q_recon),
            key=half(k_recon),
            value=half(v_recon),
            window_size=self.window_size)
        
        x = torch.cat([x_enrich, x_recon], dim=0)

        if idx is not None:
            torch.save(half(q_enrich).detach().cpu(), f'/home/ckd248/data/Attention_hard_replacement/{idx}_query_enrich.pt')
            torch.save(half(k_enrich).detach().cpu(), f'/home/ckd248/data/Attention_hard_replacement/{idx}_key_enrich.pt')
            # torch.save(half(v_enrich).detach().cpu(), f'/home/ckd248/data/Attention_rope/{idx}_value_enrich.pt')
            torch.save(half(q_recon).detach().cpu(), f'/home/ckd248/data/Attention_hard_replacement/{idx}_query_recon.pt')
            torch.save(half(k_recon).detach().cpu(), f'/home/ckd248/data/Attention_hard_replacement/{idx}_key_recon.pt')
            # torch.save(half(v_recon).detach().cpu(), f'/home/ckd248/data/Attention_rope/{idx}_value_recon.pt')
            # torch.save(x.detach().cpu(), f'/home/ckd248/data/Attention_rope/{idx}_attention.pt')
    else:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # TODO: We should use unpaded q,k,v for attention.
        # k_lens = seq_lens // get_sequence_parallel_world_size()
        # if k_lens is not None:
        #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
        #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
        #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

        x = xFuserLongContextAttention()(
            None,
            query=half(q),
            key=half(k),
            value=half(v),
            window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
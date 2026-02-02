import torch

def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).int()

def pad_and_mask(xs):
    """
    xs: list of tensors, each shape (Ni, D)
    returns:
        padded: (B, N_max, D)
        attn_mask: (B, N_max)  # True = valid, False = padding
    """
    B = len(xs)
    D = xs[0].shape[1]
    device = xs[0].device
    dtype = xs[0].dtype

    lengths = torch.tensor([x.shape[0] for x in xs], device=device)
    N_max = lengths.max().item()

    padded = torch.zeros((B, N_max, D), device=device, dtype=dtype)
    attn_mask = torch.zeros((B, N_max), device=device, dtype=torch.bool)

    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        attn_mask[i, :n] = True

    return padded, attn_mask
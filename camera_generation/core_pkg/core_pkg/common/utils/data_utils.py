import numpy as np
import torch

SCENE_CUTTING_THRESHOLD = 90
def normalize_camera_extrinsics_and_points(extrinsics, points, scene_cutting_threshold=SCENE_CUTTING_THRESHOLD):
    """
    extrinsics: (N, 4, 4) (w2c)
    points: (M, 3)

    normalized_extrinsics: (N, 4, 4) (w2c)
    normalized_points: (M, 3)
    """
    e0_inv = np.linalg.inv(extrinsics[0])

    # normlization by first camera
    normalized_extrinsics = e0_inv @ extrinsics 
    new_points = points @ e0_inv[:3, :3].T + e0_inv[:3, 3]

    # scale normalization
    avg_scale = np.array([new_points.norm(dim=-1).mean()])
    normalized_points = new_points / avg_scale 
    normalized_extrinsics[:, :3, 3] = normalized_extrinsics[:, :3, 3] / avg_scale

    # scene cutting
    d = np.linalg.norm(normalized_points, axis=1)
    r = np.percentile(d, scene_cutting_threshold)
    mask = d <= r

    return normalized_extrinsics, normalized_points, avg_scale, mask

def normalize_camera_extrinsics_and_points_tensor(extrinsics, points, scene_cutting_threshold=SCENE_CUTTING_THRESHOLD):
    """
    extrinsics: (N, 4, 4) torch.Tensor (w2c)
    points: (M, 3) torch.Tensor

    normalized_extrinsics: (N, 4, 4)
    normalized_points: (M, 3)
    avg_scale: (1,)
    """
    orig_dtype = extrinsics.dtype
    points = points.to(dtype=extrinsics.dtype, device=extrinsics.device)

    # inverse in float32 for stability (important for bf16)
    e0_inv = torch.linalg.inv(extrinsics[0].float()).to(orig_dtype)

    # normalize extrinsics
    normalized_extrinsics = e0_inv @ extrinsics

    # normalize points
    R = e0_inv[:3, :3]
    t = e0_inv[:3, 3]

    new_points = points @ R.T + t

    # scale: (1,)
    avg_scale = new_points.norm(dim=-1).mean().unsqueeze(0)

    normalized_points = new_points / avg_scale
    normalized_extrinsics[:, :3, 3] /= avg_scale

    # scene cutting
    d = torch.norm(normalized_points, dim=1)
    r = torch.quantile(d, scene_cutting_threshold / 100)
    mask = d <= r

    return normalized_extrinsics, normalized_points, avg_scale, mask

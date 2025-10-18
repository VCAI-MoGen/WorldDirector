import datetime
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils.random_utils import set_random_seed
from visualization.common_viz import get_batch, init
from viser_viz import TrajectoryData, launch_viewer


SEED = 33
W_GUIDANCE = 1.4
SAMPLE_ID = "traj_world"
PROMPT = "While the character moves right, the camera performs a boom bottom."

VISER_HOST_DEFAULT = os.getenv("VISER_HOST", "0.0.0.0")
VISER_PORT_DEFAULT = int(os.getenv("VISER_PORT", "4242"))
CAM_OUTPUT_ROOT = Path("./outputs/camera_params")


def denormalize_camera(
    matrices_norm: torch.Tensor, R0_inv: torch.Tensor, t0: torch.Tensor
) -> torch.Tensor:
    R0 = R0_inv.transpose(-1, -2)
    R_norm = matrices_norm[:, :3, :3]
    t_norm = matrices_norm[:, :3, 3]

    R_world = torch.matmul(R0.unsqueeze(0), R_norm)
    t_world = torch.matmul(t_norm, R0_inv) + t0

    eye = torch.eye(4, dtype=matrices_norm.dtype, device=matrices_norm.device)
    matrices_world = eye.unsqueeze(0).repeat(matrices_norm.shape[0], 1, 1)
    matrices_world[:, :3, :3] = R_world
    matrices_world[:, :3, 3] = t_world

    return matrices_world


def export_camera_params(
    output_root: Path, sample_id: str, matrices_world: np.ndarray
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = output_root / f"{sample_id}_{timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)

    matrices_world = np.asarray(matrices_world, dtype=np.float32)
    save_path = target_dir / "camera_params.npy"
    np.save(save_path, matrices_world)

    return save_path


if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_viz")

    # ------------------------------------------------------------------------- #
    # Interactive arguments
    seed = int(input(f"Seed (default={SEED}): ") or SEED)
    guidance = float(input(f"Guidance (default={W_GUIDANCE}): ") or W_GUIDANCE)
    sample_id = input(f"Sample ID (default={SAMPLE_ID}): ") or SAMPLE_ID
    prompt = input(f"Prompt (default='{PROMPT}'): ") or PROMPT
    # ------------------------------------------------------------------------- #

    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance

    seq_feat = diffuser.net.model.clip_sequential
    batch = get_batch(prompt, sample_id, clip_model, dataset, seq_feat, device)
    for k, v in batch.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"{kk}: {vv.shape if isinstance(vv, torch.Tensor) else type(vv)}")
                

        else:
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")
            if k == "padding_mask":
                    print(v)
                    print(batch["traj_feat"].shape[1])
    print("\n\n")

    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)

    padding_mask = out["padding_mask"][0].to(bool).cpu()
    valid_len = int(padding_mask.sum().item())
    if valid_len == 0:
        raise RuntimeError("No valid frames found in the generated trajectory.")

    traj_matrices = out["gen_samples"][0].cpu()
    traj_norm_all = traj_matrices[padding_mask]

    align_first_frame = getattr(dataset.trajectory_dataset, "align_first_frame", False)
    canonical_available = False

    traj_world_all = traj_norm_all.clone()
    R0_np = np.eye(3, dtype=np.float32)
    t0_np = np.zeros(3, dtype=np.float32)
    traj_norm_all_aligned = None
    cam_pos_norm_all = None

    R0_inv = None
    t0 = None
    if align_first_frame:
        R0_inv_src = out.get("R0_inv")
        t0_src = out.get("t0")
        if R0_inv_src is not None and t0_src is not None:
            if isinstance(R0_inv_src, torch.Tensor):
                R0_inv = R0_inv_src[0].detach().cpu()
            else:
                R0_inv = torch.tensor(R0_inv_src[0], dtype=torch.float32)
            if R0_inv.ndim == 3 and R0_inv.shape[0] == 1:
                R0_inv = R0_inv.squeeze(0)

            if isinstance(t0_src, torch.Tensor):
                t0 = t0_src[0].detach().cpu()
            else:
                t0 = torch.tensor(t0_src[0], dtype=torch.float32)
            if t0.ndim == 2 and t0.shape[0] == 1:
                t0 = t0.squeeze(0)

            traj_world_all = denormalize_camera(traj_norm_all, R0_inv, t0)
            traj_norm_all_aligned = traj_norm_all
            cam_pos_norm_all = traj_norm_all[:, :3, 3]

            R0 = R0_inv.transpose(-1, -2)
            R0_np = R0.numpy().astype(np.float32)
            t0_np = t0.numpy().astype(np.float32)
            canonical_available = True

    t_world_all = traj_world_all[:, :3, 3]

    char_raw_block: Dict[str, torch.Tensor] | None = batch.get("char_raw")
    if not isinstance(char_raw_block, dict):
        raise KeyError("Character raw data is missing from the batch.")

    char_vertices_tensor = char_raw_block.get("char_vertices")
    char_faces_tensor = char_raw_block.get("char_faces")
    char_centers_tensor = char_raw_block.get("char_centers")
    char_joints_tensor = char_raw_block.get("char_joints")


    def _strip_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[0] if tensor.ndim > 0 and tensor.shape[0] == 1 else tensor

    char_centers_tensor = _strip_batch_dim(char_centers_tensor)
    char_vertices_tensor = _strip_batch_dim(char_vertices_tensor)
    char_faces_tensor = _strip_batch_dim(char_faces_tensor)
    char_joints_tensor = _strip_batch_dim(char_joints_tensor)

    char_centers = (
        char_centers_tensor[:, :valid_len]
        .transpose(0, 1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    verts_world = (
        char_vertices_tensor[:valid_len]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    joints_world = (
        char_joints_tensor[:valid_len]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if char_faces_tensor.ndim == 3:
        faces_tensor = char_faces_tensor[0]
    else:
        faces_tensor = char_faces_tensor
    faces_ref = faces_tensor.detach().cpu().numpy().astype(np.int32)
    traj_world = traj_world_all[:valid_len]
    camera_positions_world = t_world_all[:valid_len]

    verts_norm = None
    joints_norm = None
    trajectory_norm = None
    traj_norm_np = None
    cam_pos_norm_np = None

    if canonical_available:
        traj_norm = traj_norm_all_aligned[:valid_len]
        camera_positions_norm = cam_pos_norm_all[:valid_len]
        verts_norm = np.matmul(verts_world - t0_np, R0_np)
        joints_norm = np.matmul(joints_world - t0_np, R0_np)
        trajectory_norm = np.matmul(char_centers - t0_np, R0_np)
        traj_norm_np = traj_norm.numpy().astype(np.float32)
        cam_pos_norm_np = camera_positions_norm.numpy().astype(np.float32)

    traj_world_np = traj_world.numpy().astype(np.float32)
    cam_pos_world_np = camera_positions_world.numpy().astype(np.float32)

    char_filename_entry = batch["char_filename"]
    if isinstance(char_filename_entry, (list, tuple)):
        sample_basename = Path(char_filename_entry[0]).stem
    else:
        sample_basename = Path(char_filename_entry).stem

    viewer_payload = TrajectoryData(
        sample_id=sample_basename,
        vertices_raw=verts_world,
        trajectory_raw=char_centers,
        faces=faces_ref,
        root_position=t0_np,
        vertices_norm=verts_norm,
        joints_norm=joints_norm,
        trajectory_norm=trajectory_norm,
        camera_poses_raw=traj_world_np,
        camera_poses_norm=traj_norm_np,
        camera_positions_raw=cam_pos_world_np,
        camera_positions_norm=cam_pos_norm_np,
        is_canonical=canonical_available,
    )
    export_camera_params(CAM_OUTPUT_ROOT,sample_id,traj_world_np)
    print(f"Generated {valid_len} frames for sample '{viewer_payload.sample_id}'.")

    launch_answer = input("Launch viser viewer? [Y/n]: ").strip() or "Y"
    if launch_answer.lower().startswith("y"):
        host_input = input(
            f"Viser host (default={VISER_HOST_DEFAULT}): "
        ).strip()
        port_input = input(
            f"Viser port (default={VISER_PORT_DEFAULT}): "
        ).strip()

        host = host_input or VISER_HOST_DEFAULT
        try:
            port = int(port_input) if port_input else VISER_PORT_DEFAULT
        except ValueError:
            print(
                f"Invalid port '{port_input}'. Falling back to {VISER_PORT_DEFAULT}."
            )
            port = VISER_PORT_DEFAULT

        launch_viewer(
            [viewer_payload],
            host=host,
            port=port,
        )

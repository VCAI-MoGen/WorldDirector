import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import rerun as rr
import torch
import datetime
from utils.random_utils import set_random_seed
from utils.rerun import log_sample
from visualization.common_viz import get_batch, init

# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 1.4
SAMPLE_ID = "2011_KAeAqaA0Llg_00005_00001"
PROMPT = "While the character moves right, the camera performs a boom bottom."

RENDER_SCALE = 4

# ------------------------------------------------------------------------------------- #


def get_normals(vertices: torch.Tensor, faces: torch.Tensor):
    num_frames = vertices.shape[0]

    # Case 1: Static faces (F, 3)
    if faces.ndim == 2:
        faces = faces.unsqueeze(0).expand(num_frames, -1, -1).contiguous()

    # Case 2: Dynamic faces (T, F, 3) but not yet masked
    elif faces.shape[0] != num_frames:
        faces = faces[:num_frames,:,:]

    verts_rgb = torch.ones_like(vertices)
    verts_rgb[:, :, 1] = 0  # Green
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    normals = meshes.verts_normals_padded()

    return normals, meshes

# ------------------------------------------------------------------------------------- #
def denormalize_camera(
    matrices_norm: torch.Tensor, R0_inv: torch.Tensor, t0: torch.Tensor
) -> torch.Tensor:
    R0 = R0_inv.transpose(-1,-2)
    R_norm = matrices_norm[:, :3, :3]
    t_norm = matrices_norm[:, :3, 3]

    R_world = torch.matmul(R0.unsqueeze(0), R_norm)
    t_world = torch.matmul(t_norm, R0_inv) + t0

    eye = torch.eye(4, dtype=matrices_norm.dtype, device=matrices_norm.device)
    matrices_world = eye.unsqueeze(0).repeat(matrices_norm.shape[0], 1, 1)
    matrices_world[:, :3, :3] = R_world
    matrices_world[:, :3, 3] = t_world

    return matrices_world

if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_viz")

    while True:
        # ----------------------------------------------------------------------------- #
        # Arguments
        SEED = int(input(f"Seed (default={SEED}): ") or SEED)
        W_GUIDANCE = float(input(f"Guidance (default={W_GUIDANCE}): ") or W_GUIDANCE)
        SAMPLE_ID = input(f"Sample ID (default={SAMPLE_ID}): ") or SAMPLE_ID
        PROMPT = input(f"Prompt (default='{PROMPT}'): ") or PROMPT
        # ----------------------------------------------------------------------------- #

        # Set arguments
        set_random_seed(SEED)
        diffuser.gen_seeds = np.array([SEED])
        diffuser.guidance_weight = W_GUIDANCE

        # Inference
        seq_feat = diffuser.net.model.clip_sequential
        batch = get_batch(PROMPT, SAMPLE_ID, clip_model, dataset, seq_feat, device)
        with torch.no_grad():
            out = diffuser.predict_step(batch, 0)

        # Run visualization
        padding_mask = out["padding_mask"][0].to(bool).cpu()
        padded_traj = out["gen_samples"][0].cpu()
        traj = padded_traj[padding_mask]
        padded_vertices = out["char_raw"]["char_vertices"][0]
        vertices = padded_vertices[padding_mask]
        faces = out["char_raw"]["char_faces"][0]
        normals, meshes = get_normals(vertices, faces)
        fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        caption = out["caption_raw"][0]
        if "R0_inv" in out:
            R0_inv = out["R0_inv"][0].cpu()
            if R0_inv.ndim == 3:
                R0_inv = R0_inv.squeeze(0)
            t0 = out["t0"][0].cpu()
            if t0.ndim == 2:
                t0 = t0.squeeze(0)

            traj = denormalize_camera(traj, R0_inv, t0)
        accel = "cuda" if device.type == "cuda" else "cpu"

        rr.init(f"{SAMPLE_ID}", spawn=False)
        log_sample(
            root_name="world",
            traj=traj.detach().numpy(),
            K=K,
            vertices=vertices.detach().numpy(),
            faces=faces.detach().numpy(),
            normals=normals.detach().numpy(),
            caption=caption,
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rr.save(f"{SAMPLE_ID}_{timestamp}.rrd")
        print(f"saved rerun log to {SAMPLE_ID}_{timestamp}.rrd")

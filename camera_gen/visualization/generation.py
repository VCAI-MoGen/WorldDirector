import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import rerun as rr
import torch
import torch.nn.functional as F
import os
from utils.random_utils import set_random_seed
from utils.rerun import log_sample
from visualization.common_viz import get_batch, init
import datetime
from scipy.ndimage import gaussian_filter1d
import json
# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 2.0
SAMPLE_ID = "traj"
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
        raise ValueError(f"Mismatch between vertices ({num_frames}) and faces ({faces.shape[0]})")

    verts_rgb = torch.ones_like(vertices)
    verts_rgb[:, :, 1] = 0  # Green
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    normals = meshes.verts_normals_padded()

    return normals, meshes


# ------------------------------------------------------------------------------------- #

def gen_K_seq_from_caption(
    caption: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T: int,
    smoothing: float = 1.0,
    zoom_in_factor: float = 3.0,
    zoom_out_factor: float = 0.3,
    
) -> np.ndarray:
    caption_l = caption.lower()

    if "zoom" in caption_l and "in" in caption_l:
        zoom_weights = np.linspace(1.0, zoom_in_factor, T)
    elif "zoom" in caption_l or "out" in caption_l:
        zoom_weights = np.linspace(zoom_out_factor, 1.0, T)
    else:
        zoom_weights = np.ones(T)
    zoom_weights_smoothed = gaussian_filter1d(np.array(zoom_weights), sigma = smoothing)
    fx_seq = fx * zoom_weights_smoothed
    fy_seq = fy * zoom_weights_smoothed

    K_seq = np.stack([
        np.array([
            [fx_seq[i],0,cx],
            [0,fy_seq[i],cy],
            [0,0,1],
        ], dtype=np.float32)
        for i in range(T)
    ])
    return K_seq
# ------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------- #

if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_viz")

    if True:
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
        
        
        for k, v in out.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    print(f"{kk}: {vv.shape if isinstance(vv, torch.Tensor) else type(vv)}")

            else:
                print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")
        print("\n\n")
        #Run visualization
        #padding_mask = out["padding_mask"][0].to(bool).cpu()
        #padded_traj = out["gen_samples"][0].cpu().unsqueeze(0)
        

        #####
        char_centers = out["char_raw"]["char_centers"][0].unsqueeze(0)
        padding_mask = out["padding_mask"][0].to(bool).cpu()
        padded_traj = out["gen_samples"][0].cpu()
        #padded_traj_mod = update_camera_rotations(padded_traj,char_centers)[0]
        traj = padded_traj[padding_mask]
        padded_vertices = out["char_raw"]["char_vertices"][0]
        vertices = padded_vertices[padding_mask]
        faces = out["char_raw"]["char_faces"][0]
        if faces.ndim == 3 and faces.shape[0] != vertices.shape[0]:
            faces = faces[padding_mask] 
        normals, meshes = get_normals(vertices, faces)
        fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
        caption = out["caption_raw"][0]
        T = traj.shape[0]
        K = gen_K_seq_from_caption(caption,fx,fy,cx,cy,T)
        



        accel = "cuda" if device.type == "cuda" else "cpu"

        rr.init(f"{SAMPLE_ID}", spawn=False)
        log_sample(
            root_name="world",
            traj=traj.detach().cpu().numpy(),
            K=K,
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            normals=normals.detach().cpu().numpy(),
            caption=caption,
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rr.save(f"./outputs/{SAMPLE_ID}_{timestamp}.rrd")
        print(f"saved rerun log to ./outputs/rrd_files/{SAMPLE_ID}_{timestamp}.rrd")

        cam_path =f"./outputs/camera_params/{SAMPLE_ID}_{timestamp}/"
        os.makedirs(cam_path, exist_ok=True)
        frame_list = [f"{i:06d}" for i in range(T)]
        ### save camera parameters(.json format)
        intrinsic = K.astype(np.float32)
        extrinsic = traj.detach().cpu().numpy().astype(np.float32)
        for i in range(T):
            R = extrinsic[i][:3, :3].tolist()
            t = extrinsic[i][:3, 3].tolist()
            fx_src = float(intrinsic[i][0, 0])
            fy_src = float(intrinsic[i][1, 1])
            cx_src = float(intrinsic[i][0, 2])
            cy_src = float(intrinsic[i][1, 2])
            data = {
                "R": R,
                "t": t,
                "focal":   [fx_src, fy_src],
                "princpt": [cx_src, cy_src],
            }
            with open(os.path.join(cam_path, str(frame_list[i]) + '.json'), 'w') as f:
                json.dump(data,f,indent=2)







    
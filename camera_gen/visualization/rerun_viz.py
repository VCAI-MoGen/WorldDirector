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
# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 2.0
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
def normalize(v, eps=1e-8):
    return v/(v.norm(dim=-1, keepdim=True)+eps)
def look_at_rotation(camera_pos,target_pos,up_vector=torch.tensor([0.,-1.,0.])):
    """camera pose : [N,3],
       target pose : [N,3],
       return pose : [N,3,3] rotation matrix
    """
    forward = normalize(camera_pos - target_pos)
    right = normalize(torch.cross(up_vector.expand_as(forward),forward,dim=-1))
    up = normalize(torch.cross(forward,right,dim=-1))
    rot = torch.stack([right,up,forward],dim=-1)
    return rot

def update_camera_rotations(gen_samples,char_centers):
    T=gen_samples.shape[1]
    new_extrinsics=gen_samples.clone()

    for t in range(T):
        camera_pos = gen_samples[0,t,:3,3]
        char_pos = char_centers[0,:,t]

        rot = look_at_rotation(camera_pos[None],char_pos[None])
        new_extrinsics[0,t,:3,:3] = rot[0]
    return new_extrinsics

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
        print(f"saved rerun log to ./outputs/{SAMPLE_ID}_{timestamp}.rrd")






    # sizes of batch from diffuser
        # traj_filename: <class 'list'>
        # traj_feat: torch.Size([1, 9, 300])
        # padding_mask: torch.Size([1, 300])
        # intrinsics: torch.Size([1, 4])
        # char_filename: <class 'list'>
        # char_feat: torch.Size([1, 3, 300])
        # char_raw:
        # char_raw_feat: torch.Size([1, 3, 300])
        # char_centers: torch.Size([1, 3, 300])
        # char_vertices: torch.Size([1, 300, 10475, 3])
        # char_faces: torch.Size([1, 300, 20908, 3])
        # char_padding_mask: torch.Size([1, 300])
        # caption_filename: <class 'list'>
        # caption_feat: torch.Size([1, 512, 77])
        # caption_raw: <class 'list'>
        # caption_padding_mask: torch.Size([1, 300])
        # conds:
        # char_raw_feat: torch.Size([1, 3, 300])
        # char_centers: torch.Size([1, 3, 300])
        # char_vertices: torch.Size([1, 300, 10475, 3])
        # char_faces: torch.Size([1, 300, 20908, 3])
        # caption: <class 'list'>
        # ref_samples: torch.Size([1, 300, 4, 4])
        # gen_samples: torch.Size([1, 300, 4, 4])
import torch
import numpy as np
import trimesh
from tqdm import tqdm

# ========== SDF from mesh (world-units; anisotropic voxel OK) ==========
def get_sdf(mesh_path, grid_shape, x_range, y_range, z_range, occ, batch_size):
    device = "cuda"

    # 1) Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces,    dtype=torch.long,   device=device)
    tri_verts = verts[faces]  # (F, 3, 3)
    F = tri_verts.shape[0]

    # 2) Make grid *centers* per axis (anisotropic ranges)
    X, Y, Z = grid_shape
    dx = (x_range[1] - x_range[0]) / X
    dy = (y_range[1] - y_range[0]) / Y
    dz = (z_range[1] - z_range[0]) / Z

    xs = torch.linspace(x_range[0] + dx*0.5, x_range[1] - dx*0.5, X, device=device)
    ys = torch.linspace(y_range[0] + dy*0.5, y_range[1] - dy*0.5, Y, device=device)
    zs = torch.linspace(z_range[0] + dz*0.5, z_range[1] - dz*0.5, Z, device=device)

    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # world coords

    # 3) Batched point-triangle min distance (world-units)
    print(f"⚙️  Computing distances for {len(points):,} points and {F:,} triangles...")
    sdf_vals = torch.empty(points.shape[0], device=device)

    v0, v1, v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]  # (F,3)
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    d00  = (v0v1 * v0v1).sum(-1)     # (F,)
    d01  = (v0v1 * v0v2).sum(-1)     # (F,)
    d11  = (v0v2 * v0v2).sum(-1)     # (F,)
    denom = d00 * d11 - d01 * d01 + 1e-9  # (F,)

    for i in tqdm(range(0, points.shape[0], batch_size)):
        p = points[i:i+batch_size]                 # (B,3)
        pvec = p[:, None, :] - v0[None, :, :]      # (B,F,3)

        d20 = (pvec * v0v1[None, :, :]).sum(-1)    # (B,F)
        d21 = (pvec * v0v2[None, :, :]).sum(-1)    # (B,F)

        v = (d11[None,:] * d20 - d01[None,:] * d21) / denom[None,:]   # (B,F)
        w = (d00[None,:] * d21 - d01[None,:] * d20) / denom[None,:]   # (B,F)
        u = 1.0 - v - w

        u = u.clamp(0, 1); v = v.clamp(0, 1); w = w.clamp(0, 1)

        q = (u[...,None]*v0[None,:,:] + v[...,None]*v1[None,:,:] + w[...,None]*v2[None,:,:])  # (B,F,3)
        dist = torch.linalg.norm(q - p[:,None,:], dim=-1)  # (B,F)
        sdf_vals[i:i+batch_size] = dist.min(dim=1).values  # (B,)

    sdf = sdf_vals.view(X, Y, Z)  # meters

    # 4) Sign with occupancy (optional): inside -> negative
    sdf[occ==1] *= -1
    return sdf.detach().cpu().numpy().astype(np.float32)

# ========== PLY export for Meshlab (red=near, blue=far) ==========
def export_sdf_to_ply(sdf, x_range, y_range, z_range,
                            out_ply_neg="sdf_neg.ply",
                            out_ply_pos="sdf_pos.ply",
                            sample_ratio=0.02,
                            color_gamma=2.0):
    X, Y, Z = sdf.shape
    dx = (x_range[1] - x_range[0]) / X
    dy = (y_range[1] - y_range[0]) / Y
    dz = (z_range[1] - z_range[0]) / Z

    xs = np.linspace(x_range[0] + dx * 0.5, x_range[1] - dx * 0.5, X)
    ys = np.linspace(y_range[0] + dy * 0.5, y_range[1] - dy * 0.5, Y)
    zs = np.linspace(z_range[0] + dz * 0.5, z_range[1] - dz * 0.5, Z)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    d = sdf.reshape(-1)

    # ======= Split by sign =======
    neg_mask = d < 0
    pos_mask = d > 0

    def write_ply(fname, pts_sel, dist_sel):
        if len(pts_sel) == 0:
            print(f"[Warning] No points to save in {fname}")
            return
        N = int(len(pts_sel) * sample_ratio)
        idx = np.random.choice(len(pts_sel), N, replace=False)

        pts_sel = pts_sel[idx]
        dist_sel = np.abs(dist_sel[idx])

        # Stronger color contrast: nonlinear gamma mapping
        d_max = np.percentile(dist_sel, 99)
        norm = np.clip(dist_sel / max(d_max, 1e-8), 0, 1) ** (1 / color_gamma)

        # red=near, blue=far (with stronger contrast)
        colors = np.stack([1 - norm, np.zeros_like(norm), norm], axis=-1)

        with open(fname, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts_sel)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts_sel, colors):
                r, g, b = (c * 255).astype(np.uint8)
                f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")
        print(f"Saved {fname} with {len(pts_sel)} points")

    # ======= Export negative (inside) =======
    write_ply(out_ply_neg, pts[neg_mask], d[neg_mask])

    # ======= Export positive (outside) =======
    write_ply(out_ply_pos, pts[pos_mask], d[pos_mask])

# =============================== MAIN ===============================
if __name__ == "__main__":
    mesh_path = "scene.obj"
    x_range = (-3.0, 3.0)
    y_range = (0.0, 2.0)
    z_range = (-4.0, 4.0)
    batch_size = 2000

    # occupancy grid
    occ = np.load("volume.npy")  # (X,Y,Z) in {0,1}
    grid_shape = occ.shape 
    
    # sdf
    sdf = get_sdf(mesh_path, grid_shape, x_range, y_range, z_range, occ, batch_size)
    np.save("scene_sdf.npy", sdf)
    export_sdf_to_ply(sdf, x_range, y_range, z_range)


import open3d as o3d
import numpy as np
import pickle
import torch
from smplx import SMPLX
from pytorch3d.transforms import matrix_to_axis_angle
from evo.tools.file_interface import read_kitti_poses_file
from scipy.spatial.transform import Rotation as R
MODEL_ROOT = "./et-data/human_model"  
TRAJ_NPY_PATH = "./et-data/char"
SMPLX_ROOT_DIR = "./et-data/smplh"
CAMERA_POSE_DIR = "./et-data/traj"



#
def normalize_camera_poses(camera_poses, R0, p0):
    """카메라 포즈 정규화 (TrajectoryDataset 로직 정확히 따름)"""
    T = camera_poses.shape[0]
    camera_poses_norm = np.zeros_like(camera_poses)
    
    for i in range(T):
        R_cam = camera_poses[i, :3, :3]
        t_cam = camera_poses[i, :3, 3]
        
        # TrajectoryDataset 로직 정확히 따르기:
        # raw_rot = (raw_rot.transpose(-2,-1) @ R0).T
        # raw_trans = (raw_trans - t0) @ R0
        R_cam_norm = (R_cam.T @ R0).T  # = R0.T @ R_cam
        t_cam_norm = (t_cam - p0) @ R0
        
        camera_poses_norm[i, :3, :3] = R_cam_norm
        camera_poses_norm[i, :3, 3] = t_cam_norm
        camera_poses_norm[i, 3, 3] = 1.0
    
    return camera_poses_norm

def denormalize_camera_poses(camera_poses_norm, R0, p0):
    """정규화된 카메라 포즈를 원본으로 복원"""
    T = camera_poses_norm.shape[0]
    camera_poses_restored = np.zeros_like(camera_poses_norm)
    
    for i in range(T):
        R_cam_norm = camera_poses_norm[i, :3, :3]
        t_cam_norm = camera_poses_norm[i, :3, 3]
        
        # 역변환: normalize에서 R_cam_norm = R0.T @ R_cam 이므로
        # R_cam = R0 @ R_cam_norm
        R_cam_restored = R0 @ R_cam_norm
        t_cam_restored = t_cam_norm @ R0.T + p0
        
        camera_poses_restored[i, :3, :3] = R_cam_restored
        camera_poses_restored[i, :3, 3] = t_cam_restored
        camera_poses_restored[i, 3, 3] = 1.0
    
    return camera_poses_restored

def print_coordinate_debug(sample_id, T, p0, R0, trajectory_points_raw, trajectory_norm, 
                          camera_poses_raw=None, camera_poses_norm=None, camera_poses_restored=None):
    """좌표 디버깅 정보 출력"""
    print(f"\n{'='*80}")
    print(f"COORDINATE DEBUG - Sample: {sample_id}")
    print(f"{'='*80}")
    
    # SMPLX Root Reference
    print(f"\n SMPLX ROOT REFERENCE (Frame 0)")
    print(f"   Root position (p0): [{p0[0]:8.4f}, {p0[1]:8.4f}, {p0[2]:8.4f}]")
    print(f"   Root rotation det(R0): {np.linalg.det(R0):12.8f}")
    print(f"   Orthogonality error: {np.abs(R0@R0.T - np.eye(3)).max():.2e}")
    
    # Character Trajectory (first few frames)
    debug_frames = [0, min(10, T//4), min(20, T//2), min(30, 3*T//4), T-1]
    print(f"\n CHARACTER TRAJECTORY COORDINATES")
    print(f"   Frame |        Original (Raw)        |      Normalized (Canonical)    ")
    print(f"   ------|------------------------------|----------------------------------")
    for frame in debug_frames:
        if frame < T:
            orig = trajectory_points_raw[frame]
            norm = trajectory_norm[frame]
            print(f"   {frame:5d} | [{orig[0]:8.4f}, {orig[1]:8.4f}, {orig[2]:8.4f}] | [{norm[0]:8.4f}, {norm[1]:8.4f}, {norm[2]:8.4f}]")
    
    # Camera Poses
    if camera_poses_raw is not None:
        print(f"\n CAMERA POSE COORDINATES")
        print(f"\n    Camera Positions:")
        print(f"   Frame |        Original (Raw)        |      Normalized (Canonical)     |        Restored (Check)     ")
        print(f"   ------|------------------------------|-----------------------------------|------------------------------")
        for frame in debug_frames:
            if frame < len(camera_poses_raw):
                orig_pos = camera_poses_raw[frame, :3, 3]
                norm_pos = camera_poses_norm[frame, :3, 3] if camera_poses_norm is not None else np.zeros(3)
                rest_pos = camera_poses_restored[frame, :3, 3] if camera_poses_restored is not None else np.zeros(3)
                print(f"   {frame:5d} | [{orig_pos[0]:8.4f}, {orig_pos[1]:8.4f}, {orig_pos[2]:8.4f}] | [{norm_pos[0]:8.4f}, {norm_pos[1]:8.4f}, {norm_pos[2]:8.4f}] | [{rest_pos[0]:8.4f}, {rest_pos[1]:8.4f}, {rest_pos[2]:8.4f}]")
        
        # Accuracy Check
        if camera_poses_restored is not None:
            pos_errors = np.linalg.norm(camera_poses_raw[:, :3, 3] - camera_poses_restored[:, :3, 3], axis=1)
            rot_errors = np.linalg.norm(camera_poses_raw[:, :3, :3] - camera_poses_restored[:, :3, :3], axis=(1, 2))
            
            print(f"\n NORMALIZATION/DENORMALIZATION ACCURACY")
            print(f"   Position errors - Mean: {np.mean(pos_errors):.2e}, Max: {np.max(pos_errors):.2e}")
            print(f"   Rotation errors - Mean: {np.mean(rot_errors):.2e}, Max: {np.max(rot_errors):.2e}")
            
            if np.max(pos_errors) > 1e-10 or np.max(rot_errors) > 1e-10:
                print(f"     WARNING: High errors detected!")
            else:
                print(f"    Accuracy verified")
    
    print(f"\n{'='*80}")

def main(sample_id: str):
    # 1) Trajectory 로드
    traj_path = f"{TRAJ_NPY_PATH}/{sample_id}.npy"
    trajectory_points_raw = np.load(traj_path).astype(np.float32)   # (T,3)
    T = trajectory_points_raw.shape[0]
    print(f"Loaded character trajectory: {T} frames")

    # 2) pkl 로드
    pkl_path = f"{SMPLX_ROOT_DIR}/{sample_id}.pkl"
    with open(pkl_path, "rb") as f:
        smpl_params = pickle.load(f, encoding="latin1")

    # 3) 텐서(모두 CPU)
    body_pose  = torch.tensor(smpl_params["body_pose"], dtype=torch.float32)        # (B,21,3,3)
    lhand_pose = torch.tensor(smpl_params["left_hand_pose"], dtype=torch.float32)   # (B,15,3,3)
    rhand_pose = torch.tensor(smpl_params["right_hand_pose"], dtype=torch.float32)  # (B,15,3,3)
    global_orient = torch.tensor(smpl_params["global_orient"], dtype=torch.float32) # (B,1,3,3) or (B,3,3)
    betas   = torch.tensor(smpl_params["betas"], dtype=torch.float32)               # (B,10) or (1,10)
    transl  = torch.tensor(smpl_params["transl"], dtype=torch.float32)             # (B,3)

    B = transl.shape[0]
    assert B == T, f"Frame mismatch: transl {B} vs traj {T}"

    # 4) shape 보정 + SO(3) 투영 (numpy에서)
    go_mat_np = global_orient.numpy()
    if go_mat_np.ndim == 4 and go_mat_np.shape[1] == 1:
        go_mat_np = go_mat_np[:, 0]   # (B,3,3)
    #go_mat_np = project_to_rotmat_batch(go_mat_np)    # (B,3,3), det≈1

    # 5) axis-angle 변환
    body_aa  = matrix_to_axis_angle(body_pose.view(B*21, 3, 3)).view(B, 21*3)
    lhand_aa = matrix_to_axis_angle(lhand_pose.view(B*15, 3, 3)).view(B, 15*3)
    rhand_aa = matrix_to_axis_angle(rhand_pose.view(B*15, 3, 3)).view(B, 15*3)
    go_aa    = matrix_to_axis_angle(torch.from_numpy(go_mat_np))     # (B,3)

    # 6) SMPL-X 호출 (CPU + no_grad)
    smpl = SMPLX(model_path=MODEL_ROOT, model_type="smplx", gender="neutral")
    smpl.use_pca = False
    with torch.no_grad():
        out = smpl(
            betas=betas.expand(B, -1) if betas.shape[0] == 1 else betas,
            body_pose=body_aa,               # (B,63)
            global_orient=go_aa,             # (B,3)
            transl=transl,                   # (B,3)
            left_hand_pose=lhand_aa,         # (B,45)
            right_hand_pose=rhand_aa,        # (B,45)
            jaw_pose=torch.zeros(B, 3),
            leye_pose=torch.zeros(B, 3),
            reye_pose=torch.zeros(B, 3),
            expression=torch.zeros(B, 10),
            return_verts=True
        )
    verts_raw  = out.vertices.numpy()   # (B,V,3)  meters
    joints_raw = out.joints.numpy()     # (B,J,3)
    faces = smpl.faces                  # (F,3) np.int32
    print(f"Generated SMPLX mesh: {verts_raw.shape[1]} vertices")

    # 7) 정규화: (x - p0) @ R0  ← training canonical
    root_idx = 0
    R0 = go_mat_np[0]                   # (3,3)
    p0 = joints_raw[0, root_idx]        # (3,)
    verts_norm     = (verts_raw - p0) @ R0
    joints_norm    = (joints_raw - p0) @ R0
    trajectory_norm = (trajectory_points_raw - p0) @ R0

    # 8) 진단: R0 품질 / 정규화 체크 / traj↔transl 일치도
    print(f"det(R0) = {np.linalg.det(R0):.6f} ; orth err = {np.abs(R0@R0.T - np.eye(3)).max():.2e}")
    print(f"|pelvis_after_norm| = {np.linalg.norm(joints_norm[0,root_idx]):.6e}")
    # transl vs traj 평균 오프셋/스케일 확인 (단위 불일치 탐지)
    transl_np = transl.numpy()
    off = np.mean(np.linalg.norm(transl_np - trajectory_points_raw, axis=1))
    print(f"mean |transl - traj| = {off:.6f} (units?)")

    # 9) 카메라 포즈 로드 및 정규화
    camera_poses_raw = None
    camera_poses_norm = None
    camera_poses_restored = None
    
    camera_path = f"{CAMERA_POSE_DIR}/{sample_id}.txt"
    if os.path.exists(camera_path):
        print(f"Loading camera poses: {camera_path}")
        trajectory = read_kitti_poses_file(camera_path)
        camera_poses_raw = np.array(trajectory.poses_se3).astype(np.float32)  # (T, 4, 4)
        print(f"Loaded {len(camera_poses_raw)} camera poses")
        
        # 길이 맞추기
        if len(camera_poses_raw) != T:
            print(f"Warning: Camera length ({len(camera_poses_raw)}) != SMPLX frames ({T})")
            if len(camera_poses_raw) < T:
                last_pose = camera_poses_raw[-1]
                camera_poses_raw = np.concatenate([
                    camera_poses_raw, 
                    np.tile(last_pose[None], (T - len(camera_poses_raw), 1, 1))
                ])
            else:
                camera_poses_raw = camera_poses_raw[:T]
        
        # 정규화 (같은 R0, p0 사용)
        camera_poses_norm = normalize_camera_poses(camera_poses_raw, R0, p0)
        
        # 역정규화 (검증용)
        camera_poses_restored = denormalize_camera_poses(camera_poses_norm, R0, p0)
        
        print(f"Camera poses normalized with SMPLX root reference")
    else:
        print(f"Camera poses not found: {camera_path}")

    # 10) 좌표 디버깅 정보 출력
    print_coordinate_debug(sample_id, T, p0, R0, trajectory_points_raw, trajectory_norm,
                          camera_poses_raw, camera_poses_norm, camera_poses_restored)

    # 11) 뷰어 변환(필요할 때만 적용) — Open3D가 Y축이 반대로 보이면 사용
    #    A) Y-flip만:
    #view_R = np.diag([1, -1, 1]).astype(np.float32)
    #    B) X축 180도 회전(= Y,Z 플립):
    # angle = np.pi
    # view_R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]], np.float32)
    # verts_view     = verts_norm @ view_R.T
    # trajectory_view = trajectory_norm @ view_R.T
    # # → 저장은 viewer 버전으로 하거나, 학습용과 분리:
    verts_view = verts_norm.copy()
    trajectory_view = trajectory_norm.copy()

    # 12) 저장
    out_path = f"./debug_vis/{sample_id}_for_viz.npz"
    os.makedirs("./debug_vis", exist_ok=True)
    
    save_data = {
        'vertices': verts_view,
        'trajectory': trajectory_view,
        'faces': faces,
        'root_position': p0,
        'root_rotation': R0,
        # 원본 데이터도 저장
        'vertices_raw': verts_raw,
        'trajectory_raw': trajectory_points_raw,
        'joints_raw': joints_raw,
        'joints_norm': joints_norm
    }
    
    # 카메라 데이터 추가 (있는 경우)
    if camera_poses_raw is not None:
        save_data.update({
            'camera_poses_raw': camera_poses_raw,
            'camera_poses_norm': camera_poses_norm,
            'camera_poses_restored': camera_poses_restored
        })
    
    np.savez_compressed(out_path, **save_data)
    print(f"Saved: {out_path}")
    
    # 저장 내용 요약
    print(f"\n  Saved data summary:")
    print(f"   - SMPLX: vertices, joints, trajectory (raw + normalized)")
    print(f"   - Root reference: position + rotation matrix")
    if camera_poses_raw is not None:
        print(f"   - Camera poses: raw + normalized + restored")
    print(f"   - Faces: {faces.shape[0]} triangles")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_id", required=True)
    args = ap.parse_args()
    main(args.sample_id)
from __future__ import annotations

import argparse
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from evo.tools.file_interface import read_kitti_poses_file
from pytorch3d.transforms import matrix_to_axis_angle
from smplx import SMPLX
import viser
from viser import transforms as viser_tf

## data_dir 
MODEL_ROOT = "./et-data/human_model" #smplx_netural .pkl file direction
TRAJ_NPY_PATH = "./et-data/char"
SMPLX_ROOT_DIR = "./et-data/smplh"
CAMERA_POSE_DIR = "./et-data/traj"

FRUSTUM_SCALE = 0.18
ROOT_AXIS_LENGTH = 0.25
ROOT_AXIS_RADIUS = 0.012
CAM_AXIS_LENGTH = 0.22
CAM_AXIS_RADIUS = 0.010

PELVIS_JOINT_IDX = 0
LEFT_SHOULDER_JOINT_IDX = 16
RIGHT_SHOULDER_JOINT_IDX = 17


def _normalize_rotation(matrix: np.ndarray) -> np.ndarray:
    return viser_tf.SO3.from_matrix(matrix).as_matrix().astype(matrix.dtype, copy=False)


def _normalize_rotations(matrices: np.ndarray) -> np.ndarray:
    if matrices.ndim == 2:
        return _normalize_rotation(matrices)
    return np.stack([_normalize_rotation(m) for m in matrices], axis=0)


def normalize_camera_poses(camera_poses: np.ndarray, R0: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Match TrajectoryDataset normalization for cam-to-world coordinates."""
    T = camera_poses.shape[0]
    camera_poses_norm = np.zeros_like(camera_poses)

    R0_inv = R0.T
    for i in range(T):
        R_cam = camera_poses[i, :3, :3]
        t_cam = camera_poses[i, :3, 3]

        R_cam = _normalize_rotation(R_cam)
        
        R_cam_norm = R0_inv @ R_cam
        t_cam_norm = t_cam  @ R0_inv.T

        camera_poses_norm[i, :3, :3] = R_cam_norm
        camera_poses_norm[i, :3, 3] = t_cam_norm
        camera_poses_norm[i, 3, :3] = 0.0
        camera_poses_norm[i, 3, 3] = 1.0

    return camera_poses_norm


def _load_smpl_outputs(sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load SMPL-X parameters and generate vertices / joints."""
    pkl_path = f"{SMPLX_ROOT_DIR}/{sample_id}.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing SMPL-X params: {pkl_path}")

    with open(pkl_path, "rb") as f:
        smpl_params = pickle.load(f, encoding="latin1")

    body_pose = torch.tensor(smpl_params["body_pose"], dtype=torch.float32)
    lhand_pose = torch.tensor(smpl_params["left_hand_pose"], dtype=torch.float32)
    rhand_pose = torch.tensor(smpl_params["right_hand_pose"], dtype=torch.float32)
    global_orient = torch.tensor(smpl_params["global_orient"], dtype=torch.float32)
    betas = torch.tensor(smpl_params["betas"], dtype=torch.float32)
    transl = torch.tensor(smpl_params["transl"], dtype=torch.float32)

    B = transl.shape[0]
    go_mat_np = global_orient.numpy()
    if go_mat_np.ndim == 4 and go_mat_np.shape[1] == 1:
        go_mat_np = go_mat_np[:, 0]
    
    # Normalize all rotation matrices to ensure orthonormal axes
    go_mat_np = _normalize_rotations(go_mat_np)

    body_aa = matrix_to_axis_angle(body_pose.view(B * 21, 3, 3)).view(B, 21 * 3)
    lhand_aa = matrix_to_axis_angle(lhand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)
    rhand_aa = matrix_to_axis_angle(rhand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)
    go_aa = matrix_to_axis_angle(torch.from_numpy(go_mat_np))

    smpl = SMPLX(model_path=MODEL_ROOT, model_type="smplx", gender="neutral")
    smpl.use_pca = False
    with torch.no_grad():
        out = smpl(
            betas=betas.expand(B, -1) if betas.shape[0] == 1 else betas,
            body_pose=body_aa,
            global_orient=go_aa,
            transl=transl,
            left_hand_pose=lhand_aa,
            right_hand_pose=rhand_aa,
            jaw_pose=torch.zeros(B, 3),
            leye_pose=torch.zeros(B, 3),
            reye_pose=torch.zeros(B, 3),
            expression=torch.zeros(B, 10),
            return_verts=True,
        )

    verts_raw = out.vertices.numpy().astype(np.float32)
    joints_raw = out.joints.numpy().astype(np.float32)
    faces = smpl.faces.astype(np.int32)
    transl_np = transl.numpy().astype(np.float32)

    return verts_raw, joints_raw, faces, go_mat_np.astype(np.float32), transl_np


@dataclass(slots=True)
class TrajectoryData:
    sample_id: str
    vertices_raw: np.ndarray
    trajectory_raw: np.ndarray
    faces: np.ndarray
    root_position: np.ndarray
    vertices_norm: Optional[np.ndarray] = None
    joints_norm: Optional[np.ndarray] = None
    trajectory_norm: Optional[np.ndarray] = None
    root_rotations_raw: Optional[np.ndarray] = None
    root_rotations_norm: Optional[np.ndarray] = None
    traj_frames_raw: Optional[np.ndarray] = None
    traj_frames_norm: Optional[np.ndarray] = None
    camera_poses_raw: Optional[np.ndarray] = None
    camera_poses_norm: Optional[np.ndarray] = None
    camera_positions_raw: Optional[np.ndarray] = None
    camera_positions_norm: Optional[np.ndarray] = None
    is_canonical: bool = True
    root_axes_normalized: Optional[dict[str, np.ndarray]] = None


@dataclass(slots=True)
class SampleHandles:
    data: TrajectoryData
    frame_count: int
    show_canonical_mesh: Optional[viser.GuiCheckboxHandle]
    show_canonical_traj: Optional[viser.GuiCheckboxHandle]
    show_world_mesh: viser.GuiCheckboxHandle
    show_world_traj: viser.GuiCheckboxHandle
    show_cameras: viser.GuiCheckboxHandle
    mesh_canonical: Optional[viser.MeshHandle]
    mesh_world: viser.MeshHandle
    traj_canonical: Optional[viser.LineSegmentsHandle]
    traj_world: viser.LineSegmentsHandle
    cam_path_canonical: Optional[viser.LineSegmentsHandle]
    cam_path_world: Optional[viser.LineSegmentsHandle]
    cam_frustum_canonical: Optional[viser.CameraFrustumHandle]
    cam_frustum_world: Optional[viser.CameraFrustumHandle]
    root_frame_canonical: Optional[viser.FrameHandle]
    root_frame_world: Optional[viser.FrameHandle]
    cam_frame_canonical: Optional[viser.FrameHandle]
    cam_frame_world: Optional[viser.FrameHandle]
    traj_frame_canonical: Optional[viser.FrameHandle]
    traj_frame_world: Optional[viser.FrameHandle]
    smpl_axis_text: viser.GuiMarkdownHandle
    traj_axis_text: viser.GuiMarkdownHandle
    cam_axis_text: viser.GuiMarkdownHandle


def prepare_sample(sample_id: str) -> TrajectoryData:
    npy_path = f"{TRAJ_NPY_PATH}/{sample_id}.npy"
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing character trajectory: {npy_path}")

    trajectory_points_raw = np.load(npy_path).astype(np.float32)
    verts_raw, joints_raw, faces, go_mat_np, _ = _load_smpl_outputs(sample_id)

    root_idx = 0
    R0 = go_mat_np[0]  # Already normalized in _load_smpl_outputs
    p0 = joints_raw[0, root_idx]
    verts_norm = (verts_raw) @ R0
    joints_norm = (joints_raw) @ R0
    trajectory_norm = (trajectory_points_raw) @ R0
    traj_frames_raw = _compute_path_frames(trajectory_points_raw)
    traj_frames_norm = _compute_path_frames(trajectory_norm)
    
    traj_frames_raw = _normalize_rotations(traj_frames_raw)
    traj_frames_norm = _normalize_rotations(traj_frames_norm)
    
    ref_rot_raw = _normalize_rotations(go_mat_np)
    ref_rot_norm = _normalize_rotations(go_mat_np @ R0)
    root_rotations_raw = _compute_torso_frames_from_joints(joints_raw, reference=ref_rot_raw)
    if root_rotations_raw is None:
        print(f"[viser_viz] Fallback (raw axes): using SMPL global_orient for sample '{sample_id}'.")
        root_rotations_raw = ref_rot_raw
    root_rotations_norm = _compute_torso_frames_from_joints(joints_norm, reference=ref_rot_norm)
    if root_rotations_norm is None:
        print(f"[viser_viz] Fallback (canonical axes): using SMPL global_orient for sample '{sample_id}'.")
        root_rotations_norm = ref_rot_norm
    root_rotations_raw = root_rotations_raw.astype(np.float32)
    root_rotations_norm = root_rotations_norm.astype(np.float32)
    x_vecs = go_mat_np[:, :, 0]
    y_vecs = go_mat_np[:, :, 1]
    z_vecs = go_mat_np[:, :, 2]

    # normalize each vector to unit length
    x_vecs /= np.linalg.norm(x_vecs, axis=1, keepdims=True) + 1e-8
    y_vecs /= np.linalg.norm(y_vecs, axis=1, keepdims=True) + 1e-8
    z_vecs /= np.linalg.norm(z_vecs, axis=1, keepdims=True) + 1e-8

    root_axes_normalized = {
        "x": x_vecs.astype(np.float32),
        "y": y_vecs.astype(np.float32),
        "z": z_vecs.astype(np.float32),
    }

    camera_path = f"{CAMERA_POSE_DIR}/{sample_id}.txt"
    camera_poses_raw: Optional[np.ndarray] = None
    camera_positions_raw: Optional[np.ndarray] = None
    camera_poses_norm: Optional[np.ndarray] = None
    camera_positions_norm: Optional[np.ndarray] = None

    if os.path.exists(camera_path):
        trajectory = read_kitti_poses_file(camera_path)
        camera_poses_raw = np.array(trajectory.poses_se3).astype(np.float32)
        camera_poses_raw[:, :3, :3] = _normalize_rotations(camera_poses_raw[:, :3, :3])
        camera_poses_norm = normalize_camera_poses(camera_poses_raw, R0, p0)
        camera_positions_raw = camera_poses_raw[:, :3, 3].astype(np.float32)
        camera_positions_norm = camera_poses_norm[:, :3, 3].astype(np.float32)

    return TrajectoryData(
        sample_id=sample_id,
        vertices_raw=verts_raw,
        vertices_norm=verts_norm.astype(np.float32),
        joints_norm=joints_norm.astype(np.float32),
        trajectory_raw=trajectory_points_raw,
        trajectory_norm=trajectory_norm.astype(np.float32),
        faces=faces,
        root_position=p0.astype(np.float32),
        root_rotations_raw=root_rotations_raw,
        root_rotations_norm=root_rotations_norm,
        traj_frames_raw=traj_frames_raw.astype(np.float32),
        traj_frames_norm=traj_frames_norm.astype(np.float32),
        camera_poses_raw=camera_poses_raw,
        camera_poses_norm=camera_poses_norm,
        camera_positions_raw=camera_positions_raw,
        camera_positions_norm=camera_positions_norm,
        root_axes_normalized=root_axes_normalized
    )


def _make_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    seg = np.stack([points[:-1], points[1:]], axis=1)
    return seg.astype(np.float32)


def _compute_path_frames(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    frames = np.zeros((len(points), 3, 3), dtype=np.float32)

    for i in range(len(points)):
        if i < len(points) - 1:
            forward = points[i + 1] - points[i]
        else:
            forward = points[i] - points[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        norm_forward = np.linalg.norm(forward)
        if norm_forward < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            forward = forward / norm_forward

        right = np.cross(up, forward)
        norm_right = np.linalg.norm(right)
        if norm_right < 1e-6:
            aux = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(aux, forward)
            norm_right = np.linalg.norm(right)
            if norm_right < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                right = right / norm_right
        else:
            right = right / norm_right

        up_local = np.cross(forward, right)
        norm_up = np.linalg.norm(up_local)
        if norm_up < 1e-6:
            up_local = up
        else:
            up_local = up_local / norm_up

        frames[i, :, :] = np.stack([right, up_local, forward], axis=1)

    return frames.astype(np.float32)


def _compute_torso_frames_from_joints(
    joints: np.ndarray, reference: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    if joints is None or joints.ndim != 3:
        return None
    num_joints = joints.shape[1]
    required = [PELVIS_JOINT_IDX, LEFT_SHOULDER_JOINT_IDX, RIGHT_SHOULDER_JOINT_IDX]
    if any(idx >= num_joints for idx in required):
        return None

    frames = []
    for idx, joint_set in enumerate(joints):
        pelvis = joint_set[PELVIS_JOINT_IDX]
        left_shoulder = joint_set[LEFT_SHOULDER_JOINT_IDX]
        right_shoulder = joint_set[RIGHT_SHOULDER_JOINT_IDX]

        x_axis = right_shoulder - left_shoulder
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            x_axis = x_axis / x_norm

        shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
        y_axis = shoulder_mid - pelvis
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            fallback = fallback - np.dot(fallback, x_axis) * x_axis
            if np.linalg.norm(fallback) < 1e-6:
                fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            y_axis = fallback / np.linalg.norm(fallback)
        else:
            y_axis = y_axis / y_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            z_axis = np.cross(x_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            z_norm = np.linalg.norm(z_axis)
            if z_norm < 1e-6:
                z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                z_axis = z_axis / z_norm
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
        else:
            z_axis = z_axis / z_norm

        if reference is not None:
            ref_axes = reference[min(len(reference) - 1, idx)]
            if ref_axes.shape == (3, 3):
                if np.dot(x_axis, ref_axes[:, 0]) < 0:
                    x_axis = -x_axis
                if np.dot(y_axis, ref_axes[:, 1]) < 0:
                    y_axis = -y_axis
                z_axis = np.cross(x_axis, y_axis)
                z_axis /= np.linalg.norm(z_axis)
                if np.dot(z_axis, ref_axes[:, 2]) < 0:
                    z_axis = -z_axis
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)

        frame = np.stack([x_axis, y_axis, z_axis], axis=1)
        frames.append(frame.astype(np.float32))

    if not frames:
        return None

    frames_np = np.stack(frames, axis=0)
    return _normalize_rotations(frames_np).astype(np.float32)


def _axes_markdown(title: str, rotation: Optional[np.ndarray], position: Optional[np.ndarray]) -> str:
    def _fmt(vec: Optional[np.ndarray]) -> str:
        if vec is None or vec.shape[0] != 3:
            return "N/A"
        return f"({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f})"

    if rotation is not None and rotation.shape == (3, 3):
        x_vec = rotation[:, 0]
        y_vec = rotation[:, 1]
        z_vec = rotation[:, 2]
    else:
        x_vec = y_vec = z_vec = None

    return "\n".join(
        [
            f"**{title}**",
            f"- position: `{_fmt(position)}`",
            "- axes:",
            f"    - X: `{_fmt(x_vec)}`",
            f"    - Y: `{_fmt(y_vec)}`",
            f"    - Z: `{_fmt(z_vec)}`",
        ]
    )


def _normalize_matrix(matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if matrix is None or matrix.shape != (3, 3):
        return None
    return _normalize_rotation(matrix)


# viser viewer
def launch_viewer(
    datasets: Sequence[TrajectoryData],
    host: str,
    port: int,
) -> None:
    if len(datasets) == 0:
        raise ValueError("At least one sample is required")

    max_frames = max(
        (data.vertices_norm.shape[0] if data.vertices_norm is not None else data.vertices_raw.shape[0])
        for data in datasets
    )
    label = ", ".join(data.sample_id for data in datasets)
    server = viser.ViserServer(host=host, port=port, label=f"Samples: {label}")

    sample_summary = "\n".join(
        f"- `{data.sample_id}`: "
        f"{data.vertices_norm.shape[0] if data.vertices_norm is not None else data.vertices_raw.shape[0]} frames"
        for data in datasets
    )
    info = server.gui.add_markdown(
        "**Loaded Samples**\n" + sample_summary
    )
    info.visible = True

    with server.gui.add_folder("Playback", expand_by_default=True):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=max_frames - 1 if max_frames > 0 else 0, step=1, initial_value=0
        )
        fps_slider = server.gui.add_slider("Playback FPS", min=1, max=60, step=1, initial_value=24)
        play_button = server.gui.add_button("Play")
        play_status = server.gui.add_text("Playback state", "Paused", disabled=True)

    sample_handles: list[SampleHandles] = []

    for idx, data in enumerate(datasets):
        has_canonical = (
            data.is_canonical
            and data.vertices_norm is not None
            and data.trajectory_norm is not None
        )
        initial_show_canonical_mesh = has_canonical and idx == 0
        initial_show_canonical_traj = initial_show_canonical_mesh
        initial_show_world_mesh = True
        initial_show_world_traj = True
        initial_show_cameras = data.camera_positions_raw is not None and idx == 0
        initial_show_world_any = initial_show_world_mesh or initial_show_world_traj
        initial_show_canonical_any = initial_show_canonical_mesh or initial_show_canonical_traj

        with server.gui.add_folder(
            f"Sample {idx + 1}: {data.sample_id}", expand_by_default=idx == 0
        ):
            show_canonical_mesh = None
            show_canonical_traj = None
            if has_canonical:
                with server.gui.add_folder("Canonical", expand_by_default=idx == 0):
                    show_canonical_mesh = server.gui.add_checkbox(
                        "Show mesh", initial_value=initial_show_canonical_mesh
                    )
                    show_canonical_traj = server.gui.add_checkbox(
                        "Show trajectory", initial_value=initial_show_canonical_traj
                    )
            with server.gui.add_folder("World", expand_by_default=idx == 0):
                show_world_mesh = server.gui.add_checkbox(
                    "Show mesh", initial_value=initial_show_world_mesh
                )
                show_world_traj = server.gui.add_checkbox(
                    "Show trajectory", initial_value=initial_show_world_traj
                )
            show_cameras = server.gui.add_checkbox(
                "Show cameras", initial_value=initial_show_cameras
            )
            with server.gui.add_folder("Axis Values", expand_by_default=False):
                smpl_axis_text = server.gui.add_markdown(
                    _axes_markdown("SMPL (world)", None, None)
                )
                traj_axis_text = server.gui.add_markdown(
                    _axes_markdown("Trajectory (world)", None, None)
                )
                cam_axis_text = server.gui.add_markdown(
                    _axes_markdown("Camera (world)", None, None)
                )

        world_vertices0 = data.vertices_raw[0]

        mesh_world = server.scene.add_mesh_simple(
            f"/samples/{idx}/world/mesh",
            vertices=world_vertices0,
            faces=data.faces,
            color=(255, 170, 90),
            opacity=0.55,
            visible=initial_show_world_mesh,
        )

        traj_world = server.scene.add_line_segments(
            f"/samples/{idx}/world/trajectory",
            points=_make_segments(data.trajectory_raw),
            colors=(255, 105, 180),
            line_width=4.0,
            visible=initial_show_world_traj,
        )

        mesh_canonical = None
        traj_canonical = None
        if has_canonical:
            canonical_vertices0 = data.vertices_norm[0]
            mesh_canonical = server.scene.add_mesh_simple(
                f"/samples/{idx}/canonical/mesh",
                vertices=canonical_vertices0,
                faces=data.faces,
                color=(90, 200, 255),
                flat_shading=False,
                visible=initial_show_canonical_mesh,
            )
            traj_canonical = server.scene.add_line_segments(
                f"/samples/{idx}/canonical/trajectory",
                points=_make_segments(data.trajectory_norm),
                colors=(255, 105, 150),
                line_width=4.0,
                visible=initial_show_canonical_traj,
            )

        cam_path_canonical = None
        cam_path_world = None
        cam_frustum_canonical = None
        cam_frustum_world = None
        root_frame_canonical = None
        root_frame_world = None
        cam_frame_canonical = None
        cam_frame_world = None
        traj_frame_canonical = None
        traj_frame_world = None

        if has_canonical and data.camera_positions_norm is not None:
            cam_path_canonical = server.scene.add_line_segments(
                f"/samples/{idx}/canonical/camera_path",
                points=_make_segments(data.camera_positions_norm),
                colors=(128, 0, 255),
                line_width=4.0,
                visible=initial_show_canonical_any and initial_show_cameras,
            )
        if data.camera_positions_raw is not None:
            cam_path_world = server.scene.add_line_segments(
                f"/samples/{idx}/world/camera_path",
                points=_make_segments(data.camera_positions_raw),
                colors=(128, 100, 255),
                line_width=4.0,
                visible=initial_show_world_any and initial_show_cameras,
            )
        if has_canonical and data.camera_poses_norm is not None and data.camera_positions_norm is not None:
            R_cam = data.camera_poses_norm[0][:3, :3]
            t_cam = data.camera_positions_norm[0]
            quat_norm = viser_tf.SO3.from_matrix(R_cam).wxyz
            cam_frustum_canonical = server.scene.add_camera_frustum(
                f"/samples/{idx}/canonical/camera",
                fov=np.deg2rad(60.0),
                aspect=16 / 9,
                position=t_cam,
                visible=initial_show_canonical_any and initial_show_cameras,
                wxyz=quat_norm,
                color=(80, 180, 255),
                scale=FRUSTUM_SCALE,
            )
        if data.camera_poses_raw is not None and data.camera_positions_raw is not None:
            R_cam = data.camera_poses_raw[0][:3, :3]
            t_cam = data.camera_positions_raw[0]
            quat_world = viser_tf.SO3.from_matrix(R_cam).wxyz
            cam_frustum_world = server.scene.add_camera_frustum(
                f"/samples/{idx}/world/camera",
                fov=np.deg2rad(60.0),
                aspect=16 / 9,
                position=t_cam,
                visible=initial_show_world_any and initial_show_cameras,
                wxyz=quat_world,
                color=(255, 180, 80),
                scale=FRUSTUM_SCALE,
            )
        if (
            data.trajectory_raw.shape[0] > 0
            and data.root_rotations_raw is not None
            and data.root_rotations_raw.shape[0] > 0
        ):
            root_rotation_world = viser_tf.SO3.from_matrix(data.root_rotations_raw[0]).wxyz
            root_frame_world = server.scene.add_frame(
                f"/samples/{idx}/world/root_frame",
                position=data.trajectory_raw[0],
                wxyz=root_rotation_world,
                show_axes=True,
                axes_length=ROOT_AXIS_LENGTH,
                axes_radius=ROOT_AXIS_RADIUS,
                origin_radius=ROOT_AXIS_RADIUS * 2.5,
                origin_color=(255, 200, 120),
            )
            root_frame_world.visible = initial_show_world_mesh
        if (
            has_canonical
            and data.trajectory_norm is not None
            and data.trajectory_norm.shape[0] > 0
        ):
            root_rot_canonical = (
                viser_tf.SO3.from_matrix(data.root_rotations_norm[0]).wxyz
                if data.root_rotations_norm is not None and data.root_rotations_norm.shape[0] > 0
                else (1.0, 0.0, 0.0, 0.0)
            )
            root_frame_canonical = server.scene.add_frame(
                f"/samples/{idx}/canonical/root_frame",
                position=data.trajectory_norm[0],
                wxyz=root_rot_canonical,
                show_axes=True,
                axes_length=ROOT_AXIS_LENGTH,
                axes_radius=ROOT_AXIS_RADIUS,
                origin_radius=ROOT_AXIS_RADIUS * 2.5,
                origin_color=(200, 230, 255),
            )
            root_frame_canonical.visible = initial_show_canonical_mesh
        if (
            has_canonical
            and data.camera_poses_norm is not None
            and data.camera_positions_norm is not None
            and data.camera_positions_norm.shape[0] > 0
        ):
            cam_frame_canonical = server.scene.add_frame(
                f"/samples/{idx}/canonical/camera_frame",
                position=data.camera_positions_norm[0],
                wxyz=viser_tf.SO3.from_matrix(data.camera_poses_norm[0][:3, :3]).wxyz,
                show_axes=True,
                axes_length=CAM_AXIS_LENGTH,
                axes_radius=CAM_AXIS_RADIUS,
                origin_radius=CAM_AXIS_RADIUS * 2.5,
                origin_color=(120, 200, 255),
            )
            cam_frame_canonical.visible = initial_show_canonical_any and initial_show_cameras
        if (
            data.camera_poses_raw is not None
            and data.camera_positions_raw is not None
            and data.camera_positions_raw.shape[0] > 0
        ):
            cam_frame_world = server.scene.add_frame(
                f"/samples/{idx}/world/camera_frame",
                position=data.camera_positions_raw[0],
                wxyz=viser_tf.SO3.from_matrix(data.camera_poses_raw[0][:3, :3]).wxyz,
                show_axes=True,
                axes_length=CAM_AXIS_LENGTH,
                axes_radius=CAM_AXIS_RADIUS,
                origin_radius=CAM_AXIS_RADIUS * 2.5,
                origin_color=(255, 210, 120),
            )
            cam_frame_world.visible = initial_show_world_any and initial_show_cameras
        if data.traj_frames_raw is not None and data.traj_frames_raw.shape[0] > 0:
            traj_frame_world = server.scene.add_frame(
                f"/samples/{idx}/world/trajectory_frame",
                position=data.trajectory_raw[0],
                wxyz=viser_tf.SO3.from_matrix(data.traj_frames_raw[0]).wxyz,
                show_axes=True,
                axes_length=ROOT_AXIS_LENGTH * 0.9,
                axes_radius=ROOT_AXIS_RADIUS,
                origin_radius=ROOT_AXIS_RADIUS * 2.0,
                origin_color=(255, 140, 140),
            )
            traj_frame_world.visible = initial_show_world_traj
        if (
            has_canonical
            and data.traj_frames_norm is not None
            and data.traj_frames_norm.shape[0] > 0
        ):
            traj_frame_canonical = server.scene.add_frame(
                f"/samples/{idx}/canonical/trajectory_frame",
                position=data.trajectory_norm[0],
                wxyz=viser_tf.SO3.from_matrix(data.traj_frames_norm[0]).wxyz,
                show_axes=True,
                axes_length=ROOT_AXIS_LENGTH * 0.9,
                axes_radius=ROOT_AXIS_RADIUS,
                origin_radius=ROOT_AXIS_RADIUS * 2.0,
                origin_color=(200, 140, 255),
            )
            traj_frame_canonical.visible = initial_show_canonical_traj

        sample_handles.append(
            SampleHandles(
                data=data,
                frame_count=data.vertices_raw.shape[0],
                show_canonical_mesh=show_canonical_mesh,
                show_canonical_traj=show_canonical_traj,
                show_world_mesh=show_world_mesh,
                show_world_traj=show_world_traj,
                show_cameras=show_cameras,
                mesh_canonical=mesh_canonical,
                mesh_world=mesh_world,
                traj_canonical=traj_canonical,
                traj_world=traj_world,
                cam_path_canonical=cam_path_canonical,
                cam_path_world=cam_path_world,
                cam_frustum_canonical=cam_frustum_canonical,
                cam_frustum_world=cam_frustum_world,
                root_frame_canonical=root_frame_canonical,
                root_frame_world=root_frame_world,
                cam_frame_canonical=cam_frame_canonical,
                cam_frame_world=cam_frame_world,
                traj_frame_canonical=traj_frame_canonical,
                traj_frame_world=traj_frame_world,
                smpl_axis_text=smpl_axis_text,
                traj_axis_text=traj_axis_text,
                cam_axis_text=cam_axis_text,
            )
        )

    playback_state = {"running": False}

    def update_frame(frame: int) -> None:
        for handle in sample_handles:
            idx = min(frame, handle.frame_count - 1)
            smpl_rot_mat: Optional[np.ndarray] = None
            smpl_pos: Optional[np.ndarray] = None
            traj_rot_mat: Optional[np.ndarray] = None
            traj_pos: Optional[np.ndarray] = None
            if (
                handle.data.root_rotations_raw is not None
                and handle.data.root_rotations_raw.shape[0] > 0
            ):
                rot_idx = min(idx, handle.data.root_rotations_raw.shape[0] - 1)
                smpl_rot_mat = handle.data.root_rotations_raw[rot_idx]
            if handle.data.trajectory_raw.shape[0] > 0:
                traj_pos_idx = min(idx, handle.data.trajectory_raw.shape[0] - 1)
                smpl_pos = handle.data.trajectory_raw[traj_pos_idx]
                traj_pos = handle.data.trajectory_raw[traj_pos_idx]
            if (
                handle.data.traj_frames_raw is not None
                and handle.data.traj_frames_raw.shape[0] > 0
            ):
                traj_rot_mat = handle.data.traj_frames_raw[
                    min(idx, handle.data.traj_frames_raw.shape[0] - 1)
                ]
            
            # Normalize matrices before using them
            smpl_rot_norm = _normalize_matrix(smpl_rot_mat)
            traj_rot_norm = _normalize_matrix(traj_rot_mat)
            cam_rot_norm = None

            # Canonical space updates.
            show_canonical_mesh = (
                handle.show_canonical_mesh.value if handle.show_canonical_mesh is not None else False
            )
            show_canonical_traj = (
                handle.show_canonical_traj.value if handle.show_canonical_traj is not None else False
            )
            show_canonical_any = show_canonical_mesh or show_canonical_traj
            if handle.mesh_canonical is not None:
                handle.mesh_canonical.visible = show_canonical_mesh
                if show_canonical_mesh and handle.data.vertices_norm is not None:
                    handle.mesh_canonical.vertices = handle.data.vertices_norm[idx]
            if handle.traj_canonical is not None:
                handle.traj_canonical.visible = show_canonical_traj
            if handle.root_frame_canonical is not None:
                handle.root_frame_canonical.visible = show_canonical_mesh
                if show_canonical_mesh and handle.data.trajectory_norm is not None:
                    pos_idx = min(idx, handle.data.trajectory_norm.shape[0] - 1)
                    handle.root_frame_canonical.position = handle.data.trajectory_norm[pos_idx]
                    if (
                        handle.data.root_rotations_norm is not None
                        and handle.data.root_rotations_norm.shape[0] > 0
                    ):
                        rot_idx = min(idx, handle.data.root_rotations_norm.shape[0] - 1)
                        root_rot_norm = handle.data.root_rotations_norm[rot_idx]
                        handle.root_frame_canonical.wxyz = viser_tf.SO3.from_matrix(root_rot_norm).wxyz
            if handle.traj_frame_canonical is not None:
                handle.traj_frame_canonical.visible = show_canonical_traj
                if (
                    show_canonical_traj
                    and handle.data.trajectory_norm is not None
                    and handle.data.traj_frames_norm is not None
                    and handle.data.traj_frames_norm.shape[0] > 0
                ):
                    pos_idx = min(idx, handle.data.trajectory_norm.shape[0] - 1)
                    handle.traj_frame_canonical.position = handle.data.trajectory_norm[pos_idx]
                    frame_idx = min(idx, handle.data.traj_frames_norm.shape[0] - 1)
                    traj_rot = handle.data.traj_frames_norm[frame_idx]
                    handle.traj_frame_canonical.wxyz = viser_tf.SO3.from_matrix(traj_rot).wxyz

            R_cam_canonical = None
            t_cam_canonical = None
            if (
                handle.data.camera_poses_norm is not None
                and handle.data.camera_positions_norm is not None
                and handle.data.camera_poses_norm.shape[0] > 0
                and handle.data.camera_positions_norm.shape[0] > 0
            ):
                pose_idx = min(idx, handle.data.camera_poses_norm.shape[0] - 1)
                pos_idx = min(idx, handle.data.camera_positions_norm.shape[0] - 1)
                pose_matrix = handle.data.camera_poses_norm[pose_idx]
                R_cam_canonical = pose_matrix[:3, :3]
                t_cam_canonical = handle.data.camera_positions_norm[pos_idx]

            show_cam = (
                show_canonical_any
                and handle.show_cameras.value
                and R_cam_canonical is not None
                and t_cam_canonical is not None
            )
            if handle.cam_path_canonical is not None:
                handle.cam_path_canonical.visible = show_cam
            if handle.cam_frustum_canonical is not None:
                handle.cam_frustum_canonical.visible = show_cam
                if show_cam:
                    handle.cam_frustum_canonical.position = t_cam_canonical
                    quat_norm = viser_tf.SO3.from_matrix(R_cam_canonical).wxyz
                    handle.cam_frustum_canonical.wxyz = quat_norm
            if handle.cam_frame_canonical is not None:
                handle.cam_frame_canonical.visible = show_cam
                if show_cam:
                    handle.cam_frame_canonical.position = t_cam_canonical
                    handle.cam_frame_canonical.wxyz = viser_tf.SO3.from_matrix(R_cam_canonical).wxyz

            # World space updates.
            show_world_mesh = handle.show_world_mesh.value
            show_world_traj = handle.show_world_traj.value
            show_world_any = show_world_mesh or show_world_traj
            handle.mesh_world.visible = show_world_mesh
            handle.traj_world.visible = show_world_traj
            if show_world_mesh:
                handle.mesh_world.vertices = handle.data.vertices_raw[idx]
            if handle.root_frame_world is not None:
                handle.root_frame_world.visible = show_world_mesh
                if show_world_mesh:
                    pos_idx = min(idx, handle.data.trajectory_raw.shape[0] - 1)
                    handle.root_frame_world.position = handle.data.trajectory_raw[pos_idx]
                    if (
                        handle.data.root_rotations_raw is not None
                        and handle.data.root_rotations_raw.shape[0] > 0
                    ):
                        rot_idx = min(idx, handle.data.root_rotations_raw.shape[0] - 1)
                        root_rot = handle.data.root_rotations_raw[rot_idx]
                        handle.root_frame_world.wxyz = viser_tf.SO3.from_matrix(root_rot).wxyz
            if handle.traj_frame_world is not None:
                handle.traj_frame_world.visible = show_world_traj
                if (
                    show_world_traj
                    and handle.data.traj_frames_raw is not None
                    and handle.data.traj_frames_raw.shape[0] > 0
                ):
                    pos_idx = min(idx, handle.data.trajectory_raw.shape[0] - 1)
                    handle.traj_frame_world.position = handle.data.trajectory_raw[pos_idx]
                    frame_idx = min(idx, handle.data.traj_frames_raw.shape[0] - 1)
                    traj_rot = handle.data.traj_frames_raw[frame_idx]
                    handle.traj_frame_world.wxyz = viser_tf.SO3.from_matrix(traj_rot).wxyz

            R_cam_world = None
            t_cam_world = None
            if (
                handle.data.camera_poses_raw is not None
                and handle.data.camera_positions_raw is not None
                and handle.data.camera_poses_raw.shape[0] > 0
                and handle.data.camera_positions_raw.shape[0] > 0
            ):
                pose_idx = min(idx, handle.data.camera_poses_raw.shape[0] - 1)
                pos_idx = min(idx, handle.data.camera_positions_raw.shape[0] - 1)
                pose_matrix = handle.data.camera_poses_raw[pose_idx]
                R_cam_world = pose_matrix[:3, :3]
                t_cam_world = handle.data.camera_positions_raw[pos_idx]
                cam_rot_norm = _normalize_matrix(R_cam_world)

            show_world_cam = (
                show_world_any
                and handle.show_cameras.value
                and R_cam_world is not None
                and t_cam_world is not None
            )
            if handle.cam_path_world is not None:
                handle.cam_path_world.visible = show_world_cam
            if handle.cam_frustum_world is not None:
                handle.cam_frustum_world.visible = show_world_cam
                if show_world_cam:
                    handle.cam_frustum_world.position = t_cam_world
                    quat_world = viser_tf.SO3.from_matrix(R_cam_world).wxyz
                    handle.cam_frustum_world.wxyz = quat_world
            if handle.cam_frame_world is not None:
                handle.cam_frame_world.visible = show_world_cam
                if show_world_cam:
                    handle.cam_frame_world.position = t_cam_world
                    handle.cam_frame_world.wxyz = viser_tf.SO3.from_matrix(R_cam_world).wxyz

            # Update axis display with normalized matrices
            handle.smpl_axis_text.content = _axes_markdown("SMPL (world)", smpl_rot_norm, smpl_pos)
            handle.traj_axis_text.content = _axes_markdown("Trajectory (world)", traj_rot_norm, traj_pos)
            handle.cam_axis_text.content = _axes_markdown("Camera (world)", cam_rot_norm, t_cam_world)

    def on_slider(event) -> None:
        update_frame(int(event.target.value))

    frame_slider.on_update(on_slider)

    def make_toggle_cb():
        def _cb(_event) -> None:
            update_frame(int(frame_slider.value))

        return _cb

    for handle in sample_handles:
        if handle.show_canonical_mesh is not None:
            handle.show_canonical_mesh.on_update(make_toggle_cb())
        if handle.show_canonical_traj is not None:
            handle.show_canonical_traj.on_update(make_toggle_cb())
        handle.show_world_mesh.on_update(make_toggle_cb())
        handle.show_world_traj.on_update(make_toggle_cb())
        handle.show_cameras.on_update(make_toggle_cb())

    def toggle_play(_event) -> None:
        playback_state["running"] = not playback_state["running"]
        play_status.value = "Running" if playback_state["running"] else "Paused"

    play_button.on_click(toggle_play)

    def playback_loop() -> None:
        last = time.time()
        while True:
            if not playback_state["running"]:
                time.sleep(0.02)
                last = time.time()
                continue

            target = 1.0 / float(fps_slider.value)
            now = time.time()
            if now - last < target:
                time.sleep(0.005)
                continue
            last = now

            next_frame = (int(frame_slider.value) + 1) % max_frames
            frame_slider.value = next_frame

    thread = threading.Thread(target=playback_loop, daemon=True)
    thread.start()

    update_frame(0)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        playback_state["running"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively visualize SMPL-X + camera trajectories with viser"
    )
    parser.add_argument(
        "--sample_id",
        dest="sample_ids",
        action="append",
        required=True,
        help="Sample identifier (repeat this flag to load multiple samples)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser server host")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [prepare_sample(sample_id) for sample_id in args.sample_ids]
    launch_viewer(
        datasets,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()

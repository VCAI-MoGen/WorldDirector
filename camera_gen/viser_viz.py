
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


def normalize_camera_poses(camera_poses: np.ndarray, R0: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Match TrajectoryDataset normalization for cam-to-world coordinates."""
    T = camera_poses.shape[0]
    camera_poses_norm = np.zeros_like(camera_poses)

    R0_inv = R0.T
    for i in range(T):
        R_cam = camera_poses[i, :3, :3]
        t_cam = camera_poses[i, :3, 3]

        R_cam_norm = R0_inv @ R_cam
        t_cam_norm = (t_cam - p0) @ R0_inv.T

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
    vertices_norm: np.ndarray
    joints_norm: np.ndarray
    trajectory_raw: np.ndarray
    trajectory_norm: np.ndarray
    faces: np.ndarray
    root_position: np.ndarray
    root_rotation: np.ndarray
    camera_poses_raw: Optional[np.ndarray]
    camera_poses_norm: Optional[np.ndarray]
    camera_positions_raw: Optional[np.ndarray]
    camera_positions_norm: Optional[np.ndarray]


@dataclass(slots=True)
class SampleHandles:
    data: TrajectoryData
    frame_count: int
    offset_canonical: np.ndarray
    offset_world: np.ndarray
    show_canonical: viser.GuiCheckboxHandle
    show_world: viser.GuiCheckboxHandle
    show_cameras: viser.GuiCheckboxHandle
    mesh_canonical: viser.MeshHandle
    mesh_world: viser.MeshHandle
    traj_canonical: viser.LineSegmentsHandle
    traj_world: viser.LineSegmentsHandle
    cam_path_canonical: Optional[viser.LineSegmentsHandle]
    cam_path_world: Optional[viser.LineSegmentsHandle]
    cam_frustum_canonical: Optional[viser.CameraFrustumHandle]
    cam_frustum_world: Optional[viser.CameraFrustumHandle]


def prepare_sample(sample_id: str) -> TrajectoryData:
    npy_path = f"{TRAJ_NPY_PATH}/{sample_id}.npy"
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing character trajectory: {npy_path}")

    trajectory_points_raw = np.load(npy_path).astype(np.float32)
    verts_raw, joints_raw, faces, go_mat_np, _ = _load_smpl_outputs(sample_id)


    root_idx = 0
    R0 = go_mat_np[0]
    p0 = joints_raw[0, root_idx]
    verts_norm = (verts_raw - p0) @ R0
    joints_norm = (joints_raw - p0) @ R0
    trajectory_norm = (trajectory_points_raw - p0) @ R0

    camera_path = f"{CAMERA_POSE_DIR}/{sample_id}.txt"
    camera_poses_raw: Optional[np.ndarray] = None
    camera_positions_raw: Optional[np.ndarray] = None
    camera_poses_norm: Optional[np.ndarray] = None
    camera_positions_norm: Optional[np.ndarray] = None

    if os.path.exists(camera_path):
        trajectory = read_kitti_poses_file(camera_path)
        camera_poses_raw = np.array(trajectory.poses_se3).astype(np.float32)
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
        root_rotation=R0.astype(np.float32),
        camera_poses_raw=camera_poses_raw,
        camera_poses_norm=camera_poses_norm,
        camera_positions_raw=camera_positions_raw,
        camera_positions_norm=camera_positions_norm,
    )


def _make_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    seg = np.stack([points[:-1], points[1:]], axis=1)
    return seg.astype(np.float32)

#viser viewer
def launch_viewer(
    datasets: Sequence[TrajectoryData],
    host: str,
    port: int,
    canonical_offset: float,
    world_offset: float,
) -> None:
    if len(datasets) == 0:
        raise ValueError("At least one sample is required")

    max_frames = max(data.vertices_norm.shape[0] for data in datasets)
    label = ", ".join(data.sample_id for data in datasets)
    server = viser.ViserServer(host=host, port=port, label=f"Samples: {label}")

    sample_summary = "\n".join(
        f"- `{data.sample_id}`: {data.vertices_norm.shape[0]} frames"
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
        offset_canonical = np.array([idx * canonical_offset, 0.0, 0.0], dtype=np.float32)
        offset_world = np.array([idx * world_offset, 0.0, 0.0], dtype=np.float32)
        initial_show_canonical = idx == 0
        initial_show_world = False
        initial_show_cameras = data.camera_positions_raw is not None and idx == 0

        with server.gui.add_folder(
            f"Sample {idx + 1}: {data.sample_id}", expand_by_default=idx == 0
        ):
            show_canonical = server.gui.add_checkbox(
                "Show canonical", initial_value=initial_show_canonical
            )
            show_world = server.gui.add_checkbox(
                "Show world", initial_value=initial_show_world
            )
            show_cameras = server.gui.add_checkbox(
                "Show cameras", initial_value=initial_show_cameras
            )

        canonical_vertices0 = data.vertices_norm[0] + offset_canonical
        world_vertices0 = data.vertices_raw[0] + offset_world

        mesh_canonical = server.scene.add_mesh_simple(
            f"/samples/{idx}/canonical/mesh",
            vertices=canonical_vertices0,
            faces=data.faces,
            color=(90, 200, 255),
            flat_shading=False,
            visible=initial_show_canonical,
        )
        mesh_world = server.scene.add_mesh_simple(
            f"/samples/{idx}/world/mesh",
            vertices=world_vertices0,
            faces=data.faces,
            color=(255, 170, 90),
            opacity=0.55,
            visible=initial_show_world,
        )

        traj_canonical = server.scene.add_line_segments(
            f"/samples/{idx}/canonical/trajectory",
            points=_make_segments(data.trajectory_norm + offset_canonical),
            colors=(80, 255, 120),
            line_width=2.0,
            visible=initial_show_canonical,
        )
        traj_world = server.scene.add_line_segments(
            f"/samples/{idx}/world/trajectory",
            points=_make_segments(data.trajectory_raw + offset_world),
            colors=(255, 190, 80),
            line_width=2.0,
            visible=initial_show_world,
        )

        cam_path_canonical = None
        cam_path_world = None
        cam_frustum_canonical = None
        cam_frustum_world = None

        if data.camera_positions_norm is not None:
            cam_path_canonical = server.scene.add_line_segments(
                f"/samples/{idx}/canonical/camera_path",
                points=_make_segments(data.camera_positions_norm + offset_canonical),
                colors=(80, 180, 255),
                line_width=2.0,
                visible=initial_show_canonical and initial_show_cameras,
            )
        if data.camera_positions_raw is not None:
            cam_path_world = server.scene.add_line_segments(
                f"/samples/{idx}/world/camera_path",
                points=_make_segments(data.camera_positions_raw + offset_world),
                colors=(255, 255, 120),
                line_width=1.5,
                visible=initial_show_world and initial_show_cameras,
            )
        if data.camera_poses_norm is not None and data.camera_positions_norm is not None:
            cam_frustum_canonical = server.scene.add_camera_frustum(
                f"/samples/{idx}/canonical/camera",
                fov=np.deg2rad(60.0),
                aspect=16 / 9,
                position=data.camera_positions_norm[0] + offset_canonical,
                visible=initial_show_canonical and initial_show_cameras,
                color=(80, 180, 255),
            )
        if data.camera_poses_raw is not None and data.camera_positions_raw is not None:
            cam_frustum_world = server.scene.add_camera_frustum(
                f"/samples/{idx}/world/camera",
                fov=np.deg2rad(60.0),
                aspect=16 / 9,
                position=data.camera_positions_raw[0] + offset_world,
                visible=initial_show_world and initial_show_cameras,
                color=(255, 180, 80),
            )

        sample_handles.append(
            SampleHandles(
                data=data,
                frame_count=data.vertices_norm.shape[0],
                offset_canonical=offset_canonical,
                offset_world=offset_world,
                show_canonical=show_canonical,
                show_world=show_world,
                show_cameras=show_cameras,
                mesh_canonical=mesh_canonical,
                mesh_world=mesh_world,
                traj_canonical=traj_canonical,
                traj_world=traj_world,
                cam_path_canonical=cam_path_canonical,
                cam_path_world=cam_path_world,
                cam_frustum_canonical=cam_frustum_canonical,
                cam_frustum_world=cam_frustum_world,
            )
        )

    playback_state = {"running": False}

    def update_frame(frame: int) -> None:
        for handle in sample_handles:
            idx = min(frame, handle.frame_count - 1)

            # Canonical space updates.
            show_canonical = handle.show_canonical.value
            handle.mesh_canonical.visible = show_canonical
            handle.traj_canonical.visible = show_canonical
            if show_canonical:
                handle.mesh_canonical.vertices = (
                    handle.data.vertices_norm[idx] + handle.offset_canonical
                )

            pose_norm = None
            pos_norm = None
            if handle.data.camera_poses_norm is not None and handle.data.camera_positions_norm is not None:
                pose_norm = handle.data.camera_poses_norm[min(idx, handle.data.camera_poses_norm.shape[0] - 1)]
                pos_norm = handle.data.camera_positions_norm[min(idx, handle.data.camera_positions_norm.shape[0] - 1)]

            show_cam = (
                show_canonical
                and handle.show_cameras.value
                and pose_norm is not None
                and pos_norm is not None
            )
            if handle.cam_path_canonical is not None:
                handle.cam_path_canonical.visible = show_cam
            if handle.cam_frustum_canonical is not None:
                handle.cam_frustum_canonical.visible = show_cam
                if show_cam:
                    handle.cam_frustum_canonical.position = pos_norm + handle.offset_canonical

            # World space updates.
            show_world = handle.show_world.value
            handle.mesh_world.visible = show_world
            handle.traj_world.visible = show_world
            if show_world:
                handle.mesh_world.vertices = (
                    handle.data.vertices_raw[idx] + handle.offset_world
                )

            pose_raw = None
            pos_raw = None
            if handle.data.camera_poses_raw is not None and handle.data.camera_positions_raw is not None:
                pose_raw = handle.data.camera_poses_raw[min(idx, handle.data.camera_poses_raw.shape[0] - 1)]
                pos_raw = handle.data.camera_positions_raw[min(idx, handle.data.camera_positions_raw.shape[0] - 1)]

            show_world_cam = (
                show_world
                and handle.show_cameras.value
                and pose_raw is not None
                and pos_raw is not None
            )
            if handle.cam_path_world is not None:
                handle.cam_path_world.visible = show_world_cam
            if handle.cam_frustum_world is not None:
                handle.cam_frustum_world.visible = show_world_cam
                if show_world_cam:
                    handle.cam_frustum_world.position = pos_raw + handle.offset_world

    def on_slider(event) -> None:
        update_frame(int(event.target.value))

    frame_slider.on_update(on_slider)

    def make_toggle_cb():
        def _cb(_event) -> None:
            update_frame(int(frame_slider.value))

        return _cb

    for handle in sample_handles:
        handle.show_canonical.on_update(make_toggle_cb())
        handle.show_world.on_update(make_toggle_cb())
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
    parser.add_argument(
        "--canonical-offset",
        type=float,
        default=0.0,
        help="If you want to compare the normalization of two different sequences put this option as 0.0",
    )
    parser.add_argument(
        "--world-offset",
        type=float,
        default=0.0,
        help="Offset applied along +X between samples in world space",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [prepare_sample(sample_id) for sample_id in args.sample_ids]
    launch_viewer(
        datasets,
        host=args.host,
        port=args.port,
        canonical_offset=args.canonical_offset,
        world_offset=args.world_offset,
    )


if __name__ == "__main__":
    main()

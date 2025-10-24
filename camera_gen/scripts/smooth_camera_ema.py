"""
Apply exponential moving average smoothing to camera rotations.

The script expects KITTI-style trajectory files (one 3x4 matrix per line) inside
the input directory. By default the smoothed trajectories are written to a
parallel output directory named `<input_dir>_ema`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMA smoothing for camera rotations.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("et-data/traj"),
        help="Directory containing KITTI-style trajectory txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for smoothed trajectories (default: `<input_dir>_ema`).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="EMA blending factor (0 < alpha <= 1, smaller => stronger smoothing).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for trajectory files inside input_dir.",
    )
    parser.add_argument(
        "--smooth_translation",
        action="store_true",
        help="Also apply EMA smoothing to the translation component.",
    )
    return parser.parse_args()


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    m = mat.astype(np.float64, copy=False)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def smooth_quaternions(quaternions: np.ndarray, alpha: float) -> np.ndarray:
    """EMA smoothing over a sequence of quaternions."""
    smoothed = np.empty_like(quaternions)
    smoothed[0] = quaternions[0] / np.linalg.norm(quaternions[0])

    for idx in range(1, len(quaternions)):
        prev = smoothed[idx - 1]
        current = quaternions[idx]
        smoothed[idx] = slerp(prev, current, alpha)

    return smoothed


def load_trajectory(path: Path) -> np.ndarray:
    """Load a KITTI pose file into an array of shape (N, 3, 4)."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[-1] != 12:
        raise ValueError(f"{path}: expected 12 values per row, got shape {data.shape}")
    return data.reshape(-1, 3, 4)


def save_trajectory(path: Path, matrices: np.ndarray) -> None:
    N = matrices.shape[0]
    mat = np.zeros((N,4,4), dtype=np.float32)
    mat[:,:3,:4] = matrices
    mat[:,3,3] = 1.0

    np.save(path.with_suffix(".npy"),mat)

def ema_smooth_trajectory(
    poses: np.ndarray, alpha: float, smooth_translation: bool
) -> np.ndarray:
    """Smooth camera rotations (and optionally translations) with EMA."""
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3].copy()

    quats = np.stack([matrix_to_quaternion(R) for R in rotations], axis=0)
    smoothed_quats = smooth_quaternions(quats, alpha)
    smoothed_rotations = np.stack(
        [quaternion_to_matrix(q).astype(np.float64) for q in smoothed_quats], axis=0
    )

    if smooth_translation and len(translations) > 1:
        ema = translations[0].copy()
        for idx in range(1, translations.shape[0]):
            ema = (1.0 - alpha) * ema + alpha * translations[idx]
            translations[idx] = ema

    result = poses.copy()
    result[:, :3, :3] = smoothed_rotations
    result[:, :3, 3] = translations
    return result


def iter_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.rglob(pattern))


def main() -> None:
    args = parse_args()

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = args.output_dir or (input_dir.parent / f"{input_dir.name}_ema_2")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(input_dir, args.pattern))
    if not files:
        raise RuntimeError(f"No files matching '{args.pattern}' found in {input_dir}")

    print(
        f"[smooth_camera_ema] smoothing {len(files)} files from '{input_dir}' "
        f"into '{output_dir}' (alpha={args.alpha}, smooth_translation={args.smooth_translation})."
    )

    for src_path in files:
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        poses = load_trajectory(src_path)
        if poses.shape[0] < 2:
            save_trajectory(dst_path, poses)
            print(f"[smooth_camera_ema] copied (single frame) {rel_path}")
            continue

        smoothed = ema_smooth_trajectory(poses, args.alpha, args.smooth_translation)
        save_trajectory(dst_path, smoothed)
        print(f"[smooth_camera_ema] wrote {smoothed.shape[0]} frames -> {rel_path}")


if __name__ == "__main__":
    main()

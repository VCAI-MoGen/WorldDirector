
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import torch
from evo.tools.file_interface import read_kitti_poses_file


DEFAULT_DTYPE = torch.float32


@dataclass
class Stats:
    """Running first and second order moments for vector features."""

    dim: int
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = DEFAULT_DTYPE

    def __post_init__(self) -> None:
        self.sum = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.sum_sq = torch.zeros_like(self.sum)
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        if values.ndim == 1:
            values = values.unsqueeze(0)
        values = values.to(dtype=self.dtype, device=self.device)
        self.sum += values.sum(dim=0)
        self.sum_sq += (values * values).sum(dim=0)
        self.count += int(values.shape[0])

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.sum)
        return self.sum / self.count

    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.sum)
        mean = self.mean()
        var = self.sum_sq / self.count - mean * mean
        var = torch.clamp(var, min=0.0)
        return torch.sqrt(var)


def load_split_names(dataset_dir: Path, split_files: Sequence[str]) -> list[str]:
    names: set[str] = set()
    for split_name in split_files:
        split_path = dataset_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if entry:
                    names.add(entry)
    return sorted(names)


def load_char_sequence(dataset_dir: Path, sample: str) -> torch.Tensor:
    path = dataset_dir / "char" / f"{sample}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing character centres: {path}")
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Unexpected char shape {data.shape} for {sample}")
    return torch.from_numpy(data).to(DEFAULT_DTYPE)


def load_smpl_root(dataset_dir: Path, sample: str) -> torch.Tensor:
    path = dataset_dir / "smplh" / f"{sample}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing SMPL parameters: {path}")
    with path.open("rb") as f:
        params = pickle.load(f, encoding="latin1")
    root_pose = params.get("global_orient")
    if isinstance(root_pose, torch.Tensor):
        tensor = root_pose.detach().cpu().to(DEFAULT_DTYPE)
    else:
        tensor = torch.as_tensor(root_pose, dtype=DEFAULT_DTYPE)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
 
    return tensor


def load_camera_trajectory(dataset_dir: Path, sample: str) -> torch.Tensor:
    txt_path = dataset_dir / "traj" / f"{sample}.txt"
    if txt_path.exists():
        trajectory = read_kitti_poses_file(txt_path)
        matrices = np.asarray(trajectory.poses_se3, dtype=np.float32)

    return torch.from_numpy(matrices).to(DEFAULT_DTYPE)


def compute_aligned_translations(
    matrices: torch.Tensor,
    char_first_pos: torch.Tensor,
    R0_inv: torch.Tensor,
    align_first_frame: bool,
) -> torch.Tensor:
    translations = matrices[:, :3, 3].to(DEFAULT_DTYPE)
    if align_first_frame:
        char_first = char_first_pos.to(dtype=DEFAULT_DTYPE, device=translations.device)
        R0_inv = R0_inv.to(dtype=DEFAULT_DTYPE, device=translations.device)
        translations = (translations - char_first) @ R0_inv.T
    return translations


def compute_char_features(
    char_raw: torch.Tensor,
    R0_inv: torch.Tensor,
    align_first_frame: bool,
    velocity: bool,
) -> torch.Tensor:
    char_proc = char_raw.to(dtype=DEFAULT_DTYPE)
    char_device = char_proc.device
    if align_first_frame:
        R0_proc = R0_inv.to(dtype=DEFAULT_DTYPE, device=char_device)
        char_aligned = (char_proc - char_proc[0:1]) @ R0_proc.T
    else:
        char_aligned = char_proc

    if velocity:
        if char_aligned.size(0) <= 1:
            velocities = torch.zeros((0, 3), dtype=DEFAULT_DTYPE, device=char_device)
        else:
            velocities = char_aligned[1:] - char_aligned[:-1]
        first_frame = char_raw[0:1].to(dtype=DEFAULT_DTYPE, device=char_device)
        return torch.cat([first_frame, velocities], dim=0)
    return char_aligned


def moments_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.tolist()]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./et-data"),
        help="Root directory containing traj/, char/, smplh/ folders",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["mixed_train_split.txt", "mixed_test_split.txt"],
        help="Split file names relative to dataset dir",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="0300",
        help="Identifier written into the YAML output",
    )
    parser.add_argument(
        "--num-cams",
        type=int,
        default=300,
        help="Number of frames to pad to; defaults to max observed length",
    )
    parser.add_argument(
        "--num-interframes",
        type=int,
        default=0,
        help="Interframe sampling count used in downstream configs",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_true",
        help="Disable velocity differencing for trajectory and char stats",
    )
    parser.add_argument(
        "--no-align-first-frame",
        action="store_true",
        help="Skip first-frame alignment when computing stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the YAML snippet",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample warnings",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    align_first_frame = not args.no_align_first_frame
    velocity = not args.no_velocity

    try:
        samples = load_split_names(dataset_dir, args.splits)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[error] {exc}\n")
        return 1

    if not samples:
        sys.stderr.write("[error] No samples found in provided splits\n")
        return 1

    shift_stats = Stats(3)
    norm_stats = Stats(3)
    char_stats = Stats(3)

    max_frames = 0
    processed = 0

    for sample in samples:
        try:
            raw_char = load_char_sequence(dataset_dir, sample)
            matrices = load_camera_trajectory(dataset_dir, sample)
            if align_first_frame:
                root_poses = load_smpl_root(dataset_dir, sample)
                R0_inv = root_poses[0].T
            else:
                R0_inv = torch.eye(3, dtype=DEFAULT_DTYPE)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            if not args.quiet:
                sys.stderr.write(f"[warn] Skipping {sample}: {exc}\n")
            continue

        if matrices.size(0) == 0 or raw_char.size(0) == 0:
            if not args.quiet:
                sys.stderr.write(f"[warn] Skipping {sample}: empty sequence\n")
            continue

        target_frames = args.num_cams if args.num_cams is not None else matrices.size(0)
        use_len = min(matrices.size(0), target_frames)
        matrices_use = matrices[:use_len]

        if align_first_frame:
            char_first = raw_char[0:1].to(dtype=DEFAULT_DTYPE, device=matrices_use.device)
        else:
            char_first = torch.zeros((1, 3), dtype=DEFAULT_DTYPE, device=matrices_use.device)

        translations = compute_aligned_translations(
            matrices_use, char_first, R0_inv, align_first_frame
        )
        seq_len = translations.size(0)
        max_frames = max(max_frames, target_frames)

        shift_stats.update(translations[0])

        if velocity:
            if seq_len > 1:
                diffs = translations[1:] - translations[:-1]
            else:
                diffs = torch.zeros((0, 3), dtype=DEFAULT_DTYPE, device=translations.device)
        else:
            diffs = translations
        norm_stats.update(diffs)

        char_target = args.num_cams if args.num_cams is not None else raw_char.size(0)
        char_len = min(raw_char.size(0), char_target)
        if char_len > 0:
            char_used = raw_char[:char_len]
            char_features = compute_char_features(
                char_used, R0_inv, align_first_frame, velocity
            )
            char_stats.update(char_features)

        processed += 1

    if processed == 0:
        sys.stderr.write("[error] No samples processed successfully\n")
        return 1

    num_cams = args.num_cams if args.num_cams is not None else max_frames
    num_total_frames = args.num_interframes * (num_cams - 1) + num_cams

    output = {
        "name": args.name,
        "num_interframes": args.num_interframes,
        "num_cams": num_cams,
        "num_total_frames": num_total_frames,
        "norm_mean": moments_to_list(norm_stats.mean()),
        "norm_std": moments_to_list(norm_stats.std()),
        "shift_mean": moments_to_list(shift_stats.mean()),
        "shift_std": moments_to_list(shift_stats.std()),
        "norm_mean_h": moments_to_list(char_stats.mean()),
        "norm_std_h": moments_to_list(char_stats.std()),
        "velocity": velocity,
    }

    yaml_lines = [f"name: '{output['name']}'"]
    yaml_lines.append(f"num_interframes: {output['num_interframes']}")
    yaml_lines.append(f"num_cams: {output['num_cams']}")
    yaml_lines.append(
        "num_total_frames: ${eval:'${dataset.standardization.num_interframes} * "
        "(${dataset.standardization.num_cams} - 1) + ${dataset.standardization.num_cams} '}"
    )

    def fmt_list(values: Iterable[float]) -> str:
        formatted = []
        for v in values:
            val = float(v)
            if abs(val) < 1e-3:
                formatted.append(f"{val:.8e}")   # 지수 표기
            else:
                formatted.append(f"{val:.8f}")   # 일반 소수
        return f"[{', '.join(formatted)}]"

    yaml_lines.append(f"\nnorm_mean: {fmt_list(output['norm_mean'])}")
    yaml_lines.append(f"norm_std: {fmt_list(output['norm_std'])}")
    yaml_lines.append(f"\nshift_mean: {fmt_list(output['shift_mean'])}")
    yaml_lines.append(f"shift_std: {fmt_list(output['shift_std'])}")
    yaml_lines.append(f"\nnorm_mean_h: {fmt_list(output['norm_mean_h'])}")
    yaml_lines.append(f"norm_std_h: {fmt_list(output['norm_std_h'])}")
    yaml_lines.append(f"\nvelocity: {str(output['velocity']).lower()}")

    result = "\n".join(yaml_lines) + "\n"

    if args.output is not None:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

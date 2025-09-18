
import argparse, pickle, numpy as np
from collections import Counter
from pathlib import Path
import torch
ROT_KEYS = ["body_pose", "left_hand_pose", "right_hand_pose", "global_orient"]
VEC_KEYS = ["betas", "transl"]
ALL_KEYS = ROT_KEYS + VEC_KEYS


def describe_shape(arr) -> str:
    if arr is None:
        return "None"
    return "x".join(str(x) for x in arr.shape)

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def is_bad_pkl(pkl_path: Path) -> tuple[bool, str]:
    try:
        with open(pkl_path, "rb") as f:
            sp = pickle.load(f, encoding="latin1")

        frame_lengths: dict[str, int] = {}
        arrays: dict[str, np.ndarray] = {}
        issues: list[str] = []

        for k in ALL_KEYS:
            if k not in sp:
                issues.append(f"missing {k}")
                continue

            arr = to_numpy_safe(sp[k])

            if arr is None:
                issues.append(f"{k}=None")
                continue

            if arr.size == 0:
                issues.append(f"{k} empty array")
                continue

            if not np.isfinite(arr).all():
                issues.append(f"{k} contains non-finite")
                continue

            if k in ROT_KEYS:
                if arr.ndim not in (3, 4) or arr.shape[-2:] != (3, 3):
                    issues.append(f"{k} bad shape {describe_shape(arr)}")
                    continue
                frame_lengths[k] = arr.shape[0]

            elif k == "transl":
                if arr.ndim != 2 or arr.shape[1] != 3:
                    issues.append(f"{k} bad shape {describe_shape(arr)}")
                    continue
                frame_lengths[k] = arr.shape[0]

            elif k == "betas":
                if arr.ndim == 1:
                    frame_lengths[k] = 1
                elif arr.ndim == 2:
                    frame_lengths[k] = arr.shape[0]
                else:
                    issues.append(f"{k} bad shape {describe_shape(arr)}")
                    continue

            arrays[k] = arr

        # Require at least one frame across time-dependent keys
        transl_len = frame_lengths.get("transl", 0)
        if transl_len < 1:
            issues.append("transl has no frames")

        # Ensure rotations provide at least the first frame
        for rot_key in ROT_KEYS:
            if frame_lengths.get(rot_key, 0) < 1:
                issues.append(f"{rot_key} has no frames")

        # Check consistency across rotation sequences when more than one frame is present
        expected_len = max(frame_lengths.get(k, 0) for k in ROT_KEYS)
        if expected_len > 1:
            mismatched = {
                k: frame_lengths[k]
                for k in ROT_KEYS
                if frame_lengths.get(k, 0) not in (1, expected_len)
            }
            if mismatched:
                mismatch_str = ", ".join(f"{k}:{v}" for k, v in mismatched.items())
                issues.append(f"inconsistent frame counts ({mismatch_str})")

        # Verify root reference can be established
        go_np = arrays.get("global_orient")
        transl_np = arrays.get("transl")

        if go_np is None:
            issues.append("missing usable global_orient data")
        if transl_np is None:
            issues.append("missing usable transl data")

        if go_np is not None and transl_np is not None:
            if go_np.ndim == 4 and go_np.shape[1] == 1:
                go_np = go_np[:, 0]
            try:
                R0 = np.asarray(go_np[0], dtype=np.float64)
            except Exception:
                issues.append("unable to extract first global_orient frame")
            else:
                try:
                    t0 = np.asarray(transl_np[0], dtype=np.float64)
                except Exception:
                    issues.append("unable to extract first transl frame")
                else:
                    if not (np.isfinite(R0).all() and np.isfinite(t0).all()):
                        issues.append("non-finite R0/t0 in first frame")
                    else:
                        should_be_I = R0 @ R0.T
                        if not np.allclose(should_be_I, np.eye(3), atol=1e-3):
                            max_err = np.abs(should_be_I - np.eye(3)).max()
                            issues.append(f"R0 not orthonormal (max_err={max_err:.2e})")
                        det = np.linalg.det(R0)
                        if not np.allclose(det, 1.0, atol=1e-3):
                            issues.append(f"det(R0)={det:.4f}")

        if issues:
            return True, "; ".join(issues)

        return False, ""
    except Exception as e:
        return True, f"exception {type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplh_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--pattern", default="*.pkl")
    ap.add_argument("--out_good", default="good.txt")
    ap.add_argument("--out_bad", default="bad.txt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--expected_list",
        nargs="*",
        type=Path,
        help="Optional text file(s) listing sequence stems that must be present and valid.",
    )
    args = ap.parse_args()

    base = Path(args.smplh_dir)
    files = sorted(base.rglob(args.pattern) if args.recursive else base.glob(args.pattern))
    if args.limit:
        files = files[:args.limit]

    stem_to_paths: dict[str, list[Path]] = {}
    for p in files:
        stem_to_paths.setdefault(p.stem, []).append(p)

    good, bad = [], []
    issue_counter: Counter[str] = Counter()

    def record_result(stem: str, path: Path):
        bad_flag, why = is_bad_pkl(path)
        if bad_flag:
            bad.append(f"{stem}\t{why}\t{path}")
            for entry in why.split(";"):
                cleaned = entry.strip()
                if cleaned:
                    issue_counter[cleaned] += 1
        else:
            good.append(stem)

    if args.expected_list:
        expected_stems = []
        for list_path in args.expected_list:
            expected_stems.extend(
                line.strip() for line in list_path.read_text().splitlines() if line.strip()
            )

        seen = set()
        for stem in expected_stems:
            if stem in seen:
                continue
            seen.add(stem)
            paths = stem_to_paths.get(stem)
            if not paths:
                bad.append(f"{stem}\tmissing file\t<none>")
                issue_counter["missing file"] += 1
                continue
            if len(paths) > 1:
                msg = f"duplicate files ({len(paths)})"
                bad.append(f"{stem}\t{msg}\t{paths[0].parent}")
                issue_counter[msg] += 1
                continue
            record_result(stem, paths[0])

        # Optionally warn about extra files not in expected list
        for stem, paths in stem_to_paths.items():
            if stem not in seen and len(paths) == 1:
                record_result(stem, paths[0])
            elif stem not in seen:
                msg = f"duplicate files ({len(paths)})"
                bad.append(f"{stem}\t{msg}\t{paths[0].parent}")
                issue_counter[msg] += 1
    else:
        for stem, paths in stem_to_paths.items():
            if len(paths) > 1:
                msg = f"duplicate files ({len(paths)})"
                bad.append(f"{stem}\t{msg}\t{paths[0].parent}")
                issue_counter[msg] += 1
                continue
            record_result(stem, paths[0])

    Path(args.out_good).write_text("\n".join(good) + ("\n" if good else ""))
    Path(args.out_bad).write_text("\n".join(bad) + ("\n" if bad else ""))

    print(f"scanned={len(files)}  good={len(good)}  bad={len(bad)}")
    if bad[:10]:
        print("bad samples (preview):")
        for line in bad[:10]:
            print("  ", line)
    if issue_counter:
        print("bad reason counts (top 15):")
        for reason, count in issue_counter.most_common(15):
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()

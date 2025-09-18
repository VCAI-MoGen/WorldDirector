from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from smplx import SMPLX
    from pytorch3d.pytorch3d.transforms import matrix_to_axis_angle
    from pytorch3d.pytorch3d.transforms import matrix_to_rotation_6d
    _SMPLX_AVAILABLE = True
except Exception:
    _SMPLX_AVAILABLE = False


# ------------------------------------------------------------------------------------- #


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _project_to_so3(mat: torch.Tensor) -> torch.Tensor:
    shape = mat.shape
    mat = mat.reshape(-1, 3, 3)
    U, _, Vh = torch.linalg.svd(mat)
    proj = U @ Vh
    det = torch.det(proj)
    neg_mask = det < 0
    if neg_mask.any():
        Vh[neg_mask, -1] *= -1
        proj = U @ Vh
    return proj.reshape(shape)

def _summarize(name: str, arr: np.ndarray) -> str:
    try:
        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
    except Exception:
        min_val = max_val = "n/a"
    return f"{name}: shape={arr.shape} min={min_val} max={max_val}"


class CharacterDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        standardize: bool,
        num_feats: int,
        num_cams: int,
        sequential: bool,
        num_vertices: int,
        num_faces: int,
        load_vertices: bool,
        **kwargs,
    ):
        super().__init__()
        self.modality = "char"
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / self.name
        self.smpl_dir = self.dataset_dir / "smplh"
        self.filenames: List[str] = []

        self.standardize = False
        self.velocity = False

        self.num_cams = num_cams
        self.num_feats = num_feats
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.sequential = sequential

        self.load_vertices = load_vertices

        self._smplx_model = None
        self._smplx_ready = False
        self.model_root = self.dataset_dir / "human_model"

    # ------------------------------------------------------------------------------ #

    def _ensure_smplx(self):
        if not _SMPLX_AVAILABLE:
            raise RuntimeError("SMPLX package is not available in this environment")
        if self._smplx_ready:
            if self._smplx_model is None:
                raise RuntimeError("SMPLX model failed to load previously")
            return
        self._smplx_ready = True
        if not self.model_root.exists():
            raise RuntimeError(f"SMPLX model directory not found: {self.model_root}")
        try:
            self._smplx_model = SMPLX(
                model_path=str(self.model_root),
                model_type="smplx",
                gender="neutral",
            )
            self._smplx_model.use_pca = False
        except Exception as exc:
            self._smplx_model = None
            raise RuntimeError(f"Failed to load SMPLX model: {exc}") from exc

    # ------------------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.filenames)

    # ------------------------------------------------------------------------------ #

    def _load_smpl_params(self, filename: str) -> Tuple[dict, Path]:
        pkl_filename = filename + ".pkl"
        smpl_path = self.smpl_dir / pkl_filename
        custom_path = (self.dataset_dir.parent / "custom") / pkl_filename

        chosen_path: Path | None = None
        if smpl_path.exists():
            chosen_path = smpl_path
        elif custom_path.exists():
            chosen_path = custom_path

        if chosen_path is None:
            raise FileNotFoundError(f"SMPL parameter file not found for {filename}")

        with open(chosen_path, "rb") as f:
            params = pickle.load(f, encoding="latin1")
        return params, chosen_path

    # ------------------------------------------------------------------------------ #

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        # ------------------------------------------------------------------ #
        # Load character trajectory (centers)
        char_path = self.data_dir / f"{filename}.npy"
        if not char_path.exists():
            raise FileNotFoundError(f"Character feature not found: {char_path}")
        raw_char_feature = torch.from_numpy(np.load(char_path)).to(torch.float32)
        padding_size = self.num_cams - raw_char_feature.shape[0]
        padded_raw_char_feature = F.pad(raw_char_feature, (0, 0, 0, padding_size)).permute(1, 0)

        # ------------------------------------------------------------------ #
        # Load SMPL parameters and compute root reference
        smpl_params, smpl_path = self._load_smpl_params(filename)
        reasons: List[str] = []

        go = smpl_params.get("global_orient")
        transl = smpl_params.get("transl")
        body_pose = smpl_params.get("body_pose")
        lhand_pose = smpl_params.get("left_hand_pose")
        rhand_pose = smpl_params.get("right_hand_pose")
        betas = smpl_params.get("betas")

        if go is None:
            reasons.append("missing global_orient")
        if transl is None:
            reasons.append("missing transl")

        if go is None or transl is None:
            detail = []
            if go is not None:
                detail.append(_summarize("global_orient", _to_numpy(go)))
            if transl is not None:
                detail.append(_summarize("transl", _to_numpy(transl)))
            raise RuntimeError(
                f"[CharacterDataset] {filename}: insufficient SMPL params ({', '.join(reasons)}) | {'; '.join(detail)}"
            )

        if len(transl) == 0:
            raise RuntimeError(f"[CharacterDataset] {filename}: transl empty")

        go_np = _to_numpy(go)
        transl_np = _to_numpy(transl)
        if go_np is None or not np.isfinite(go_np).all():
            raise RuntimeError(
                f"[CharacterDataset] {filename}: global_orient invalid ({_summarize('global_orient', go_np)})"
            )
        if transl_np is None or not np.isfinite(transl_np).all():
            raise RuntimeError(
                f"[CharacterDataset] {filename}: transl invalid ({_summarize('transl', transl_np)})"
            )

        if go_np.ndim == 4 and go_np.shape[1] == 1:
            go_np = go_np[:, 0]
        if go_np.shape[-2:] != (3, 3):
            raise RuntimeError(
                f"[CharacterDataset] {filename}: global_orient unexpected shape {go_np.shape}"
            )
        root_rotations = _project_to_so3(torch.from_numpy(go_np).to(torch.float32))
        R0 = root_rotations[0]
        t0 = torch.as_tensor(transl_np[0], dtype=torch.float32)

        # Run SMPLX forward to refine pelvis position when possible
        self._ensure_smplx()
        if body_pose is None or lhand_pose is None or rhand_pose is None or betas is None:
            raise RuntimeError(
                f"[CharacterDataset] {filename}: missing SMPL pose parameters"
            )
        try:
            bp = _project_to_so3(torch.tensor(_to_numpy(body_pose)[0], dtype=torch.float32).view(21, 3, 3))
            lh = _project_to_so3(torch.tensor(_to_numpy(lhand_pose)[0], dtype=torch.float32).view(15, 3, 3))
            rh = _project_to_so3(torch.tensor(_to_numpy(rhand_pose)[0], dtype=torch.float32).view(15, 3, 3))
            go0 = root_rotations[0].unsqueeze(0)
            betas0 = torch.tensor(_to_numpy(betas)[0], dtype=torch.float32).view(1, -1)
            transl0 = t0.view(1, 3)

            body_aa = matrix_to_axis_angle(bp).view(1, 21 * 3)
            lhand_aa = matrix_to_axis_angle(lh).view(1, 15 * 3)
            rhand_aa = matrix_to_axis_angle(rh).view(1, 15 * 3)
            go_aa = matrix_to_axis_angle(go0)

            with torch.no_grad():
                out = self._smplx_model(
                    betas=betas0,
                    body_pose=body_aa,
                    global_orient=go_aa,
                    transl=transl0,
                    left_hand_pose=lhand_aa,
                    right_hand_pose=rhand_aa,
                    jaw_pose=torch.zeros(1, 3),
                    leye_pose=torch.zeros(1, 3),
                    reye_pose=torch.zeros(1, 3),
                    expression=torch.zeros(1, 10),
                    return_verts=False,
                )
            pelvis = out.joints[0][0].to(torch.float32)
            if torch.isfinite(pelvis).all():
                t0 = pelvis
            else:
                raise RuntimeError(
                    f"[CharacterDataset] {filename}: pelvis contains non-finite values"
                )
        except Exception as exc:
            raise RuntimeError(f"[CharacterDataset] {filename}: SMPLX forward failed ({exc})")

        if not torch.isfinite(R0).all() or not torch.isfinite(t0).all():
            raise RuntimeError(
                f"[CharacterDataset] {filename}: R0/t0 non-finite"
            )

        char_root_ref = {"R0": R0, "t0": t0}

        # ------------------------------------------------------------------ #
        # Normalize trajectory by root
        char_feature = (raw_char_feature - t0) @ R0

        seq_len = char_feature.shape[0]
        if root_rotations.shape[0] < seq_len:
            pad_rot = root_rotations[-1:].expand(seq_len - root_rotations.shape[0], -1, -1)
            root_rotations = torch.cat([root_rotations, pad_rot], dim=0)
        else:
            root_rotations = root_rotations[:seq_len]

        R0_T = R0.transpose(0, 1).unsqueeze(0)
        root_rot_norm = torch.matmul(R0_T, root_rotations)
        root_rot6d = matrix_to_rotation_6d(root_rot_norm)
        char_feature = torch.cat([char_feature, root_rot6d], dim=1)
        
        root_rot6d_raw = root_rot6d.clone()

        
        if not torch.isfinite(char_feature).all():
            raise RuntimeError(f'[CharacterDataset] {filename}: char_feature contains non-finite values')

        padded_char_feature = F.pad(
            char_feature,
            (0, 0, 0, self.num_cams - char_feature.shape[0]),
        )
        padded_root_rot6d = F.pad(
            root_rot6d_raw,
            (0, 0, 0, self.num_cams - seq_len),
        )

        if self.sequential:
            padded_char_feature = padded_char_feature.permute(1, 0)
        else:
            padded_char_feature = padded_char_feature.reshape(-1)

        raw_feats = {
            "char_raw_feat": padded_raw_char_feature,
            "char_centers": padded_raw_char_feature,
            "char_root_rot6d": padded_root_rot6d,
            "char_root_ref": char_root_ref,
        }

        return filename + ".npy", padded_char_feature, raw_feats

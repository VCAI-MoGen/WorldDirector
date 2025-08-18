import pickle
import torch
import argparse
import numpy as np
import os
from pytorch3d.transforms import matrix_to_axis_angle
import pprint

def get_args():
    parser = argparse.ArgumentParser(description="Load SMPLX .pkl or .npy file and convert to axis-angle.")
    parser.add_argument('--file_path', type=str, required=True,
                        help="Path to the folder containing SMPLX files")
    parser.add_argument('--sample_id', type=str, required=True,
                        help="Sample ID (e.g., 2011_KAeAqaA0Llg_00005_00001)")
    parser.add_argument('--ext', type=str, required=True,
                        help="File extension: .pkl or .npy")
    return parser.parse_args()

def load_params(file_path, sample_id, ext):
    filename = sample_id + ext
    file = os.path.join(file_path, filename)

    if ext == '.pkl':
        with open(file, "rb") as f:
            params = pickle.load(f, encoding='latin1')

    elif ext == '.npy':
        data = np.load(file, allow_pickle=True)

        if isinstance(data, np.ndarray):
            if data.shape == ():  # scalar object
                return data.item()

            else:
                print(f".npy file loaded as array: shape={data.shape}, dtype={data.dtype}")
                print(f"type of data[0]: {type(data[0])}")
                try:
                    preview = data[0]
                    print(f"data[0] preview: shape={getattr(preview, 'shape', None)}, type={type(preview)}")
                except Exception as e:
                    print(f"Failed to preview data[0]: {e}")

                # Return wrapped form
                return {"__raw_array__": data}

        elif isinstance(data, np.lib.npyio.NpzFile):
            return dict(data)

        else:
            raise ValueError("Unsupported .npy format")

    else:
        raise ValueError("Unsupported file extension")

    return params  # only reached for .pkl


def main():
    args = get_args()
    params = load_params(args.file_path, args.sample_id, args.ext)

    print(f"Loaded keys from {args.sample_id + args.ext}:")

    # ---- Case 1: .pkl file (SMPL dict with poses) ----
    if args.ext == '.pkl':
        for k, v in params.items():
            if hasattr(v, 'shape'):
                print(f"{k}: shape = {v.shape}")
            else:
                print(f"{k}: type = {type(v)}")

        def to_tensor(key):
            return torch.tensor(params[key], dtype=torch.float32)

        global_orient = to_tensor('global_orient')
        body_pose = to_tensor('body_pose')
        left_hand_pose = to_tensor('left_hand_pose')
        right_hand_pose = to_tensor('right_hand_pose')
        betas = to_tensor('betas')
        transl = to_tensor('transl')

        B = global_orient.shape[0]
        global_orient_aa = matrix_to_axis_angle(global_orient[:, 0])
        body_pose_aa = matrix_to_axis_angle(body_pose.view(B * 21, 3, 3)).view(B, 21 * 3)
        left_hand_pose_aa = matrix_to_axis_angle(left_hand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)
        right_hand_pose_aa = matrix_to_axis_angle(right_hand_pose.view(B * 15, 3, 3)).view(B, 15 * 3)

        print(f"Converted axis-angle shapes:")
        print(f"global_orient_aa shape  = {global_orient_aa.shape}")
        print(f"body_pose_aa shape      = {body_pose_aa.shape}")
        print(f"left_hand_pose_aa shape = {left_hand_pose_aa.shape}")
        print(f"right_hand_pose_aa shape= {right_hand_pose_aa.shape}")

    # ---- Case 2: .npy file (generic dict) ----
    elif args.ext == '.npy':
        if "__raw_array__" in params:
            arr = params["__raw_array__"]
            print(f"\n Loaded raw array: shape = {arr.shape}, dtype = {arr.dtype}")
            for i, item in enumerate(arr):
                print(f"[{i}] type = {type(item)}, shape = {getattr(item, 'shape', None)}")
                #print(item)
                if i >= 10:
                    print("... truncated ...")
                    break
        else:
            for k, v in params.items():
                print(f"{k}: shape = {getattr(v, 'shape', None)}, type = {type(v)}")
            print("Full content (truncated):")
            pprint.pprint(params, compact=True)

if __name__ == "__main__":
    main()

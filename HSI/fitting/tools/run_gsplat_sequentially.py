
import os
import json

dataset_path = '/home/ckd248/data/Scene2VidBench/Mesh'
data_root_paths = [os.path.join(dataset_path, root_path) for root_path in sorted(os.listdir(dataset_path))] 
for root_path in data_root_paths:
    cmd = f"python run_gsplat_origin.py --root_path {root_path}/traj"
    # --src_ref_images /home/ckd248/data/SCVideo/Video_Generation/VACE/data/ref_human/trumans_human0.png \
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        with open('run_gsplat_sequentially error.txt', 'a') as f:
            f.writelines(f'{root_path}\n')
        print('something bad happened when generating frame list. terminate the script.')

import os
import json

dataset_path = '/home/ckd248/data/Scene2VidBench/GS'
seed = 111
data_root_paths = [os.path.join(dataset_path, root_path) for root_path in sorted(os.listdir(dataset_path))] 
for i, root_path in enumerate(data_root_paths):
    if i < 31:
        continue
    data_path = os.path.join(root_path, 'data.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    text = data['text'] + " The screen is fixed."

    if not os.path.exists(os.path.join(root_path, 'traj', 'human_masked_video.mp4')):
        continue
    if os.path.exists(os.path.join(root_path, 'traj', 'video.mp4')):
        continue
    
    cmd = f"torchrun --master-port 29502 --nproc_per_node=2 vace/vace_wan_inference.py --dit_fsdp \
        --t5_fsdp --ulysses_size 2 --ring_size 1 --size 720p --model_name 'vace-14B' --ckpt_dir models/Wan2.1-VACE-14B \
                --root_path {os.path.join(root_path, 'traj')} \
                    --prompt '{text}'\
                         --base_seed {seed}"
    # --src_ref_images /home/ckd248/data/SCVideo/Video_Generation/VACE/data/ref_human/trumans_human0.png \
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when generating frame list. terminate the script.')
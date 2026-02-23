import os
import torch
from tqdm import tqdm

main_dir = '/data1/cympyc1785/SceneData/DL3DV/scenes'
data_dir = os.listdir(main_dir)
max_len = 0
for d_dir in data_dir:
    data_path_list = os.path.join(main_dir, d_dir)
    max_len = 0
    for data in tqdm(os.listdir(data_path_list)):
        scene_path = os.path.join(data_path_list, data, 'scene.pt')
        c = torch.load(scene_path)
        max_len = max(c.shape[0], max_len)
    print(max_len)
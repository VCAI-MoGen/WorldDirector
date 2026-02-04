import torch
import os
import json
from tqdm import tqdm

from config import cfg, cfg_dict
from core_pkg.common.base import Trainer


trainer = Trainer(cfg)
trainer._make_batch_generator()

data_root_dir = trainer.trainset_loader.dataset_path

for data in tqdm(trainer.trainset_loader):
    data_dir = os.path.join(data_root_dir, data['data_name'])
    target_data_dir = os.path.join(data_dir, "point_avg_scale.json")

    json.dump(data['avg_scale'].item(), open(target_data_dir, "w+"))

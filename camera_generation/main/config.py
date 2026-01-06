import os
import os.path as osp
import sys

class Config:

    # Path
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    dataset_dir = osp.join(root_dir, 'dataset')

    # Training
    lr = 2e-4
    batch_size = 2
    epochs = 10000

    lr_vae = 1e-4
    batch_size_vae = 10
    epochs_vae = 10000

    num_gpus = 1
    num_thread = 8
   

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
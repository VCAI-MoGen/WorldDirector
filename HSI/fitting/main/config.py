import os
import os.path as osp
import sys

class Config:
    
    ## fitting
    kpt_proj_shape = (8, 8)
    depth_render_shape = (256, 256)
    lr_dec_factor = 10
    body_3d_size = 2 # meter
    batch_size = 81
    end_epoch = 3

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, data_root_path, subject_id):
        self.data_root_path = data_root_path
        self.subject_id = subject_id
        self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def set_itr_opt(self, epoch):
        if epoch == 0:
            self.itr_opt_num = 300
        else:
            self.itr_opt_num = 100

    def set_stage(self, epoch, itr):
        if epoch == 0:
            self.lr_dec_itr = [200]
            self.stage_itr = [100]
            self.depth_loss_weight = 0.0
            self.col_loss_weight = 0.0
            if itr < self.stage_itr[0]:
                self.lr = 1e-1
                self.warmup = True
                self.use_depthmap_loss = False
                self.use_col_loss = False
            else:
                self.lr = 1e-2
                self.warmup = False
                self.use_depthmap_loss = False
                self.use_col_loss = False
        elif epoch == 1:
            self.lr = 1e-2
            self.lr_dec_itr = [75]
            self.depth_loss_weight = 10.0
            self.col_loss_weight = 0.0
            self.warmup = False
            self.use_depthmap_loss = True
            self.use_col_loss = False
        else:
            self.lr = 1e-2
            self.lr_dec_itr = [75]
            self.depth_loss_weight = 10.0
            self.col_loss_weight = 100
            self.warmup = False
            self.use_depthmap_loss = True
            self.use_col_loss = True

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

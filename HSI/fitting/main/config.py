import os
import os.path as osp
import sys

class Config:
    
    ## fitting
    kpt_proj_shape = (8, 8)
    depth_render_shape = (256, 256)
    smplx_uvmap_shape = (1024, 1024)
    lr_dec_factor = 10
    body_3d_size = 2 # meter
    batch_size = 81
    end_epoch = 3

    max_sh_degree = 3
    increase_sh_degree_interval = 1000
    densify_end_itr = 15000
    densify_start_itr = 500
    densify_interval = 100
    opacity_reset_interval = 3000
    densify_grad_thr = 0.0002
    opacity_min = 0.005
    dense_percent_thr = 0.01

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
    # add for LHM
    sample_points_path = osp.join(root_dir, 'tools', 'LHM/pretrained_models/dense_sample_points')
    voxel_grid_path = osp.join(root_dir, 'tools', 'LHM/pretrained_models/voxel_grid')

    # learning rate
    lr = 1e-2
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scale_lr = 0.005
    rotation_lr = 0.001
    warmup_itr = 100

    kpt_loss_weight = 1
    depth_loss_weight = 1

    use_fixed_trans = False

    def set_args(self, data_root_path, subject_id):
        self.data_root_path = data_root_path
        self.subject_id = subject_id
        self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.result_dir, exist_ok=True)
 
    def set_itr_opt_num(self, epoch):
        if epoch == 1:
            self.itr_opt_num = 300
        else:
            self.itr_opt_num = 250

    def set_stage(self, epoch, itr):
        if epoch == 0:
            self.lr = 2e-2
            self.lr_dec_itr = [200]
            self.stage_itr = [50]
            self.depth_loss_weight = 1.0
            self.kpt_loss_weight = 1.0
            if itr < self.stage_itr[0]:
                self.warmup = True
                self.use_depthmap_loss = True
                self.use_kpt_loss = False
            else:
                self.warmup = False
                self.use_depthmap_loss = True
                self.use_kpt_loss = False
        elif epoch == 1:
            self.lr = 1e-2
            self.lr_dec_itr = [200]
            self.stage_itr = [50]
            self.depth_loss_weight = 0.01
            self.kpt_loss_weight = 10.0
            if itr < self.stage_itr[0]:
                self.warmup = False
                self.use_depthmap_loss = True
                self.use_kpt_loss = True
            else:
                self.warmup = False
                self.use_depthmap_loss = True
                self.use_kpt_loss = True
        else:
            self.lr = 1e-2
            self.lr_dec_itr = [150]
            self.depth_loss_weight = 0.01
            self.kpt_loss_weight = 10.0
            self.warmup = False
            self.use_depthmap_loss = True
            self.use_kpt_loss = True

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

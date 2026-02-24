import os
import os.path as osp
import sys
import torch

def build_dataset_dir_list():
    dataset_dir_list = []

    dl3dv_data_path = '/data1/cympyc1785/SceneData/DL3DV/scenes'
    dl3dv_dataset_dir_list = [os.path.join(dl3dv_data_path, scene_type) for scene_type in sorted(os.listdir(dl3dv_data_path))]
    dataset_dir_list += dl3dv_dataset_dir_list

    dynamicverse_data_path = '/data1/cympyc1785/SceneData/DynamicVerse/scenes'
    dv_dataset_dir_list = [os.path.join(dynamicverse_data_path, scene_type) for scene_type in sorted(os.listdir(dynamicverse_data_path)) if 'dynpose' not in scene_type]
    dataset_dir_list += dv_dataset_dir_list

    dynpose_data_path = '/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k'
    dp_dataset_dir_list = [os.path.join(dynpose_data_path, scene_type) for scene_type in sorted(os.listdir(dynpose_data_path))]
    dataset_dir_list += dp_dataset_dir_list

    datadop_dir_list = ['/home/ckd248/data/SCVideo/camera_generation/tools/GenDoP/DataDoP/ours']
    dataset_dir_list += datadop_dir_list

    return dataset_dir_list

class Config:

    exp_name = 'multiple_cross_dit_same_pc_pe'

    # Path
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    # dataset_dir = osp.join('/data1/cympyc1785/SceneData/DL3DV/scenes/1K')
    dataset_dir_list = build_dataset_dir_list()
    val_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260220_075745_mmdit_wo_pc_pos_enc/ckpts/900.pth'
    use_concerto = True
    random_seed = 42

    # Training
    lr = 1e-3
    batch_size = 128
    accum_steps = 1
    epochs = 50000
    save_epoch = 50
    load_points = True
    load_saved_pc_embeds = False

    # Diffusion
    model_type = 'baseline' # ['baseline', 'director', 'baseline_mmdit]
    sampling_type = 'ddim'
    diffusion_max_step = 1000
    diffusion_inference_step = 50
    beta_start = 0.00085
    beta_end = 0.012
    prediction_type = 'epsilon'
    beta_schedule = 'scaled_linear'
    clip_sample = False
    set_alpha_to_one = False
    steps_offset = 1

    # VAE
    vae_beta = 1e-3
    # vae_latent_scale = 0.48848
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260123_074547/ckpts/900.pth'
    vae_latent_scale = 0.4467666
    vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260202_065659/ckpts/400.pth'

    num_gpus = 1
    num_thread = 8
    num_frames = 49

    text_len = 512
    t5_dtype = torch.bfloat16
    t5_checkpoint_dir = '/home/ckd248/data/SCVideo/camera_generation/tools/Wan2.2-TI2V-5B'
    t5_checkpoint_path = 'models_t5_umt5-xxl-enc-bf16.pth'
    t5_tokenizer_path = 'google/umt5-xxl'

    clip_version = 'ViT-B/32'
    clip_max_length = 77

    # clatr_ckpt_path = 'CLaTr_checkpoints/epoch99_directorial.ckpt'
    clatr_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/main/evaluate/CLaTr/CLaTr_checkpoints/epoch109_dl3dv_seg_2.ckpt'
    # clatr_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/main/evaluate/CLaTr/CLaTr_checkpoints/epoch89_dl3dv_seg_kl_1e-8.ckpt'

cfg = Config()
cfg_dict = {
    k: v for k, v in vars(Config).items()
    if not k.startswith("__")
}

from core_pkg.common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
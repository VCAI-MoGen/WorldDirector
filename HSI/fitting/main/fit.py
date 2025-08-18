import argparse
import numpy as np
import cv2
from config import cfg
import torch
import torch.nn as nn
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from glob import glob
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from base import Trainer
from utils.vis import render_mesh
from pytorch3d.io import save_ply

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    return args

def axis_angle_to_rotation_6d(x):
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))

def rotation_6d_to_axis_angle(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))

def main():
    args = parse_args()
    root_path = args.root_path
    cfg.set_args(root_path.split('/')[-1])
    
    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # register initial smplx parameters
    smplx_params = {}

    smplx_shape = nn.Parameter(torch.zeros((smpl_x.shape_param_dim)).float().cuda())
    smplx_scale = nn.Parameter(torch.tensor([0.5], requires_grad=True).float().cuda())

    for frame_idx in trainer.smplx_params.keys():
        smplx_params[frame_idx] = {}
        for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose']:
            smplx_params[frame_idx][key] = nn.Parameter(axis_angle_to_rotation_6d(trainer.smplx_params[frame_idx][key].cuda()))
        for key in ['jaw_pose', 'leye_pose', 'reye_pose']:
            smplx_params[frame_idx][key] = nn.Parameter(axis_angle_to_rotation_6d(trainer.smplx_params[frame_idx][key].cuda())) 
        smplx_params[frame_idx]['expr'] = nn.Parameter(trainer.smplx_params[frame_idx]['expr'].cuda())
        smplx_params[frame_idx]['trans'] = nn.Parameter(trainer.smplx_params[frame_idx]['trans'].cuda())

    scene_gs_params = trainer.model.module.scene_gs_params
    human_gs_params = trainer.model.module.human_gs_params

    for epoch in range(cfg.end_epoch):
        cfg.set_itr_opt_num(epoch)

        for itr_data, data in enumerate(trainer.batch_generator):
            batch_size = data['kpt_img'].shape[0]

            for itr_opt in range(cfg.itr_opt_num):
                cfg.set_stage(epoch, itr_opt)

                # optimizer
                if (epoch == 0) and (itr_opt == 0):
                    # smplx root pose and translatioin and scale
                    optimizable_params = []
                    # for frame_idx in data['frame_idx']:
                    #     for key in ['root_pose', 'trans']:
                    #         optimizable_params.append({'params': [smplx_params[int(frame_idx)][key]], 'name': key, 'lr': cfg.lr})
                    optimizable_params.append({'params': [smplx_scale], 'name': 'smplx_scale', 'lr': cfg.lr})
                    trainer.get_optimizer(optimizable_params)
                elif ((epoch == 0 or epoch == 1) and (itr_opt == cfg.stage_itr[0])):
                    # all parameters
                    if epoch == (cfg.end_epoch - 1):
                        optimizable_params = [] # do not optimize shared parameters to make per-frame parameters consistent with the shared ones
                    else:
                        optimizable_params = [{'params': [smplx_shape], 'name': 'smplx_shape', 'lr': cfg.lr}] # all shared parameters
                        optimizable_params.append({'params': [smplx_scale], 'name': 'smplx_scale', 'lr': cfg.lr})
                    for frame_idx in data['frame_idx']:
                        # for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose', 'trans', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr']:
                        for key in ['root_pose', 'trans']:
                        # for key in ['trans']:
                            optimizable_params.append({'params': [smplx_params[int(frame_idx)][key]], 'name': key, 'lr': cfg.lr})
                    trainer.get_optimizer(optimizable_params)
                elif ((epoch > 1) and (itr_opt == 0)):
                    optimizable_params = []
                    for key in ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr']:
                        optimizable_params.append({'params': [smplx_params[int(frame_idx)][key]], 'name': key, 'lr': cfg.lr})

                    # optimizable_params += human_gs_params
                    optimizable_params += scene_gs_params

                    trainer.get_optimizer(optimizable_params)

                # inputs
                smplx_inputs = {'shape': [smplx_shape for _ in range(batch_size)],
                                'scale': [smplx_scale for _ in range(batch_size)]}
                for frame_idx in data['frame_idx']:
                    for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                        if key not in smplx_inputs:
                            smplx_inputs[key] = [smplx_params[int(frame_idx)][key]]
                        else:
                            smplx_inputs[key].append(smplx_params[int(frame_idx)][key])
                for key in smplx_inputs.keys():
                    smplx_inputs[key] = torch.stack(smplx_inputs[key])

                # forwrad
                trainer.set_lr(itr_opt)
                trainer.optimizer.zero_grad()
                loss, out = trainer.model(smplx_inputs, data, return_output=(itr_opt==cfg.itr_opt_num-1))
                loss = {k:loss[k].mean() for k in loss}
                
                # backward
                sum(loss[k] for k in loss).backward()
                trainer.optimizer.step()
                print(cfg.result_dir)

                # log
                screen = [
                    'epoch %d/%d itr_data %d/%d itr_opt %d/%d:' % (epoch, cfg.end_epoch, itr_data, trainer.itr_per_epoch, itr_opt, cfg.itr_opt_num),
                    'lr: %g' % (trainer.get_lr()),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                print(screen)
                print(f'smplx_scale: {round(smplx_scale.item(), 4)}')
            
            # save
            if epoch != (cfg.end_epoch-1):
                continue

            save_root_path = osp.join(root_path, 'smplx_optimized')
            os.makedirs(save_root_path, exist_ok=True)
            smplx_mesh_cam = out['smplx_mesh_cam'].detach().cpu()
            smplx_trans = out['smplx_trans'].detach().cpu().numpy()
            human_traj = out['human_traj'].detach().cpu().numpy()
            for i in range(batch_size):
                frame_idx = int(data['frame_idx'][i])

                # mesh
                save_path = osp.join(save_root_path, 'meshes')
                os.makedirs(save_path, exist_ok=True)
                save_ply(osp.join(save_path, str(frame_idx) + '.ply'), torch.FloatTensor(smplx_mesh_cam[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())

                # render
                save_path = osp.join(save_root_path, 'renders')
                os.makedirs(save_path, exist_ok=True)
                render_smplx = render_mesh(smplx_mesh_cam[i].numpy(), smpl_x.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_smplx.jpg'), render_smplx)

                # smplx parameter
                save_path = osp.join(save_root_path, 'smplx_params')
                os.makedirs(save_path, exist_ok=True)

                with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                    json.dump({'root_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['root_pose'].detach().cpu()).numpy().tolist(), \
                            'body_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['body_pose'].detach().cpu()).numpy().tolist(), \
                            'jaw_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['jaw_pose'].detach().cpu()).numpy().tolist(), \
                            'leye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['leye_pose'].detach().cpu()).numpy().tolist(), \
                            'reye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['reye_pose'].detach().cpu()).numpy().tolist(), \
                            'lhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['lhand_pose'].detach().cpu()).numpy().tolist(), \
                            'rhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['rhand_pose'].detach().cpu()).numpy().tolist(), \
                            'expr': smplx_params[frame_idx]['expr'].detach().cpu().numpy().tolist(), \
                            'trans': smplx_trans[i].tolist()}, f)
                    
                # gs render
                if 'scene_human_render' in out:
                    gs_render = out['scene_human_render'].detach().cpu().numpy()
                    save_path = osp.join(save_root_path, 'gs_render')
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(osp.join(save_path, str(frame_idx) + '_gs_render.jpg'), cv2.cvtColor((gs_render[i].transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

                if (itr_data == 0) and (i == 0):
                    # shape parameter
                    with open(osp.join(save_root_path, 'shape_param.json'), 'w') as f:
                        json.dump(smplx_shape.detach().cpu().numpy().tolist(), f)
                    # mesh scale
                    with open(osp.join(save_root_path, 'scale.json'), 'w') as f:
                        json.dump(smplx_scale.detach().cpu().numpy().tolist(), f)

                # human trajectory
                save_path = osp.join(save_root_path, 'traj')
                os.makedirs(save_path, exist_ok=True)

                with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                    json.dump({'traj': human_traj[i].tolist()}, f)

    # video
    save_path = osp.join(save_root_path, '..', 'smplx_optimized.mp4')
    video_shape = cv2.imread(glob(osp.join(save_root_path, 'renders', '*.jpg'))[0]).shape[:2] # height, width
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_shape[1]*2, video_shape[0]))
    frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in glob(osp.join(save_root_path, 'renders', '*_smplx.jpg'))])
    for frame_idx in frame_idx_list:
        orig_img_path = trainer.trainset_loader.img_paths[frame_idx]
        orig_img = cv2.imread(orig_img_path)
        render = cv2.imread(osp.join(save_root_path, 'renders', str(frame_idx) + '_smplx.jpg'))
        out = np.concatenate((orig_img, render),1)
        cv2.putText(out, str(frame_idx), (int(0.02*render.shape[1]), int(0.1*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [51,51,255], 3, 2) # write frame index
        video_out.write(out)
    video_out.release()

    # gs render video
    if cfg.use_gs_loss:
        save_path = osp.join(save_root_path, '..', 'smplx_optimized_gs.mp4')
        video_shape = cv2.imread(glob(osp.join(save_root_path, 'gs_render', '*.jpg'))[0]).shape[:2] # height, width
        video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_shape[1]*2, video_shape[0]))
        frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in glob(osp.join(save_root_path, 'gs_render', '*_gs_render.jpg'))])
        for frame_idx in frame_idx_list:
            orig_img_path = trainer.trainset_loader.img_paths[frame_idx]
            orig_img = cv2.imread(orig_img_path)
            render = cv2.imread(osp.join(save_root_path, 'gs_render', str(frame_idx) + '_gs_render.jpg'))
            out = np.concatenate((orig_img, render),1)
            cv2.putText(out, str(frame_idx), (int(0.02*render.shape[1]), int(0.1*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [51,51,255], 3, 2) # write frame index
            video_out.write(out)
        video_out.release()


if __name__ == "__main__":
    main()

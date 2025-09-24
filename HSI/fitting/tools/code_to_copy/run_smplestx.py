import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression
from glob import glob
import json
import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    root_path = args.root_path
    file_name = root_path.split('/')[-1]
    exp_name = f"inference_{file_name}_{args.ckpt_name}_{time_str}"
    
    # add from run_hand4whole.py
    save_path = osp.join(root_path, 'smplx_init')
    save_path_lhm = osp.join(root_path, 'smplx_init_lhm')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_lhm, exist_ok=True)
    img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
    img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
    video_save = imageio.get_writer(osp.join(root_path, 'smplx_init.mp4'), fps=30)
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
    bbox = None

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    for frame_idx in tqdm(frame_idx_list):
        
        # prepare input image
        img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
            print(f"{frame_idx} Dropped!")
        elif not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
            frame = np.concatenate((vis_img, rendered_img),1)
            frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,0,0), 3)
            video_save.append_data(frame.astype(np.uint8))

            # save SMPL-X parameters
            root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
            body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
            lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
            rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
            jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
            shape = out['smplx_shape'].detach().cpu().numpy()[0]
            expr = out['smplx_expr'].detach().cpu().numpy()[0]
            trans = out['cam_trans'].detach().cpu().numpy()[0]
            with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                        'body_pose': body_pose.reshape(-1,3).tolist(), \
                        'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                        'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                        'leye_pose': [0,0,0],\
                        'reye_pose': [0,0,0],\
                        'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                        'shape': shape.reshape(-1).tolist(), \
                        'expr': expr.reshape(-1).tolist()}, f)
            with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json'), 'r') as f:
                cam_param = json.load(f)
            with open(osp.join(save_path_lhm, str(frame_idx).zfill(4) + '.json'), 'w') as f:
                json.dump({'betas': shape.reshape(-1).tolist(), \
                        'root_pose': root_pose.reshape(-1).tolist(), \
                        'body_pose': body_pose.reshape(-1,3).tolist(), \
                        'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                        'leye_pose': [0,0,0],\
                        'reye_pose': [0,0,0],\
                        'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                        'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                        'trans': trans.reshape(-1).tolist(), \
                        'focal': cam_param['focal'], \
                        'princpt': [original_img_width / 2, original_img_height / 2], \
                        'img_size_wh': [original_img_width, original_img_height], \
                        'pad_ratio': 0.0}, f)
    video_save.close()

if __name__ == "__main__":
    main()

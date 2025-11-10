import sys
import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx

class SMPLX(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 10
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = {gender: smplx.create(cfg.human_model_path, 'smplx', gender=gender, num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg) for gender in ['neutral', 'male', 'female']}
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces.astype(np.int64)

        # SMPLX joint set
        self.joint = {
                'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
                        'Jaw', 'L_Eye', 'R_Eye', # face joints
                        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
                        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
                        )
                        }
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')

        # keypoint
        self.kpt = {
                'num': 135, # 25 (body kpts) + 40 (hand kpts) + 70 (face keypoints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body kpts
                         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand kpts
                         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand kpts
                         'Head', 'Jaw', *['Face_' + str(i) for i in range(1,69)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
                        ),
                'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body kpts
                    37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand kpts
                    52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand kpts
                    15,22, # head, jaw
                    76,77,78,79,80,81,82,83,84,85, # eyebrow
                    86,87,88,89, # nose
                    90,91,92,93,94, # below nose
                    95,96,97,98,99,100,101,102,103,104,105,106, # eyes
                    107, # right mouth
                    108,109,110,111,112, # upper mouth
                    113, # left mouth
                    114,115,116,117,118, # lower mouth
                    119, # right lip
                    120,121,122, # upper lip
                    123, # left lip
                    124,125,126, # lower lip
                    127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
                    )
                }
        self.kpt['root_idx'] = self.kpt['name'].index('Pelvis')
    
    def get_scale(self, scale):
        return torch.abs(scale)

smpl_x = SMPLX()

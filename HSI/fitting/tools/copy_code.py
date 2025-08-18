import os
import os.path as osp

src_path = './code_to_copy'

# DECA
cmd = 'cp ' + osp.join(src_path, 'run_deca.py') + ' ./DECA/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'deca.py') + ' ./DECA/decalib/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'datasets', 'datasets.py') + ' ./DECA/decalib/datasets/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'demos', 'demo_reconstruct.py') + ' ./DECA/demos/.'
os.system(cmd)

# SMPLer-X
cmd = 'cp ' + osp.join(src_path, 'run_smplestx.py') + ' ./SMPLest-X/.'
os.system(cmd)

# sapiens
cmd = 'cp ' + osp.join(src_path, 'sapiens/lite/demo/*') + ' ./sapiens/lite/demo/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'sapiens/lite/scripts/demo/torchscript/*') + ' ./sapiens/lite/scripts/demo/torchscript/.'
os.system(cmd)

# segment-anything
cmd = 'cp ' + osp.join(src_path, 'run_sam.py') + ' ./segment-anything/.'
os.system(cmd)

# VGGT
cmd = 'cp ' + osp.join(src_path, 'run_vggt.py') + ' ./vggt/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'reconstruct_scene.py') + ' ./vggt/.'
os.system(cmd)

# LHM
cmd = 'cp ' + osp.join(src_path, 'human_lrm.py') + ' ./LHM/LHM/runners/infer/.'
os.system(cmd)
import os
import sys

# Before this, create conda environment, activate it, and install torch
os.chdir('./fitting/tools')
"""
# Install pytorch3d
cmd = "git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && pip install -e ."
os.system(cmd)
os.chdir('../')

# Clone and install pre-trained models

# VGGT
cmd = "git clone https://github.com/facebookresearch/vggt.git && cd vggt && pip install -r requirements.txt && pip install -r requirements_demo.txt"
os.system(cmd)
os.chdir('../')
"""
# SMPLest
cmd = 'git clone https://github.com/SMPLCap/SMPLest-X.git && cd SMPLest-X && pip install -r requirements.txt && apt-get install libosmesa6 -y && pip install pyopengl==3.1.5 imageio-ffmpeg'
os.system(cmd)
os.chdir('../')

# Sapiens
cmd = 'git clone https://github.com/facebookresearch/sapiens.git && cd sapiens && export SAPIENS_ROOT=$(pwd) && export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite'
os.system(cmd)

cmd = 'cd $SAPIENS_ROOT/engine && pip install -e . && cd $SAPIENS_ROOT/cv && pip install -e . && cd $SAPIENS_ROOT/det && pip install -e . && cd ..'
os.system(cmd)

os.chdir('../')

print("Download sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2 from https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b ")
print("Download sapiens_1b_render_people_epoch_88_torchscript.pt2 from https://huggingface.co/facebook/sapiens-depth-1b-torchscript/tree/main")

print("Place them in sapiens/lite/scripts/demo/torchscript/checkpoints directory")
print("If you are ready, press enter")
input()

# DECA
cmd = 'git clone https://github.com/yfeng95/DECA.git && cd DECA && bash fetch_data.sh && pip install yacs==0.1.8 face-alignment==1.3.4 ninja fvcore && pip install git+https://github.com/mattloper/chumpy'
os.system(cmd)

# Segment Anything
cmd = 'git clone https://github.com/facebookresearch/segment-anything.git && cd segment-anything && pip install -e .'
os.system(cmd)

os.chdir('../')

print("Download vit_h from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints")
print("Place it in segment-anything directory")
print("If you are ready, press enter")
input()

cmd = 'pip install open3d'
os.system(cmd)
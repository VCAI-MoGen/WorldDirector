# Fitting SMPL-X to a monocular video and 3D scene

* This branch provides code to fit SMPL-X parametric model to a monocular video and 3d Scene.

## Directory
```
${ROOT}
|-- main
|-- common
|-- |-- utils/human_model_files
|-- |-- |-- smplx/SMPLX_FEMALE.npz
|-- |-- |-- smplx/SMPLX_MALE.npz
|-- |-- |-- smplx/SMPLX_NEUTRAL.npz
|-- |-- |-- smplx/MANO_SMPLX_vertex_ids.pkl
|-- |-- |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|-- |-- |-- smplx/smplx_flip_correspondences.npz
|-- |-- |-- flame/flame_dynamic_embedding.npy
|-- |-- |-- flame/FLAME_FEMALE.pkl
|-- |-- |-- flame/FLAME_MALE.pkl
|-- |-- |-- flame/FLAME_NEUTRAL.pkl
|-- |-- |-- flame/flame_static_embedding.pkl
|-- |-- |-- flame/FLAME_texture.npz
|-- |-- |-- flame/2019/generic_model.pkl
|-- data
|-- |-- Custom
|-- |-- |-- data
|-- tools
|-- |-- code_to_copy
|-- |-- DECA
|-- |-- sapiens
|-- |-- segment-anything
|-- |-- SMPLest-X
|-- |-- vggt
|-- output
```
* `main` contains high-level code for the fitting and configurations.
* `common` contains kernel code. Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). Download FLAME 2019 and 2020 version from [here](https://flame.is.tue.mpg.de/download.php).
* `data` contains data loading code.
* `tools` contains pre-processing code.
* `output` contains log, visualized outputs, and fitting result.

## Custom videos (your own videos)
* Go to `tools` folder and clone [DECA](https://github.com/yfeng95/DECA), [SMPLest-X](https://github.com/SMPLCap/SMPLest-X), [sapiens](https://github.com/facebookresearch/sapiens), [segment-anything](https://github.com/facebookresearch/segment-anything), and [VGGT](https://github.com/facebookresearch/vggt). Install them. For the `SMPLest-X`, we use `smplest_x_h.pth.tar`. For the `segment-anything`, we use `sam_vit_h_4b8939.pth`. For the `sapiens`, we install `sapiens-lite` for inference and use `sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2` for 2D pose estimation and `sapiens_1b_render_people_epoch_88_torchscript.pt2` for depth estimation.
* Run `python copy_code.py` to copy customized code in `code_to_copy` to the above repos.
* Place your video at `data/Custom/data/$SUBJECT_ID/video.mp4`.
* Go to `tools` folder and run `python extract_frames.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID` to extract frames.
* Go to `data/Custom/data/$SUBJECT_ID` and make `frame_list_all.txt`, which includes frame indices to use for the fitting. We recommend to use frames where most of human is visible without being truncated. Each line should contain a frame index and a new line, for example '1\n'. If you want to use all frames, go to `tools` and run `python generate_frame_list_all.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID`.
* Go to `tools` foler and run `python run.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID --use_vggt`. In this way, we get camera parameters from VGGT.
* `run.py` will output camera parameters, initial FLAME parameters, initial SMPL-X parameters, and 2D whole-body keypoints, optimized/smoothed SMPL-X parameters, scene information from vggt, and aligned SMPL-X parameters with 3D scene in `data/Custom/data/$SUBJECT_ID`.

## Troubleshoots
Q. Face of fitted smplx looks bad [[issue13](https://github.com/mks0601/ExAvatar_RELEASE/issues/13)] [[issue39](https://github.com/mks0601/ExAvatar_RELEASE/issues/39)]

A. Checkout renderings in `data/Custom/data/flame_init/renders`. Some people run the fitting without preparing DECA's results.


Q. RuntimeError: indices should be either on cpu or on the same device as the indexted tensor

A. Run `export CUDA_VISIBLE_DEVICES=0` before running the fitting.
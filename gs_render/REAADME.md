# root_path structure
```
root_path/
├── static/
├── traj/
│ ├── cam_params/ # Camera parameters predicted from VGGT
│ ├── cam_params_scene/ # Camera parameters in 3D Gaussian Splatting
│ ├── depths/ # Human depth prediction (Sapiens) + human mask (SAM)
│ ├── depths_scene/ # Depth render using cam_params_scene
│ ├── depths_scene_vis/ # Visualizations of depths_scene
│ ├── depths_vggt/ # Depth prediction from VGGT
│ ├── depths_vggt_optimized/ # Depths optimized with depths_scene
│ ├── flame_init/ # FLAME parameters from DECA
│ ├── frames/ # Generated video frames
│ ├── human_bbox/ # Human bounding boxes from Sapiens
│ ├── keypoints_whole_body/ # 2D keypoints from Sapiens
│ ├── masks/ # Human mask visualization in depths
│ ├── scene_info/ # Predictions from VGGT
│ ├── scene_renders/ # Renders by cam_params_scene
│ ├── smplx_init/ # SMPL-X parameters from SMPLest-X
│ ├── smplx_init_lhm/ # Deprecated
│ └── smplx_optimized/ # Optimized & smoothed SMPL-X params, mesh, trajectory
└── detailed/
└── cam_params_scene/ # Generated cam params from E.T.
```

# Usage of gaussian-splatting-lightning
```bash
git checkout gs
cd tools/gaussian-splatting-lightning
python viewer.py {ply_path} --gsplat --root_path {root_path}
```
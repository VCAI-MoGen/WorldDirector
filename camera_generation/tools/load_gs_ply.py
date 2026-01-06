import numpy as np
from plyfile import PlyData
from gaussian_utils import GaussianPlyUtils

ply_file_path = "/home/ckd248/data/SCVideo/gs_render/tools/gaussian-splatting-lightning/outputs/bicyle/point_cloud/iteration_30000/point_cloud.ply"
gs = GaussianPlyUtils.load_from_ply(ply_file_path).to_parameter_structure()
for k, v in gs.__dict__.items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    else:
        print(k, v)
print(gs.features_dc[5])
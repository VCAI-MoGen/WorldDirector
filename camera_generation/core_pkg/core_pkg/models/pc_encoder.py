import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.utils import offset2batch
from concerto.structure import Point
from torch.nn.utils.rnn import pad_sequence

import numpy as np

def nearest_diff(A, B):
    """
    A: (N, 3) numpy array
    B: (M, 3) numpy array

    return:
        diff: (M, 3)  # A_nearest - B
        idx:  (M,)    # A에서 선택된 index
    """
    # (M, N, 3)
    diff = A[None, :, :] - B[:, None, :]

    # (M, N)
    dist2 = np.sum(diff ** 2, axis=2)

    # (M,)
    idx = np.argmin(dist2, axis=1)

    # (M, 3)
    nearest_A = A[idx]

    # (M, 3)
    diff_vec = nearest_A - B

    return diff_vec, idx

def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    projection = feat @ v
    # projection = projection[:, 9:12]
    projection = projection[:, :3] * 0.2 + projection[:, 3:6] * 0.2 + projection[:, 6:9] * 0.1 + projection[:, 9:12] * 0.5
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color

class PCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        config = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        self.pc_transform = concerto.transform.Compose(config)
        # self.pc_transform = concerto.transform.default()
        self.model = concerto.model.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").cuda()
        # self.model = concerto.model.load("concerto_large", repo_id="Pointcept/Concerto").cuda()
        self.point = None

    def decode(self, point:Point, recover_origin_scale=False):
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        if recover_origin_scale:
            feat = point.feat[point.inverse]
        else:
            feat = point.feat
        return feat, point

    def decode_with_attention(self, point:Point, attention, recover_origin_scale=False):
        """
        attention: (N, 1)
        """
        ret_attention = attention
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            ret_attention = ret_attention[inverse]
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            ret_attention = ret_attention[inverse]
            point = parent
        if recover_origin_scale:
            feat = point.feat[point.inverse]
            ret_attention = ret_attention[point.inverse]
        else:
            feat = point.feat
        return feat, point, ret_attention

    def forward(self, pc_list):
        """
        pc_list      : List of (N, 6)
        """
        
        device = pc_list[0].device
        pc_embed_list = []
        point_obj_list = []
        for pc in pc_list:
            pc = pc.cpu().numpy().copy()
            R_y_down_to_z_up = np.array([
                [1,  0,  0],
                [0,  0, -1],
                [0,  1,  0],
            ])
            pc[:, :3]  = pc[:, :3] @ R_y_down_to_z_up.T
            # pc_offset = np.cumsum(pc_len.cpu().numpy()) # (B, )
            # pc_batch = offset2batch(torch.tensor(pc_offset)).numpy()
            point = {
                "coord": pc[:, :3],  # (N, 3)
                "color": pc[:, 3:],  # (N, 3)
                # "batch": pc_batch, # (N, )
                # "offset": pc_offset, #(B, )
                "normal": np.zeros_like(pc[:, :3]),  # (N, 3)
                # "segment": numpy.array,  # (N,) optional
            }
            x_min, y_min, z_min = point["coord"].min(axis=0)
            x_max, y_max, _ = point["coord"].max(axis=0)
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]

            point = self.pc_transform(point)
            point_coord = point['coord'].cpu().numpy()

            # print(point_coord.tolist()[0] in pc[:, :3].tolist())
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)
            
            point = self.model(point)
            point_embed = point.feat
            pc_embed_list.append(point_embed)
            point_obj_list.append(point)
            # point_coord = point.coord.cpu().numpy()
            # point_coord += shift
            # self.point = point
        pc_embeds = pad_sequence(pc_embed_list, batch_first=True)  # (B, N_max, D)

        lengths = torch.tensor([x.shape[0] for x in pc_embed_list])
        masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
        masks = masks.to(pc_embeds).bool()

        return pc_embeds, masks, point_obj_list # (B, M, 768), (B, M), List of Point
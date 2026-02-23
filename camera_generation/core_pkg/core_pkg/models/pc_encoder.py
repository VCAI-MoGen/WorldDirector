import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.structure import Point
from torch.nn.utils.rnn import pad_sequence
from concerto.utils import batch2offset

import numpy as np

class PCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        config = [
            dict(type="CenterShift", apply_z=True),
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal", "batch"]}),
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
                keys=("coord", "grid_coord", "color", "inverse", "batch"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        self.pc_transform = concerto.transform.Compose(config)
        self.model = concerto.model.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").cuda()
        # self.model = concerto.model.load("concerto_large", repo_id="Pointcept/Concerto").cuda()
        self.point = None

    def decode(self, point:Point, recover_origin_scale=False, num_iter=2):
        for _ in range(num_iter):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        if num_iter == 2 or recover_origin_scale:
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
        else:
            feat = point.feat
        return feat, point

    def decode_with_attention(self, point:Point, attention, recover_origin_scale=False, num_iter=2):
        """
        attention: (N, 1)
        """
        ret_attention = attention
        for _ in range(num_iter):
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

    def forward(self, point_obj):
        for key in point_obj.keys():
            if isinstance(point_obj[key], torch.Tensor):
                point_obj[key] = point_obj[key].cuda(non_blocking=True)
        point = Point(point_obj)
        point = self.model(point)

        # Decode
        _, point = self.decode(point, num_iter=1)

        feat = point.feat
        offset = batch2offset(point['batch'])
        start = torch.cat([offset.new_zeros(1), offset[:-1]])

        pc_embed_list = [
            feat[start[i]:offset[i]]
            for i in range(len(offset))
        ]

        pc_embeds = pad_sequence(pc_embed_list, batch_first=True)  # (B, N_max, D)

        lengths = torch.tensor([x.shape[0] for x in pc_embed_list])
        masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
        masks = masks.to(pc_embeds).bool()

        return pc_embeds, masks, point # (B, M, 768), (B, M), List of Point
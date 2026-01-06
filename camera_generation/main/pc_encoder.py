import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.utils import offset2batch

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
        self.model = concerto.model.load("concerto_large", repo_id="Pointcept/Concerto").cuda()
        self.point_embeddings = torch.randn(2, 800, 768)

    def forward(self, pc, pc_len):
        """
        pc      : (N, 6)
        """
        
        pc = pc.numpy()
        print(pc.shape)
        pc_offset = np.cumsum(pc_len.numpy()) # (N, )
        pc_batch = offset2batch(torch.tensor(pc_offset)).numpy()
        print(pc_batch)
        print(pc_batch.shape)
        print(pc_offset)
        point = {
            "coord": pc[:, :3],  # (N, 3)
            "color": pc[:, 3:],  # (N, 3)
            "batch": pc_batch, # (N, )
            "offset": pc_offset, #(B, )
            "normal": np.zeros_like(pc[:, :3]),  # (N, 3)
            # "segment": numpy.array,  # (N,) optional
        }
        print(point["offset"].shape)
        point = self.pc_transform(point)
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        print(point.keys())
        print(point)
        print(point['offset'])
        point = self.model(point)
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
        feat = point.feat[point.inverse]
        print(point.keys())
        print(feat.shape)
        print(point.offset)
        print(point.batch)
        print(point.coord.shape)
        print(point.grid_coord.shape)
        print(point.pad.shape)
        print(point.unpad.shape)
        print(point.cu_seqlens_key.shape)
        point_embed = point.feat[None]
        
        point_embed = self.point_embeddings.to("cuda")

        return point_embed # (1, M, 768)
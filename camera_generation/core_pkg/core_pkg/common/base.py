import os
import os.path as osp
import math
import time
import glob
import abc
from collections.abc import Mapping, Sequence
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data.dataloader import default_collate
from core_pkg.models.camera_diffusion_model import get_model
from dataset_large import CamDataset

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # config
        self.cfg = cfg

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg=cfg, log_name = 'train_logs.txt')

    def get_optimizer(self, optimizable_parameters):
        self.optimizer = torch.optim.Adam(optimizable_parameters, lr=self.cfg.lr, eps=1e-15)

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self, include_val=True):
        # data load and construct batch generator
        self.trainset_loader = CamDataset(cfg=self.cfg)
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / self.cfg.num_gpus / self.cfg.batch_size)
        self.batch_generator = DataLoader(dataset=self.trainset_loader, \
                                          batch_size=self.cfg.num_gpus*self.cfg.batch_size, \
                                            shuffle=True, num_workers=self.cfg.num_thread, \
                                            persistent_workers=True, collate_fn=collate_fn)
        if include_val:
            self.validset_loader = CamDataset(cfg=self.cfg, type='val')
            self.valid_batch_generator = DataLoader(dataset=self.validset_loader, \
                                            batch_size=self.cfg.num_gpus*self.cfg.batch_size, \
                                                shuffle=False, num_workers=self.cfg.num_thread, \
                                                persistent_workers=True, collate_fn=collate_fn)

    def _make_model(self):
        model = get_model()
        model = DataParallel(model).cuda()
        model.eval()
        self.model = model

def collate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        new_batch = {}

        for key in batch[0]:
            if key == "pc_embeds":
                embeds_list = [d[key] for d in batch]   # [(L1,D), (L2,D), ...]

                lengths = [e.shape[0] for e in embeds_list]
                max_len = max(lengths)
                dim = embeds_list[0].shape[1]

                device = embeds_list[0].device
                dtype = embeds_list[0].dtype

                padded_embeds = []
                masks = []

                for e, L in zip(embeds_list, lengths):
                    pad_len = max_len - L

                    # zero padding
                    if pad_len > 0:
                        pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
                        padded = torch.cat([e, pad], dim=0)
                    else:
                        padded = e

                    padded_embeds.append(padded)

                    # mask (1 = valid, 0 = pad)
                    mask = torch.cat([
                        torch.ones(L, device=device),
                        torch.zeros(pad_len, device=device)
                    ])
                    masks.append(mask)

                new_batch["pc_embeds"] = torch.stack(padded_embeds, dim=0)  # (B, max_len, D)
                new_batch["pc_masks"] = torch.stack(masks, dim=0).bool()      # (B, max_len)

            elif key == "pc":
                new_batch[key] = point_collate_fn([d[key] for d in batch])

            else:
                new_batch[key] = collate_fn([d[key] for d in batch])

        return new_batch
    else:
        return default_collate(batch)
    
def point_collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [point_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                point_collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    point_collate_fn(
                        [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                    ),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)
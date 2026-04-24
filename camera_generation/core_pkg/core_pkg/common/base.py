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
from core_pkg.common.utils.pc_utils import get_ray_sim_per_point
from torch.utils.data import random_split
from core_pkg.models.camera_diffusion_model import get_model
from dataset_large import CamDataset
from torch.nn.utils.rnn import pad_sequence

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

    def _make_batch_generator(self, include_train=True, include_val=True):
        # data load and construct batch generator

        # test_set_path = '/home/ckd248/data/SCVideo/camera_generation/main/test_data_final.txt'
        # with open(test_set_path, 'r') as f:
        #     test_set_list = list(f.readlines())
        #     test_set_list = ['_'.join(x.strip().split('/')[-3:]) for x in test_set_list]
        # train_set_path = '/home/ckd248/data/SCVideo/camera_generation/main/train_data_final.txt'
        # with open(train_set_path, 'r') as f:
        #     train_set_list = list(f.readlines())
        #     train_set_list = ['_'.join(x.strip().split('/')[-3:]) for x in train_set_list]
        
        # if include_train:
        #     self.trainset_loader = CamDataset(cfg=self.cfg, target_list=train_set_list)
        #     print(self.trainset_loader.__len__())
        # if include_val:
        #     self.validset_loader = CamDataset(cfg=self.cfg, target_list=test_set_list)
        #     print(self.validset_loader.__len__())
        
        generator = torch.Generator().manual_seed(self.cfg.random_seed)

        dataset = CamDataset(cfg=self.cfg)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        # test_size = len(dataset)
        # train_size = 0
        # test_size = 0
        # train_size = len(dataset)
        self.trainset_loader, self.validset_loader = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        if include_train:
            self.batch_generator = DataLoader(dataset=self.trainset_loader, \
                                            batch_size=self.cfg.num_gpus*self.cfg.batch_size, \
                                                shuffle=True, num_workers=self.cfg.num_thread, \
                                                persistent_workers=True, drop_last=True, collate_fn=collate_fn)
            self.itr_per_epoch = math.ceil(len(self.trainset_loader) / self.cfg.num_gpus / self.cfg.batch_size)
        if include_val:
            self.valid_batch_generator = DataLoader(dataset=self.validset_loader, \
                                            batch_size=self.cfg.num_gpus*self.cfg.batch_size, \
                                                shuffle=False, num_workers=self.cfg.num_thread, \
                                                persistent_workers=True, collate_fn=collate_fn)
        # from tqdm import tqdm
        # train_data_list = []
        # test_data_list = []
        # for data in tqdm(self.batch_generator):
        #     train_data_list += data['data_name']
        # train_data_list = list(set(train_data_list))
        # with open("train_set_list.txt", "w") as f:
        #     for item in train_data_list:
        #         f.write(str(item) + "\n")
            
        # for data in tqdm(self.valid_batch_generator):
        #     test_data_list += data['data_name']
        # with open("test_set_list.txt", "w") as f:
        #     for item in test_data_list:
        #         f.write(str(item) + "\n")
        # print("Finish")
        # input()
                

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
            if key == 'pc_sim':
                sim_list = [d[key] for d in batch]
                sims = pad_sequence(sim_list, batch_first=True)
                new_batch['pc_sim'] = sims
            elif key == "pc_embeds":
                embeds_list = [d[key] for d in batch]   # [(L1,D), (L2,D), ...]

                lengths = [e.shape[0] for e in embeds_list]
                pc_embeds = pad_sequence(embeds_list, batch_first=True)  # (B, N_max, D)
                lengths = torch.tensor([x.shape[0] for x in embeds_list])
                masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
                masks = masks.to(pc_embeds).bool()
                new_batch["pc_embeds"] = pc_embeds  # (B, max_len, D)
                new_batch["pc_masks"] = masks     # (B, max_len)
                # max_len = max(lengths)
                # dim = embeds_list[0].shape[1]

                # device = embeds_list[0].device
                # dtype = embeds_list[0].dtype

                # padded_embeds = []
                # masks = []

                # for e, L in zip(embeds_list, lengths):
                #     pad_len = max_len - L

                #     # zero padding
                #     if pad_len > 0:
                #         pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
                #         padded = torch.cat([e, pad], dim=0)
                #     else:
                #         padded = e

                #     padded_embeds.append(padded)

                #     # mask (1 = valid, 0 = pad)
                #     mask = torch.cat([
                #         torch.ones(L, device=device),
                #         torch.zeros(pad_len, device=device)
                #     ])
                #     masks.append(mask)

                # new_batch["pc_embeds"] = torch.stack(pc_embeds, dim=0)  # (B, max_len, D)
                # new_batch["pc_masks"] = torch.stack(masks, dim=0).bool()      # (B, max_len)

            elif key == "pc":
                new_batch[key] = point_collate_fn([d[key] for d in batch])
            elif key == 'points' or key == 'inverse' or key == 'points_mask':
                new_batch[key] = [d[key] for d in batch]

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
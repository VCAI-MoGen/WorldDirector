import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_model
from dataset import Dataset

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # config
        self.cfg = cfg

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(self.cfg.log_dir, log_name=log_name)

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
        self.optimizer = torch.optim.Adam(optimizable_parameters, lr=0.0, eps=1e-15)

    def set_lr(self, itr):
        if len(self.cfg.lr_dec_itr) == 0:
            return

        for e in self.cfg.lr_dec_itr:
            if itr < e:
                break
        if itr < self.cfg.lr_dec_itr[-1]:
            idx = self.cfg.lr_dec_itr.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_itr))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        self.trainset_loader = Dataset(transforms.ToTensor(), cfg=self.cfg)
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / self.cfg.num_gpus / self.cfg.batch_size)
        self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=self.cfg.num_gpus*self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_thread)
        self.smplx_params = self.trainset_loader.smplx_params
        self.sdf_dict = self.trainset_loader.sdf_dict

    def _make_model(self):
        model = get_model()
        model = DataParallel(model).cuda()
        model.eval()
        self.model = model

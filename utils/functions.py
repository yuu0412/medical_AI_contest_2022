from cProfile import label
from operator import is_
import numpy as np
import torch
import time
from contextlib import contextmanager
import logging
import sys
import random
import os
from scipy import signal
from torchvision import transforms
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        # fileがない場合は作成
        f = open(path, 'w')
        f.close()

    def __call__(self, score, model, check_loss=True):

        if check_loss:
            score = -score

        if self.best_score is None:
            is_best = True
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            is_best = False
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            is_best = True
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        return is_best

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def init_logger(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)

    handler1 = logging.FileHandler(f'{path}.log')
    logger.addHandler(handler1)

    return logger

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_scheduler(optimizer, config):
    scheduler_name = config.scheduler.scheduler_to_use
    params = config.scheduler[scheduler_name]["params"]
    scheduler = torch.optim.lr_scheduler.__getattribute__(scheduler_name)(optimizer, **params)
    return scheduler

def get_transform(cfg):
    valid_transforms = []
    for name in cfg.transforms:
        if cfg.transforms[name]["use"]:
            if "params" in cfg.transforms[name]:
                params = cfg.transforms[name]["params"]
                transform = transforms.__getattribute__(name)(**params)
            else:
                transform = transforms.__getattribute__(name)()
            valid_transforms.append(transform)
    return transforms.Compose(valid_transforms)

def save_plot(data, path, title=None, xlabel=None, ylabel=None):
    plt.figure() # グラフの初期化
    for k, v in data.items():
        plt.plot(v, label=k)
    plt.title(title) 
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(path)
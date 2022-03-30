
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.run_fold import run_fold
from utils.functions import init_logger, seed_torch, get_scheduler, get_transform
from sklearn.model_selection import StratifiedKFold
from dataset import MedAI2022Dataset
from sklearn.preprocessing import LabelEncoder
from models.timm_models import CNNClassifier
from torchvision import transforms
import os
import sys
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from torch_lr_finder import LRFinder
from utils.functions import save_plot
import albumentations as A
import cv2
from datetime import datetime
import wandb
from torchinfo import summary

@hydra.main(config_path="config", config_name="run")
def main(cfg):

    results = {}

    dt_now = datetime.now()
    today_str = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    results["log_path"] = today_str

    df_train = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/df_train_with_pass.csv')
    train_data = df_train[["id", "source", "path"]]
    train_label = df_train["pneumonia"]
    print(cfg)
    seed_torch(cfg.seed)
    logger = init_logger("train_log")
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    if not cfg.DEBUG:
        wandb.init(project="medical_ai_contest_2022", name=today_str, entity="yuu0412")
        wandb.config.update({
            "model_name": cfg.model.model_name,
            "max_epochs": cfg.max_epochs,
            "train_batch_size": cfg.train_loader.batch_size,
            "val_batch_size": cfg.val_loader.batch_size,
            "optimizer": cfg.optimizer.name,
            "init_lr": cfg.optimizer.lr,
            "scheduler": cfg.scheduler.scheduler_to_use,
        })
    # transform = get_transform(cfg)
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(p=0.5),
        #A.CLAHE(p=0.5),
        #A.GridDistortion(p=0.5),
        #A.OpticalDistortion(p=0.5),
        #A.RandomGamma(p=0.5),
        A.GaussNoise(p=0.5),
        A.HueSaturationValue(p=0.5),
        #A.ToGray(p=0.5),
        A.Cutout(p=0.5),
    ])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    cv_score = 0


    for i, (train_idx, val_idx) in enumerate(kf.split(train_data, train_label)):
        print(f'start cv{i}...')
        if cfg.all_train:
            print("all train")
            train_x, val_x = train_data, train_data.iloc[val_idx]
            train_y, val_y = train_label, train_label.iloc[val_idx]
        else:
            print(f"{kf.n_splits} fold train")
            train_x, val_x = train_data.iloc[train_idx], train_data.iloc[val_idx]
            train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]
        if cfg.DEBUG:
            train_x, val_x = train_x[:cfg.train_loader.batch_size], val_x[:cfg.val_loader.batch_size]
            train_y, val_y = train_y[:cfg.train_loader.batch_size], val_y[:cfg.val_loader.batch_size]
        train_dataset = MedAI2022Dataset(train_x, train_y, is_train=True, transform=transform)
        val_dataset = MedAI2022Dataset(val_x, val_y, is_train=False, transform=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_loader.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_loader.batch_size)
        if cfg.ssl.use:
            model = CNNClassifier(cfg.model.model_name, ssl_path=cfg.ssl.path)
        else:
            model = CNNClassifier(cfg.model.model_name)

        if cfg.DEBUG:
            summary(model, input_size=(32, 3,3,3))

        criterion = nn.__getattribute__(cfg.criterion.name)()
        optimizer = torch.optim.__getattribute__(cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        
        """
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=0.1, num_iter=10)
        print(f"lr:{lr_finder.history['lr']}")
        print(f"loss:{lr_finder.history['loss']}")
        #save_plot({"lr_history": [lr_finder.history["lr"], lr_finder.history["loss"]]}, path="lr_history", xlabel="learinig_rate", ylabel="loss")
        plt.plot(lr_finder.history["lr"], lr_finder.history["loss"])
        plt.xscale("log")
        plt.xlabel("lr")
        plt.ylabel("loss")
        plt.savefig("lr_history")
        lr_finder.plot()
        lr_finder.reset()
        """

        scheduler = get_scheduler(optimizer, cfg)

        max_epochs = cfg.max_epochs
        best_score = run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, fold=i, cfg=cfg)
        results[f"fold_{i}"] = best_score
        cv_score += best_score

    logger.info(results)
    cv_score = cv_score / kf.n_splits
    results["cv"] = cv_score
    logger.info(f"cv_score: {cv_score}")

    # スコアの保存
    if not cfg.DEBUG:
        wandb.log({"cv": cv_score})
        score_file = hydra.utils.get_original_cwd()+"/data/output/score.csv"
        columns = list(results.keys())
        scores = np.array(list(results.values()))
        df_score = pd.DataFrame(np.expand_dims(scores, 0), columns=columns)
        if os.path.isfile(score_file):
            df_old_score = pd.read_csv(score_file)
            df_score = pd.concat([df_old_score, df_score])
        df_score.to_csv(score_file, index=False)


if __name__ == '__main__':
    main()
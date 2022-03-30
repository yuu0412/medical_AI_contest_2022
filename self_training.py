from tkinter import N
from models.timm_models import CNNClassifier
import torch
import torch.nn as nn
import hydra
import pandas as pd
import numpy as np
import math
from dataset import MedAI2022Dataset
from torchvision import transforms
from utils.run_fold import run_fold
from utils.functions import init_logger, seed_torch, get_transform, get_scheduler, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

@hydra.main(config_path="config", config_name="MedAI2022")
def self_training_1(cfg):
    # データの読み込み
    df_train = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/df_train_with_pass.csv')
    train_data = df_train[["id", "source", "path"]]
    train_label = df_train["pneumonia"]
    df_unlabeled = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/df_unlabeled_with_pass.csv')
    df_unlabeled = df_unlabeled[df_unlabeled["source"]=="A"] # Aだけ取り出す
    unlabeled_data = df_unlabeled[["id", "source", "path"]]

    # seed値の固定
    seed_torch(50)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logger = init_logger("self_training")

    # 学習済みモデルの呼び出し
    model = CNNClassifier(cfg.self_training.model_name)
    model.load_state_dict(torch.load(hydra.utils.get_original_cwd()+"/data/output/logs/"+cfg.self_training.weight_path))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    unlabeled_dataset = MedAI2022Dataset(unlabeled_data, is_train=False, transform=transform)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=cfg.test_loader.batch_size, shuffle=False, drop_last=False)

    # ラベル無しデータに対する疑似ラベルの付与
    model.eval()
    model.to(device)

    pred_labels = []
    for batch in tqdm(unlabeled_loader):
        images = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).tolist()
        pred_labels += outputs

    df_unlabeled["pred"] = pred_labels
    df_pseudo_A = df_unlabeled[df_unlabeled["source"]=="A"].reset_index(drop=True)
    df_negative = df_pseudo_A[df_pseudo_A['pred']<0.1] # 予測結果が0.1以下のデータを抽出し、
    df_negative['pneumonia'] = 0 # ラベル=0のデータとして扱う
    df_positive = df_pseudo_A[df_pseudo_A['pred']>0.9] # 予測結果が0.9以上のデータを抽出し、
    df_positive['pneumonia'] = 1 # ラベル=1のデータとして扱う
    df_pseudo_A = pd.concat([df_negative, df_positive]).reset_index(drop=True)
    pseudo_A_data = df_pseudo_A[["id", "source", "path"]]
    pseudo_A_label = df_pseudo_A["pneumonia"]

    # 全データで学習.val用のデータはラベル付きデータを用いる。
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  
    transform = get_transform(cfg)

    for i, (train_idx, val_idx) in enumerate(kf.split(train_data, train_label)):
        print(f'start cv{i}...')
        train_x, val_x = train_data.iloc[train_idx], train_data.iloc[val_idx]
        train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]
        #train_x = pd.concat([train_x, pseudo_A_data])
        #train_y = pd.concat([train_y, pseudo_A_label])
        #print(train_x.shape)
        #print(train_y.shape)
        train_dataset = MedAI2022Dataset(train_x, train_y, is_train=True, transform=transform)
        unlabeled_dataset = MedAI2022Dataset(pseudo_A_data, pseudo_A_label, is_train=False, transform=transform)
        val_dataset = MedAI2022Dataset(val_x, val_y, is_train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_loader.batch_size, shuffle=True, drop_last=True)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=cfg.val_loader.batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_loader.batch_size)
        
        model = CNNClassifier(cfg.model.model_name)

        criterion = nn.__getattribute__(cfg.criterion.name)()
        optimizer = torch.optim.__getattribute__(cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        scheduler = get_scheduler(optimizer, cfg)

        max_epochs = cfg.max_epochs
        #run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, cv=i, config=cfg)
        
        model.to(device)
        early_stopping = EarlyStopping(patience=cfg.early_stopping.patience, verbose=True, delta=0, path=f'best_param_of_cv_{i}.pt', trace_func=logger.info)
        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []

        T1 = 10

        for epoch in range(max_epochs):
            
            logger.info(f'=============== epoch:{epoch+1}/{max_epochs} ===============')
            if epoch >= T1: # T1以上
                train_loss, train_score = training(model, train_loader, epoch, criterion, optimizer, scheduler, device, logger, unlabeled_loader)
            else:
                train_loss, train_score = training(model, train_loader, epoch, criterion, optimizer, scheduler, device, logger)
            train_losses.append(train_loss)
            train_scores.append(train_score)

            val_loss, val_score = evaluation(model, val_loader, criterion, device, logger)
            val_losses.append(val_loss)
            val_scores.append(val_score)

            logger.info(f'[result of epoch {epoch+1}/{max_epochs}]')
            logger.info(f'train_loss:{train_loss} train_score:{train_score}')
            logger.info(f'val_loss:{val_loss} val_score:{val_score}')

            ########################
            #     early stopping   #
            ########################
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("early stopping is adopted.")
                break
            """
            if epoch >= T1-1:
                pred_labels = make_pseudo_labels(model, unlabeled_loader, device)
                df_unlabeled["pred"] = pred_labels
                df_pseudo_A = df_unlabeled[df_unlabeled["source"]=="A"].reset_index(drop=True)
                df_negative = df_pseudo_A[df_pseudo_A['pred']<0.1] # 予測結果が0.1以下のデータを抽出し、
                df_negative['pneumonia'] = 0 # ラベル=0のデータとして扱う
                df_positive = df_pseudo_A[df_pseudo_A['pred']>0.9] # 予測結果が0.9以上のデータを抽出し、
                df_positive['pneumonia'] = 1 # ラベル=1のデータとして扱う
                df_pseudo_A = pd.concat([df_negative, df_positive]).reset_index(drop=True)
                pseudo_A_data = df_pseudo_A[["id", "source", "path"]]
                pseudo_A_label = df_pseudo_A["pneumonia"]
                pseudo_bs = math.ceil(len(pseudo_A_label) / len(train_loader))
                print(f"n_pseudo_label is {pseudo_bs}")
                if pseudo_bs > 0:
                    unlabeled_dataset = MedAI2022Dataset(pseudo_A_data, pseudo_A_label, is_train=True, transform=transform)
                    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=pseudo_bs)
                    is_self_training = True
                    print(f"n_train_batch:{len(train_loader)} n_unlabeled_batch:{len(unlabeled_loader)}")
                else:
                    is_self_training = False
                """

@hydra.main(config_path="config", config_name="MedAI2022")
def self_training_2(cfg):
    # データの読み込み
    df_train = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/df_train_with_pass.csv')
    train_data = df_train[["id", "source", "path"]]
    train_label = df_train["pneumonia"]
    df_unlabeled = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/df_unlabeled_with_pass.csv')
    df_unlabeled = df_unlabeled[df_unlabeled["source"]=="A"] # Aだけ取り出す
    unlabeled_data = df_unlabeled[["id", "source", "path"]]

    # seed値の固定
    seed_torch(50)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logger = init_logger("self_training")
    """
    # 学習済みモデルの呼び出し
    model = CNNClassifier(cfg.self_training.model_name)
    model.load_state_dict(torch.load(hydra.utils.get_original_cwd()+"/data/output/logs/"+cfg.self_training.weight_path))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    unlabeled_dataset = MedAI2022Dataset(unlabeled_data, is_train=False, transform=transform)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=cfg.test_loader.batch_size, shuffle=False, drop_last=False)

    # ラベル無しデータに対する疑似ラベルの付与
    model.eval()
    model.to(device)

    pred_labels = []
    for batch in tqdm(unlabeled_loader):
        images = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).tolist()
        pred_labels += outputs

    df_unlabeled["pred"] = pred_labels
    df_pseudo_A = df_unlabeled[df_unlabeled["source"]=="A"].reset_index(drop=True)
    df_negative = df_pseudo_A[df_pseudo_A['pred']<0.1] # 予測結果が0.1以下のデータを抽出し、
    df_negative['pneumonia'] = 0 # ラベル=0のデータとして扱う
    df_positive = df_pseudo_A[df_pseudo_A['pred']>0.9] # 予測結果が0.9以上のデータを抽出し、
    df_positive['pneumonia'] = 1 # ラベル=1のデータとして扱う
    df_pseudo_A = pd.concat([df_negative, df_positive]).reset_index(drop=True)
    pseudo_A_data = df_pseudo_A[["id", "source", "path"]]
    pseudo_A_label = df_pseudo_A["pneumonia"]
    """
    # 全データで学習.val用のデータはラベル付きデータを用いる。
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  
    transform = get_transform(cfg)

    for i, (train_idx, val_idx) in enumerate(kf.split(train_data, train_label)):
        print(f'start cv{i}...')
        train_x, val_x = train_data.iloc[train_idx], train_data.iloc[val_idx]
        train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]
        #train_x = pd.concat([train_x, pseudo_A_data])
        #train_y = pd.concat([train_y, pseudo_A_label])
        #print(train_x.shape)
        #print(train_y.shape)
        train_dataset = MedAI2022Dataset(train_x, train_y, is_train=True, transform=transform)
        unlabeled_dataset = MedAI2022Dataset(unlabeled_data, is_train=False, transform=transform)
        val_dataset = MedAI2022Dataset(val_x, val_y, is_train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_loader.batch_size, shuffle=True, drop_last=True)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=cfg.val_loader.batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_loader.batch_size)
        
        model = CNNClassifier(cfg.model.model_name)

        criterion = nn.__getattribute__(cfg.criterion.name)()
        optimizer = torch.optim.__getattribute__(cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        scheduler = get_scheduler(optimizer, cfg)

        max_epochs = cfg.max_epochs
        #run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, cv=i, config=cfg)
        
        model.to(device)
        early_stopping = EarlyStopping(patience=cfg.early_stopping.patience, verbose=True, delta=0, path=f'best_param_of_cv_{i}.pt', trace_func=logger.info)
        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []

        T1 = 10

        for epoch in range(max_epochs):
            
            logger.info(f'=============== epoch:{epoch+1}/{max_epochs} ===============')
            if epoch >= T1 and is_self_training: # T1以上
                train_loss, train_score = training(model, train_loader, epoch, criterion, optimizer, scheduler, device, logger, unlabeled_loader)
            else:
                train_loss, train_score = training(model, train_loader, epoch, criterion, optimizer, scheduler, device, logger)
            train_losses.append(train_loss)
            train_scores.append(train_score)

            val_loss, val_score = evaluation(model, val_loader, criterion, device, logger)
            val_losses.append(val_loss)
            val_scores.append(val_score)

            logger.info(f'[result of epoch {epoch+1}/{max_epochs}]')
            logger.info(f'train_loss:{train_loss} train_score:{train_score}')
            logger.info(f'val_loss:{val_loss} val_score:{val_score}')

            ########################
            #     early stopping   #
            ########################
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("early stopping is adopted.")
                break
            
            if epoch >= T1-1:
                pred_labels = make_pseudo_labels(model, unlabeled_loader, device)
                    
                df_unlabeled["pred"] = pred_labels
                df_pseudo_A = df_unlabeled[df_unlabeled["source"]=="A"].reset_index(drop=True)
                df_negative = df_pseudo_A[df_pseudo_A['pred']<0.1] # 予測結果が0.1以下のデータを抽出し、
                df_negative['pneumonia'] = 0 # ラベル=0のデータとして扱う
                df_positive = df_pseudo_A[df_pseudo_A['pred']>0.9] # 予測結果が0.9以上のデータを抽出し、
                df_positive['pneumonia'] = 1 # ラベル=1のデータとして扱う
                df_pseudo_A = pd.concat([df_negative, df_positive]).reset_index(drop=True)
                pseudo_A_data = df_pseudo_A[["id", "source", "path"]]
                pseudo_A_label = df_pseudo_A["pneumonia"]
                pseudo_bs = math.ceil(len(pseudo_A_label) / len(train_loader))
                print(f"n_pseudo_label is {pseudo_bs}")
                if pseudo_bs > 0:
                    unlabeled_dataset = MedAI2022Dataset(pseudo_A_data, pseudo_A_label, is_train=True, transform=transform)
                    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=pseudo_bs)
                    is_self_training = True
                    print(f"n_train_batch:{len(train_loader)} n_unlabeled_batch:{len(unlabeled_loader)}")
                else:
                    is_self_training = False


def training(model, train_loader, epoch, criterion, optimizer, scheduler=None, device="cpu", logger=None, unlabeled_loader=None):

    losses = []
    auc_record = []
    model.train()

    for n in range(len(train_loader)):
        train_loader = iter(train_loader)
        train_images, train_labels = next(train_loader)
        train_images = train_images.to(device)
        train_labels = train_labels.to(device=device, dtype=torch.float)
        labeled_outputs = model(train_images)
        pred_train_labels = np.where(labeled_outputs.to('cpu').detach().numpy()>0.5, 1, 0)
        labeled_loss = criterion(labeled_outputs, train_labels)
        train_labels = train_labels.to('cpu').detach().numpy()
        if unlabeled_loader is not None and n < len(unlabeled_loader):
            unlabeled_loader = iter(unlabeled_loader)
            unlabeled_images, unlabeled_labels = next(unlabeled_loader)
            unlabeled_images = unlabeled_images.to(device)
            unlabeled_labels = unlabeled_labels.to(device=device, dtype=torch.float)
            unlabeled_outputs = model(unlabeled_images)
            pred_unlabeled_labels = np.where(unlabeled_outputs.to('cpu').detach().numpy()>0.5, 1, 0)
            unlabeled_loss = criterion(unlabeled_outputs, unlabeled_labels)
            loss = calc_total_loss(epoch, labeled_loss, unlabeled_loss, alpha=3, T1=10, T2=100)
            losses.append(loss)
            unlabeled_labels = unlabeled_labels.to('cpu').detach().numpy()
            labels = np.concatenate([train_labels, unlabeled_labels])
            pred_labels = np.concatenate([pred_train_labels, pred_unlabeled_labels])
            auc = roc_auc_score(labels, pred_labels)
            auc_record.append(auc)
        else:
            loss = labeled_loss
            losses.append(loss)
            auc = roc_auc_score(train_labels, pred_train_labels)
            auc_record.append(auc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        """
        if n % 10 == 0:
            logger.info(f'[{n}/{len(train_loader)}]: loss:{sum(losses) / len(losses)} score:{sum(auc_record) / len(auc_record)}')
        """
    
    auc_average = sum(auc_record) / len(auc_record)
    loss_average = sum(losses) / len(losses)

    return loss_average, auc_average


def evaluation(model, val_loader, criterion, device="cpu", logger=None):
    losses = []
    auc_record = []
    model.eval()
    for n, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)
            pred_labels = torch.round(outputs)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            labels = labels.to('cpu').detach().numpy()
            pred_labels = pred_labels.to('cpu').detach().numpy()
            auc = roc_auc_score(labels, pred_labels)
            auc_record.append(auc)
        """
        if n % 10 == 0:
            logger.info(f'[{n}/{len(val_loader)}]: loss:{sum(losses) / len(losses)} score:{sum(auc_record) / len(auc_record)}')
        """
    loss_average = sum(losses) / len(losses)
    auc_average = sum(auc_record) / len(auc_record)

    return loss_average, auc_average


def make_pseudo_labels(model, unlabeled_loader, device):
    model.eval()
    model.to(device)
    pred_labels = []
    for batch in tqdm(unlabeled_loader):
        images = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).tolist()
        pred_labels += outputs
    return pred_labels

def calc_total_loss(epoch, labeled_loss, unlabeled_loss, alpha=3, T1=100, T2=600):
    if epoch < T1:
        a = 0
    elif T1 <= epoch < T2:
        a = (epoch-T1)*alpha/(T2-T1)
    elif epoch >= T2:
        a = alpha 
    total_loss = labeled_loss + unlabeled_loss*a
    return total_loss

if __name__ == "__main__":
    self_training_1()
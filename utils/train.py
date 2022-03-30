import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

import numpy as np
from tqdm import tqdm
import wandb

def training(model, train_loader, criterion, optimizer, scheduler=None, device="cpu", logger=None):

    losses = []
    auc_record = []
    model.train()

    for n, batch in enumerate(train_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.float).unsqueeze(1)
        outputs = model(images)
        pred_labels = np.where(outputs.to('cpu').detach().numpy()>0.5, 1, 0)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        labels = labels.to('cpu').detach().numpy()
      
        try:
            auc = roc_auc_score(labels, pred_labels)
            auc_record.append(auc)
        except ValueError:
            print("true labels include only one class. ")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        """
        if n % 10 == 0:
            logger.info(f'[{n}/{len(train_loader)}]: loss:{sum(losses) / len(losses)} score:{sum(auc_record) / len(auc_record)}')
        """
        
        del images, labels
    
    auc_average = sum(auc_record) / len(auc_record)
    loss_average = sum(losses) / len(losses)

    return loss_average, auc_average
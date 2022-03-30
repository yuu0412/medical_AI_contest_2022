from sklearn.metrics import fbeta_score
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import wandb

def evaluation(model, val_loader, criterion, device="cpu", logger=None):
    losses = []
    auc_record = []
    model.eval()
    for n, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.float).unsqueeze(1)

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
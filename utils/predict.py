import torch
from tqdm import tqdm
import numpy as np
import ttach as tta

def predict(models, test_loader, device="cpu", logger=None, ensamble=None, transform=None):
    res = []
    pred_labels = np.zeros(1000)
    if len(models)>1 and ensamble is None:
        raise Exception("複数のモデルを使用する場合はアンサンブルを指定")
    for n, model in enumerate(models):
        #print(pred_labels.shape)
        total_outputs = []
        print(f"using model_{n}")
        model.eval()
        model.to(device)
        if transform is not None:
            print("using tta...")
            print(transform)
            model = tta.ClassificationTTAWrapper(model, transform)
        for batch in tqdm(test_loader):
            images = batch.to(device)
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                total_outputs += (outputs/len(models)).tolist()
                print(total_outputs)
                #print(np.array(total_outputs).shape)
        pred_labels += np.array(total_outputs).squeeze()
    # res = np.where(pred_labels>0.5, 1, 0).tolist()
    res = pred_labels
    return res

if __name__ == "__main__":
    predict()
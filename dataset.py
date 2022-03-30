from os import path
import torch
import numpy as np
import cv2

class MedAI2022Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label=None, is_train=True, transform=None):
        self.paths = data["path"]
        self.labels = label
        self.transform = transform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        path = self.paths.iloc[idx]
        img = cv2.imread(path)
        if self.transform is not None:
            img = self.transforms(image=img)["image"]
        if self.labels is not None:
            y = self.labels.iloc[idx]
            return self.img2tensor(img/255.0), y
        else:
            return self.img2tensor(img/255.0) 
    
    def img2tensor(img,dtype:np.dtype=np.float32):
        if img.ndim==2 : img = np.expand_dims(img,2)
        img = np.transpose(img,(2,0,1))
        return torch.from_numpy(img.astype(dtype, copy=False))
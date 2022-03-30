import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CNNClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, ssl_path=None):
        super().__init__()
        print(f"use {model_name}")"
        print(f"pretrain is {pretrained}")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=0)
        if ssl_path is not None:
            print(f"use ssl: {ssl_path}")
            self.backbone = self.backbone.load_state_dict(torch.load(ssl_path))
        self.head = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        #x = torch.sigmoid(x)
        #x = x.squeeze()

        return x

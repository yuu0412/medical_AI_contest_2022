import torch
from torch import nn
import copy
from utils.functions import save_plot, seed_torch

from lightly.data import LightlyDataset
from lightly.data import DINOCollateFunction, SimCLRCollateFunction
from lightly.loss import DINOLoss, NTXentLoss
from lightly.models.modules import DINOProjectionHead, SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
import timm
import hydra

class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
    
    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


@hydra.main(config_path="config", config_name="ssl")
def ssl(cfg):
    #resnet = torchvision.models.resnet18()
    # backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = timm.create_model(cfg.backbone.name, pretrained=False, in_chans=3, num_classes=0)

    seed_torch(42)

    if cfg.ssl.name == "SimCLR":
        model = SimCLR(backbone)
        collate_fn = SimCLRCollateFunction(
            input_size=32,
            gaussian_blur=0.,
        )
    elif cfg.ssl_name == "DINO":
        model = DINO()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = LightlyDataset(hydra.utils.get_original_cwd()+"/data/input/image/image")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    print("Starting Training")
    loss_log = []
    for epoch in range(cfg.epochs):
        total_loss = 0
        for (x0, x1), _, _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        loss_log.append(avg_loss.item())
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        if epoch % 50==0 and epoch != 0:
            out_path = f"{cfg.ssl.name}_{cfg.backbone.name}_{epoch}.pt"
            torch.save(backbone.state_dict(), out_path)

    backbone = model.backbone
    out_path = f"{cfg.ssl.name}_{cfg.backbone.name}_{cfg.epochs}.pt"
    torch.save(backbone.state_dict(), out_path)
    print(loss_log)
    save_plot({"loss":loss_log}, "loss.jpg")

if __name__=="__main__":
    ssl()
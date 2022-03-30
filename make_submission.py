from models.timm_models import CNNClassifier
import torch
import pandas as pd
import numpy as np
from utils.predict import predict
from dataset import MedAI2022Dataset
from torchvision import transforms
import hydra
from datetime import datetime
import ttach as tta

@hydra.main(config_path="config", config_name="make_submission")
def make_submission(cfg):
    df_test = pd.read_csv(hydra.utils.get_original_cwd()+"/data/input/df_test_with_pass.csv")
    test_data = df_test[["id", "source", "path"]]

    test_dataset = MedAI2022Dataset(test_data, is_train=False, transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_loader.batch_size)
    
    models = []
    for weight_path in cfg.make_submission.weight_paths:
        model = CNNClassifier(cfg.make_submission.model_name)
        print(f"using model weight is {weight_path}")
        model.load_state_dict(torch.load(hydra.utils.get_original_cwd()+"/data/output/logs/train/"+weight_path))
        models.append(model)

    device = "cuda" if torch.cuda.is_available else "cpu"

    if cfg.tta.use:
        transform = tta.Compose([ # get_transformに置き換える予定
            tta.HorizontalFlip(),
            #tta.FiveCrops(150, 150)
        ])
    else:
        transform = None

    pred_labels = predict(models, test_loader, device, ensamble="averaging", transform=transform)
    print(pred_labels)
    df_labels = pd.Series(pred_labels)
    df_submission = pd.concat([df_test["id"], df_labels], axis=1).set_axis(['id', 'pneumonia'], axis=1)
    print(f'saved submission file to {hydra.utils.get_original_cwd()}/data/output/submission/{datetime.now()}.csv')
    dt_now = datetime.now()
    today_str = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    df_submission.to_csv(f'{hydra.utils.get_original_cwd()}/data/output/submission/{today_str}.csv', index=False)

if __name__ == "__main__":
    make_submission()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from datetime import datetime

from .dataset import StandardDataset, QuadrupletDataset
from .models import CustomEfficientNetB0, NovelSiameseWrapper
from utils.config_loader import load_config

def setup_dirs(backbone):
    """
    creates the directories to save results
    """
    base_path = os.path.join("results", "training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths = {
        "base_logs": os.path.join(base_path, "logs", backbone, "baseline"),
        "novel_logs": os.path.join(base_path, "logs", backbone, "novel"),
        "base_weights": os.path.join(base_path, "weights", backbone, "baseline"),
        "novel_weights": os.path.join(base_path, "weights", backbone, "novel")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths, timestamp

def train_one_epoch(model, loader, optimizer, criterion, device, is_novel=False):
    model.train()
    running_loss = 0.0

    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if is_novel:
            results, embeddings = model(imgs)
            loss_bce = criterion['bce'](results.view(-1), labels.view(-1))
            loss_inv_f = criterion['mse'](embeddings[:, 0], embeddings[:, 1])
            loss_inv_r = criterion['mse'](embeddings[:, 2], embeddings[:, 3])
            loss = loss_bce + loss_inv_f + loss_inv_r
        else:
            preds, _ = model(imgs)
            loss = criterion['bce'](preds.view(-1), labels)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def run_experiment(config_path):
    cfg = load_config(config_path)
    backbone_name = cfg['model']['backbone']
    dataset_name = cfg['data']['dataset']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setting up directories and getting the timestamp
    paths, ts = setup_dirs(backbone_name)

    # training the baseline model
    print(f"\n Update: training the baseline {backbone_name}")
    base_ds = StandardDataset(f"data/manifests/{dataset_name}/baseline/train.csv", cfg)
    base_loader = DataLoader(base_ds, batch_size=cfg['train']['batch_size_base'], shuffle=True)
    
    model_b = CustomEfficientNetB0(cfg).to(device)
    opt_b = optim.Adam(model_b.parameters(), lr=cfg['train']['learning_rate'])
    crit_b = {'bce': nn.BCEWithLogitsLoss()}
    
    weight_fn_b = f"baseline_{ts}.pth"
    log_path_b = os.path.join(paths['base_logs'], f"logs_{ts}.csv")
    
    with open(log_path_b, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "datetime", "dataset", "weight_file", "epoch", "loss"])
        for epoch in range(cfg['train']['epochs']):
            loss = train_one_epoch(model_b, base_loader, opt_b, crit_b, device, False)
            writer.writerow(["baseline", ts, dataset_name, weight_fn_b, epoch + 1, f"{loss:.4f}"])
            print(f"Epoch {epoch+1} Loss: {loss:.4f}")
    
    torch.save(model_b.state_dict(), os.path.join(paths['base_weights'], weight_fn_b))

    # training the novel model
    print(f"\n Update: training the novel {backbone_name}")
    novel_ds = QuadrupletDataset(f"data/manifests/{dataset_name}/novel/train.csv", cfg)
    novel_loader = DataLoader(novel_ds, batch_size=cfg['train']['batch_size_novel'], shuffle=True)
    
    backbone_n = CustomEfficientNetB0(cfg).to(device)
    model_n = NovelSiameseWrapper(backbone_n).to(device)
    opt_n = optim.Adam(model_n.parameters(), lr=cfg['train']['learning_rate'])
    crit_n = {'bce': nn.BCEWithLogitsLoss(), 'mse': nn.MSELoss()}
    
    weight_fn_n = f"novel_{ts}.pth"
    log_path_n = os.path.join(paths['novel_logs'], f"logs_{ts}.csv")
    
    with open(log_path_n, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "datetime", "dataset", "weight_file", "epoch", "loss"])
        for epoch in range(cfg['train']['epochs']):
            loss = train_one_epoch(model_n, novel_loader, opt_n, crit_n, device, True)
            writer.writerow(["novel", ts, dataset_name, weight_fn_n, epoch + 1, f"{loss:.4f}"])
            print(f"Epoch {epoch+1} Loss: {loss:.4f}")
            
    torch.save(backbone_n.state_dict(), os.path.join(paths['novel_weights'], weight_fn_n))
    print(f"\nSuccess: results saved for {ts}")

if __name__ == "__main__":
    run_experiment("configs/base_config.yaml")
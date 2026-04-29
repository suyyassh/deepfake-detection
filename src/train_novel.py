import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from datetime import datetime

from .dataset import QuadrupletDataset
from .models import CustomEfficientNetB0, NovelSiameseWrapper
from utils.config_loader import load_config

def setup_dirs(backbone):
    base_path = os.path.join("results", "training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "novel_logs": os.path.join(base_path, "logs", backbone, "novel_stabilized"),
        "novel_weights": os.path.join(base_path, "weights", backbone, "novel_stabilized")
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths, timestamp

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg):
    model.train()
    running_loss, running_bce, running_pull, running_push = 0.0, 0.0, 0.0, 0.0
    
    warmup = cfg['train'].get('warmup_epochs', 3)
    l_pull = 0.1 if epoch >= warmup else 0.0
    l_push = 0.1 if epoch >= warmup else 0.0

    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        
        results, embeddings = model(imgs)
        
        loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())
        loss_pull = criterion['mse'](embeddings[:, 0], embeddings[:, 1]) + \
                    criterion['mse'](embeddings[:, 2], embeddings[:, 3])
        
        margin = 1.0
        dist_raw = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
        dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
        loss_push = torch.relu(margin - dist_raw).mean() + torch.relu(margin - dist_comp).mean()
        
        total_loss = loss_bce + (l_pull * loss_pull) + (l_push * loss_push)
        total_loss.backward()

        # STABILIZER 1: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += total_loss.item()
        running_bce += loss_bce.item()
        running_pull += loss_pull.item()
        running_push += loss_push.item()

    n = len(loader)
    return running_loss/n, running_bce/n, running_pull/n, running_push/n

def validate_one_epoch(model, loader, criterion, device, epoch, cfg):
    model.eval()
    running_loss, running_bce, running_pull, running_push = 0.0, 0.0, 0.0, 0.0
    
    warmup = cfg['train'].get('warmup_epochs', 3)
    l_pull = 0.1 if epoch >= warmup else 0.0
    l_push = 0.1 if epoch >= warmup else 0.0

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            results, embeddings = model(imgs)
            
            loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())
            loss_pull = criterion['mse'](embeddings[:, 0], embeddings[:, 1]) + \
                        criterion['mse'](embeddings[:, 2], embeddings[:, 3])
            
            margin = 1.0
            dist_raw = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
            dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
            loss_push = torch.relu(margin - dist_raw).mean() + torch.relu(margin - dist_comp).mean()
            
            total_loss = loss_bce + (l_pull * loss_pull) + (l_push * loss_push)
            
            running_loss += total_loss.item()
            running_bce += loss_bce.item()
            running_pull += loss_pull.item()
            running_push += loss_push.item()

    n = len(loader)
    return running_loss/n, running_bce/n, running_pull/n, running_push/n

def run_experiment():
    cfg = load_config("configs/base_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, ts = setup_dirs(cfg['model']['backbone'])

    # 1. Initialize Model & Load Baseline Weights
    backbone = CustomEfficientNetB0(cfg).to(device)
    
    # FIND YOUR BEST BASELINE WEIGHT FILENAME IN: results/training/weights/efficientnet_b0/baseline/
    # Replace the path below with your actual successful baseline .pth file
    baseline_path = "results/training/weights/efficientnet_b0/baseline/baseline_20260429_044124.pth"
    
    if os.path.exists(baseline_path):
        print(f"Success: Loading baseline weights from {baseline_path}")
        backbone.load_state_dict(torch.load(baseline_path))
    else:
        print("Warning: Baseline weights not found! Training from scratch.")

    model = NovelSiameseWrapper(backbone, cfg).to(device)

    # 2. DataLoaders
    novel_ds = QuadrupletDataset(f"data/manifests/{cfg['data']['dataset']}/novel/train.csv", cfg)
    train_loader = DataLoader(novel_ds, batch_size=cfg['train']['batch_size_novel'], shuffle=True)
    
    val_ds = QuadrupletDataset(f"data/manifests/{cfg['data']['dataset']}/novel/val.csv", cfg)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size_novel'], shuffle=False)

    # 3. STABILIZER 2: Lower Learning Rate (1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    criterion = {'bce': nn.BCEWithLogitsLoss(), 'mse': nn.MSELoss()}
    
    best_val_loss = float('inf')
    log_file = os.path.join(paths['novel_logs'], f"stabilized_log_{ts}.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "t_loss", "t_bce", "v_loss", "v_bce", "v_pull", "v_push"])

        for epoch in range(cfg['train']['epochs']):
            t_loss, t_bce, t_pull, t_push = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg)
            v_loss, v_bce, v_pull, v_push = validate_one_epoch(model, val_loader, criterion, device, epoch, cfg)

            print(f"Epoch {epoch+1} | T_BCE: {t_bce:.4f} | V_BCE: {v_bce:.4f} | V_Pull: {v_pull:.4f} | V_Push: {v_push:.4f}")
            writer.writerow([epoch+1, t_loss, t_bce, v_loss, v_bce, v_pull, v_push])

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(backbone.state_dict(), os.path.join(paths['novel_weights'], f"novel_stabilized_{ts}.pth"))

if __name__ == "__main__":
    run_experiment()
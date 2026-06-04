# trains the baseline model on compressed (low-quality) images.
# manifests are read from data/manifests/<dataset>/baseline_lq/
# weights are saved to results/training/weights/<backbone>/baseline_lq/
# logs are saved to results/training/logs/<backbone>/baseline_lq/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from datetime import datetime

from src.dataset import StandardDataset
from src.models import UniversalBackbone
from utils.config_loader import load_config, validate_config


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        preds, _ = model(imgs)
        loss = criterion(preds.view(-1), labels.view(-1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            preds, _ = model(imgs)
            loss = criterion(preds.view(-1), labels.view(-1).float())
            running_loss += loss.item()
    return running_loss / len(loader)


def run_experiment(config_path):
    cfg          = load_config(config_path)
    validate_config(cfg)
    backbone     = cfg['model']['backbone']
    dataset      = cfg['data']['dataset']
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")

    # directories
    log_dir    = os.path.join("results", "training", "logs",    backbone, "baseline_lq")
    weight_dir = os.path.join("results", "training", "weights", backbone, "baseline_lq")
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # manifests
    train_csv = f"data/manifests/{dataset}/baseline_lq/train.csv"
    val_csv   = f"data/manifests/{dataset}/baseline_lq/val.csv"
    for p in [train_csv, val_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Manifest not found: {p}\n"
                f"Run preprocessing/create_lq_manifests.py first."
            )

    # data loaders
    train_ds     = StandardDataset(train_csv, cfg)
    val_ds       = StandardDataset(val_csv,   cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size_base'],
                              shuffle=True,  num_workers=cfg['data']['num_workers'])
    val_loader   = DataLoader(val_ds,   batch_size=cfg['train']['batch_size_base'],
                              shuffle=False, num_workers=cfg['data']['num_workers'])

    # model, optimiser, criterion
    model     = UniversalBackbone(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    weight_fn    = f"baseline_lq_{ts}.pth"
    log_path     = os.path.join(log_dir, f"logs_{ts}.csv")
    early_stop   = EarlyStopping(patience=3)
    best_val     = float('inf')

    print(f"\nUpdate: training baseline_lq on compressed images ({backbone})")
    print(f"  train manifest : {train_csv}  ({len(train_ds)} rows)")
    print(f"  val manifest   : {val_csv}    ({len(val_ds)} rows)")
    print(f"  weights → {os.path.join(weight_dir, weight_fn)}")
    print(f"  log     → {log_path}\n")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "datetime", "dataset", "weight_file",
                         "epoch", "train_loss", "val_loss"])

        for epoch in range(cfg['train']['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = validate_one_epoch(model, val_loader, criterion, device)

            writer.writerow(["baseline_lq", ts, dataset, weight_fn,
                             epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}"])
            f.flush()   # write immediately so logs survive an interrupted run

            print(f"Epoch {epoch+1:>3} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(weight_dir, weight_fn))

            early_stop(val_loss)
            if early_stop.early_stop:
                print("Update: early stopping triggered for baseline_lq.")
                break

    print(f"\nSuccess: baseline_lq results saved for {ts}")
    print(f"  weight file: {weight_fn}")


if __name__ == "__main__":
    run_experiment("configs/base_config.yaml")
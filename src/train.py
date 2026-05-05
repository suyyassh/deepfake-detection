import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from datetime import datetime

from .dataset import StandardDataset, QuadrupletDataset
from .models import UniversalBackbone, NovelSiameseWrapper
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

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
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
            self.counter = 0

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg, is_novel=False):
    model.train()

    if is_novel:
        model.backbone.eval()

    running_loss, running_bce, running_pull, running_push = 0.0, 0.0, 0.0, 0.0

    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if is_novel:
            results, embeddings = model(imgs)
            
            # classification loss
            loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())
            
            # pull loss
            loss_pull_f = criterion['mse'](embeddings[:, 0], embeddings[:, 1]) 
            loss_pull_r = criterion['mse'](embeddings[:, 2], embeddings[:, 3]) 
            loss_pull = loss_pull_f + loss_pull_r
            
            # push loss
            margin = 1.0
            dist_raw = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
            dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
            loss_push = torch.relu(margin - dist_raw).mean() + torch.relu(margin - dist_comp).mean()
            
            # dynamic loss scheduling
            warmup = cfg['train'].get('warmup_epochs', 3)
            if epoch < warmup:
                lambda_pull = 0.0 
                lambda_push = 0.0
            else:
                lambda_pull = 0.1 
                lambda_push = 0.1
                
            loss = loss_bce + (lambda_pull * loss_pull) + (lambda_push * loss_push)
            
            # tracking components
            running_bce += loss_bce.item()
            running_pull += loss_pull.item()
            running_push += loss_push.item()
        else:
            preds, _ = model(imgs)
            loss = criterion['bce'](preds.view(-1), labels.view(-1).float())
            running_bce += loss.item()
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    n = len(loader)
    return running_loss / n, running_bce / n, running_pull / n, running_push / n

def validate_one_epoch(model, loader, criterion, device, epoch, cfg, is_novel=False):
    model.eval()
    running_loss, running_bce, running_pull, running_push = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs, labels = imgs.to(device), labels.to(device)
            
            if is_novel:
                results, embeddings = model(imgs)
                
                loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())
                
                loss_pull_f = criterion['mse'](embeddings[:, 0], embeddings[:, 1])
                loss_pull_r = criterion['mse'](embeddings[:, 2], embeddings[:, 3])
                loss_pull = loss_pull_f + loss_pull_r
                
                margin = 1.0
                dist_raw = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
                dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
                loss_push = torch.relu(margin - dist_raw).mean() + torch.relu(margin - dist_comp).mean()
                
                # dynamic loss scheduling
                warmup = cfg['train'].get('warmup_epochs', 3)
                if epoch < warmup:
                    lambda_pull = 0.0 
                    lambda_push = 0.0
                else:
                    lambda_pull = 0.1 
                    lambda_push = 0.1
                    
                loss = loss_bce + (lambda_pull * loss_pull) + (lambda_push * loss_push)
                
                # track components
                running_bce += loss_bce.item()
                running_pull += loss_pull.item()
                running_push += loss_push.item()
            else:
                preds, _ = model(imgs)
                loss = criterion['bce'](preds.view(-1), labels.view(-1).float())
                running_bce += loss.item()
                
            running_loss += loss.item()
            
    n = len(loader)
    return running_loss / n, running_bce / n, running_pull / n, running_push / n

def run_experiment(config_path):
    cfg = load_config(config_path)
    backbone_name = cfg['model']['backbone']
    dataset_name = cfg['data']['dataset']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setting up directories and getting the timestamp
    paths, ts = setup_dirs(backbone_name)

    # # training the baseline model
    # print(f"\n Update: training the baseline {backbone_name}")

    # # loading the training data
    # base_ds = StandardDataset(f"data/manifests/{dataset_name}/baseline/train.csv", cfg)
    # base_loader = DataLoader(base_ds, batch_size=cfg['train']['batch_size_base'], shuffle=True)
    
    # # loading the validation data
    # base_val_ds = StandardDataset(f"data/manifests/{dataset_name}/baseline/val.csv", cfg)
    # base_val_loader = DataLoader(base_val_ds, batch_size=cfg['train']['batch_size_base'], shuffle=False)
    
    # model_b = UniversalBackbone(cfg).to(device)
    # opt_b = optim.Adam(model_b.parameters(), lr=cfg['train']['learning_rate'])
    # crit_b = {'bce': nn.BCEWithLogitsLoss()}
    
    # weight_fn_b = f"baseline_{ts}.pth"
    # log_path_b = os.path.join(paths['base_logs'], f"logs_{ts}.csv")
    
    # # adding early stopping
    # early_stopper_b = EarlyStopping(patience=3)
    # best_val_loss_b = float('inf')
    
    # # creating logs and saving weights
    # with open(log_path_b, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["model", "datetime", "dataset", "weight_file", "epoch", "train_loss", "val_loss"])
        
    #     for epoch in range(cfg['train']['epochs']):
    #         train_loss, _, _, _ = train_one_epoch(model_b, base_loader, opt_b, crit_b, device, epoch, cfg, False)
    #         val_loss, _, _, _ = validate_one_epoch(model_b, base_val_loader, crit_b, device, epoch, cfg, False)
            
    #         writer.writerow(["baseline", ts, dataset_name, weight_fn_b, epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}"])
    #         print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    #         # checkpoint only if the validation loss improves
    #         if val_loss < best_val_loss_b:
    #             best_val_loss_b = val_loss
    #             torch.save(model_b.state_dict(), os.path.join(paths['base_weights'], weight_fn_b))
                
    #         early_stopper_b(val_loss)
    #         if early_stopper_b.early_stop:
    #             print("Update: early stopping triggered for Baseline model.")
    #             break

    # training the novel model
    print(f"\n Update: training the novel {backbone_name}")

    # loading the training data
    novel_ds = QuadrupletDataset(f"data/manifests/{dataset_name}/novel/train.csv", cfg)
    novel_loader = DataLoader(novel_ds, batch_size=cfg['train']['batch_size_novel'], shuffle=True)

    # loading the validation data
    novel_val_ds = QuadrupletDataset(f"data/manifests/{dataset_name}/novel/val.csv", cfg)
    novel_val_loader = DataLoader(novel_val_ds, batch_size=cfg['train']['batch_size_novel'], shuffle=False)
    
    backbone_n = UniversalBackbone(cfg).to(device)
    model_n = NovelSiameseWrapper(backbone_n, cfg).to(device)

    for param in backbone_n.parameters():
        param.requires_grad = False

    opt_n = optim.Adam(model_n.parameters(), lr=cfg['train']['learning_rate'])
    crit_n = {'bce': nn.BCEWithLogitsLoss(), 'mse': nn.MSELoss()}
    
    weight_fn_n = f"novel_{ts}.pth"
    log_path_n = os.path.join(paths['novel_logs'], f"logs_{ts}.csv")

    # adding early stopping
    early_stopper_n = EarlyStopping(patience=3)
    best_val_loss_n = float('inf')
    
    # creating logs and saving weights
    with open(log_path_n, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "datetime", "dataset", "weight_file", "epoch", 
            "train_loss", "train_bce", "train_pull", "train_push", 
            "val_loss", "val_bce", "val_pull", "val_push"
        ])
        
        for epoch in range(cfg['train']['epochs']):
            
            warmup = cfg['train'].get('warmup_epochs', 3)
            if epoch == warmup:
                print("Update: Unfreezing backbone for full fine-tuning...")
                for param in backbone_n.parameters():
                    param.requires_grad = True

            t_loss, t_bce, t_pull, t_push = train_one_epoch(model_n, novel_loader, opt_n, crit_n, device, epoch, cfg, True)
            v_loss, v_bce, v_pull, v_push = validate_one_epoch(model_n, novel_val_loader, crit_n, device, epoch, cfg, True)
            
            writer.writerow([
                "novel", ts, dataset_name, weight_fn_n, epoch + 1, 
                f"{t_loss:.4f}", f"{t_bce:.4f}", f"{t_pull:.4f}", f"{t_push:.4f}",
                f"{v_loss:.4f}", f"{v_bce:.4f}", f"{v_pull:.4f}", f"{v_push:.4f}"
            ])
            
            print(f"Epoch {epoch+1} | "
                  f"T_Loss: {t_loss:.4f} (BCE: {t_bce:.4f}, Pull: {t_pull:.4f}, Push: {t_push:.4f}) | "
                  f"V_Loss: {v_loss:.4f} (BCE: {v_bce:.4f}, Pull: {v_pull:.4f}, Push: {v_push:.4f})")
            
            if v_loss < best_val_loss_n:
                best_val_loss_n = v_loss
                torch.save(backbone_n.state_dict(), os.path.join(paths['novel_weights'], weight_fn_n))
                
            early_stopper_n(v_loss)
            if early_stopper_n.early_stop:
                print("Update: early stopping triggered for Novel model.")
                break
                
    print(f"\nSuccess: results saved for {ts}")

if __name__ == "__main__":
    run_experiment("configs/base_config.yaml")
# hyperparameter tuning script for the novel siamese model.
# tunes contrastive loss weights (lambda_pull, lambda_push) and warmup duration.
# each combination trains a novel model on top of a SHARED baseline checkpoint
# so the baseline is trained only once, keeping the comparison fair.
#
# results are saved to:
#   results/tuning/logs/<backbone>/tuning_<ts>.csv       -- one row per run
#   results/tuning/weights/<backbone>/<run_label>/       -- best checkpoint per run
#
# usage (from repo root):
#     python -m src.tune_novel
#
# edit the GRID at the bottom of this file to change what is searched.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from src.dataset import StandardDataset, QuadrupletDataset
from src.models import UniversalBackbone, NovelSiameseWrapper
from utils.config_loader import load_config, validate_config


# early stopping

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0


# train loops

def train_epoch_novel(model, loader, optimizer, criterion, device,
                      epoch, warmup, lambda_pull, lambda_push):
    model.train()
    if epoch < warmup:
        model.backbone.eval()

    r_loss = r_bce = r_pull = r_push = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        results, embeddings = model(imgs)
        loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())

        loss_pull = (criterion['mse'](embeddings[:, 0], embeddings[:, 1]) +
                     criterion['mse'](embeddings[:, 2], embeddings[:, 3]))

        margin   = 1.0
        dist_raw  = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
        dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
        loss_push = (torch.relu(margin - dist_raw).mean() +
                     torch.relu(margin - dist_comp).mean())

        lp = 0.0 if epoch < warmup else lambda_pull
        lpu = 0.0 if epoch < warmup else lambda_push
        loss = loss_bce + lp * loss_pull + lpu * loss_push

        loss.backward()
        optimizer.step()

        r_loss += loss.item(); r_bce += loss_bce.item()
        r_pull += loss_pull.item(); r_push += loss_push.item()

    n = len(loader)
    return r_loss/n, r_bce/n, r_pull/n, r_push/n


def val_epoch_novel(model, loader, criterion, device,
                    epoch, warmup, lambda_pull, lambda_push):
    model.eval()
    r_loss = r_bce = r_pull = r_push = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            results, embeddings = model(imgs)

            loss_bce = criterion['bce'](results.view(-1), labels.view(-1).float())
            loss_pull = (criterion['mse'](embeddings[:, 0], embeddings[:, 1]) +
                         criterion['mse'](embeddings[:, 2], embeddings[:, 3]))
            margin   = 1.0
            dist_raw  = torch.norm(embeddings[:, 0] - embeddings[:, 2], p=2, dim=1)
            dist_comp = torch.norm(embeddings[:, 1] - embeddings[:, 3], p=2, dim=1)
            loss_push = (torch.relu(margin - dist_raw).mean() +
                         torch.relu(margin - dist_comp).mean())

            lp = 0.0 if epoch < warmup else lambda_pull
            lpu = 0.0 if epoch < warmup else lambda_push
            loss = loss_bce + lp * loss_pull + lpu * loss_push

            r_loss += loss.item(); r_bce += loss_bce.item()
            r_pull += loss_pull.item(); r_push += loss_push.item()

    n = len(loader)
    return r_loss/n, r_bce/n, r_pull/n, r_push/n


# evaluation helper

def evaluate(model, test_csv, val_csv, cfg, device):
    """
    returns auc and val-calibrated accuracy on the given test manifest.
    threshold is frozen from val_csv — no test-set leakage.
    """
    def collect(csv_path):
        ds     = StandardDataset(csv_path, cfg)
        loader = DataLoader(ds, batch_size=cfg['train']['batch_size_base'],
                            shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                out, _ = model(imgs.to(device))
                preds.extend(torch.sigmoid(out).cpu().view(-1).tolist())
                labels.extend(lbls.tolist())
        return preds, labels

    # calibrate threshold on val set
    v_preds, v_labels = collect(val_csv)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(v_labels, v_preds)
    idx     = (tpr - fpr).argmax()
    thresh  = float(thresholds[idx])
    if not (0.0 < thresh < 1.0):
        thresh = 0.5

    # evaluate on test set
    t_preds, t_labels = collect(test_csv)
    auc  = roc_auc_score(t_labels, t_preds)
    acc  = accuracy_score(t_labels, [1 if p > thresh else 0 for p in t_preds])
    return round(auc, 4), round(acc, 4), round(thresh, 4)


# single novel training run

def run_novel(cfg, device, baseline_weight_path, backbone_name, dataset_name,
              lambda_pull, lambda_push, warmup, epochs,
              weight_dir, run_label, ts):
    """
    trains one novel model with the given hyperparameters.
    returns (best_val_loss, epochs_run, weight_path).
    """
    novel_train_csv = f"data/manifests/{dataset_name}/novel/train.csv"
    novel_val_csv   = f"data/manifests/{dataset_name}/novel/val.csv"

    novel_ds  = QuadrupletDataset(novel_train_csv, cfg)
    val_ds    = QuadrupletDataset(novel_val_csv,   cfg)
    loader    = DataLoader(novel_ds, batch_size=cfg['train']['batch_size_novel'],
                           shuffle=True,  num_workers=cfg['data']['num_workers'])
    val_loader= DataLoader(val_ds,   batch_size=cfg['train']['batch_size_novel'],
                           shuffle=False, num_workers=cfg['data']['num_workers'])

    backbone_n = UniversalBackbone(cfg).to(device)
    model_n    = NovelSiameseWrapper(backbone_n, cfg).to(device)
    backbone_n.load_state_dict(torch.load(baseline_weight_path, weights_only=True))

    # freeze backbone during warmup
    for param in backbone_n.parameters():
        param.requires_grad = False

    optimizer  = optim.Adam(model_n.parameters(), lr=cfg['train']['learning_rate'])
    criterion  = {'bce': nn.BCEWithLogitsLoss(), 'mse': nn.MSELoss()}
    stopper    = EarlyStopping(patience=3)
    best_val   = float('inf')
    weight_path = os.path.join(weight_dir, f"novel_{run_label}_{ts}.pth")
    epochs_run  = 0

    for epoch in range(epochs):
        if epoch == warmup:
            for param in backbone_n.parameters():
                param.requires_grad = True

        t_loss, t_bce, t_pull, t_push = train_epoch_novel(
            model_n, loader, optimizer, criterion, device,
            epoch, warmup, lambda_pull, lambda_push)
        v_loss, v_bce, v_pull, v_push = val_epoch_novel(
            model_n, val_loader, criterion, device,
            epoch, warmup, lambda_pull, lambda_push)

        epochs_run += 1
        print(f"ep {epoch+1:>2} | "
              f"T {t_loss:.4f} (bce={t_bce:.4f} pull={t_pull:.4f} push={t_push:.4f}) | "
              f"V {v_loss:.4f} (bce={v_bce:.4f} pull={v_pull:.4f} push={v_push:.4f})")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(backbone_n.state_dict(), weight_path)

        stopper(v_loss)
        if stopper.early_stop:
            print(f"Update: early stopping at epoch {epoch+1}")
            break

    return best_val, epochs_run, weight_path


# main tuning loops

def run_tuning(config_path, grid):
    """
    grid: list of dicts, each with keys lambda_pull, lambda_push, warmup_epochs.
    trains the baseline once, then runs each grid point sequentially.
    """
    cfg          = load_config(config_path)
    validate_config(cfg)
    backbone     = cfg['model']['backbone']
    dataset      = cfg['data']['dataset']
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs       = cfg['train']['epochs']

    log_dir    = os.path.join("results", "tuning", "logs",    backbone)
    weight_root= os.path.join("results", "tuning", "weights", backbone)
    os.makedirs(log_dir, exist_ok=True)

    summary_path = os.path.join(log_dir, f"tuning_{ts}.csv")

    test_raw_csv  = f"data/manifests/{dataset}/test/test_raw.csv"
    test_comp_csv = f"data/manifests/{dataset}/test/test_compressed.csv"
    val_csv       = f"data/manifests/{dataset}/baseline/val.csv"

    # train baseline once
    print(f"Step 1/2 — training shared baseline ({backbone})")

    base_train_csv = f"data/manifests/{dataset}/baseline/train.csv"
    base_val_csv   = f"data/manifests/{dataset}/baseline/val.csv"

    base_ds    = StandardDataset(base_train_csv, cfg)
    base_val_ds= StandardDataset(base_val_csv,   cfg)
    base_loader= DataLoader(base_ds,    batch_size=cfg['train']['batch_size_base'],
                            shuffle=True,  num_workers=cfg['data']['num_workers'])
    base_val_l = DataLoader(base_val_ds,batch_size=cfg['train']['batch_size_base'],
                            shuffle=False, num_workers=cfg['data']['num_workers'])

    model_b  = UniversalBackbone(cfg).to(device)
    opt_b    = optim.Adam(model_b.parameters(), lr=cfg['train']['learning_rate'])
    crit_b   = nn.BCEWithLogitsLoss()
    stopper_b= EarlyStopping(patience=3)
    best_b   = float('inf')

    base_weight_dir = os.path.join(weight_root, "baseline")
    os.makedirs(base_weight_dir, exist_ok=True)
    baseline_weight_path = os.path.join(base_weight_dir, f"baseline_{ts}.pth")

    for epoch in range(epochs):
        model_b.train()
        r = 0.0
        for imgs, labels, _ in base_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_b.zero_grad()
            preds, _ = model_b(imgs)
            loss = crit_b(preds.view(-1), labels.view(-1).float())
            loss.backward(); opt_b.step()
            r += loss.item()
        t_loss = r / len(base_loader)

        model_b.eval(); rv = 0.0
        with torch.no_grad():
            for imgs, labels, _ in base_val_l:
                imgs, labels = imgs.to(device), labels.to(device)
                preds, _ = model_b(imgs)
                rv += crit_b(preds.view(-1), labels.view(-1).float()).item()
        v_loss = rv / len(base_val_l)

        print(f"ep {epoch+1:>2} | Train {t_loss:.4f} | Val {v_loss:.4f}")
        if v_loss < best_b:
            best_b = v_loss
            torch.save(model_b.state_dict(), baseline_weight_path)
        stopper_b(v_loss)
        if stopper_b.early_stop:
            print(f"Update: early stopping at epoch {epoch+1}")
            break

    print(f"Update: baseline weights saved at {baseline_weight_path}")

    # grid search
    print(f"Step 2/2 — novel grid search ({len(grid)} runs)")

    rows = []
    for i, params in enumerate(grid):
        lp     = params['lambda_pull']
        lpu    = params['lambda_push']
        warmup = params['warmup_epochs']
        label  = f"lp{lp}_lpu{lpu}_w{warmup}"

        print(f"\nRun {i+1}/{len(grid)} — {label}")
        print(f"lambda_pull={lp}  lambda_push={lpu}  warmup={warmup}  epochs={epochs}")

        run_weight_dir = os.path.join(weight_root, label)
        os.makedirs(run_weight_dir, exist_ok=True)

        best_val, n_epochs, weight_path = run_novel(
            cfg, device, baseline_weight_path, backbone, dataset,
            lp, lpu, warmup, epochs,
            run_weight_dir, label, ts)

        # load best checkpoint and evaluate
        model_eval = UniversalBackbone(cfg).to(device)
        model_eval.load_state_dict(torch.load(weight_path, weights_only=True,
                                              map_location=device))
        model_eval.eval()

        auc_raw,  acc_raw,  thr_raw  = evaluate(model_eval, test_raw_csv,  val_csv, cfg, device)
        auc_comp, acc_comp, thr_comp = evaluate(model_eval, test_comp_csv, val_csv, cfg, device)

        row = {
            "run":           label,
            "lambda_pull":   lp,
            "lambda_push":   lpu,
            "warmup_epochs": warmup,
            "epochs_run":    n_epochs,
            "best_val_loss": round(best_val, 4),
            "auc_raw":       auc_raw,
            "acc_raw":       acc_raw,
            "threshold_raw": thr_raw,
            "auc_comp":      auc_comp,
            "acc_comp":      acc_comp,
            "threshold_comp":thr_comp,
            "auc_gap":       round(auc_raw - auc_comp, 4),
            "weight_file":   os.path.basename(weight_path),
            "datetime":      ts,
        }
        rows.append(row)

        print(f"RAW: AUC={auc_raw}  acc={acc_raw}  thr={thr_raw}")
        print(f"Compressed: AUC={auc_comp}  acc={acc_comp}  thr={thr_comp}")
        print(f"AUC gap (raw-comp): {row['auc_gap']}")

        # write incrementally so results survive an interrupted run
        write_header = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # summary
    print("Successful: tuning complete! summary (sorted by AUC on Compressed)")
    rows_sorted = sorted(rows, key=lambda r: r['auc_comp'], reverse=True)
    print(f"{'Run':<30} {'AUC_RAW':>8} {'AUC_COMP':>9} {'AUC_GAP':>8} {'ACC_COMP':>9}")
    for r in rows_sorted:
        print(f"{r['run']:<30} {r['auc_raw']:>8} {r['auc_comp']:>9} "
              f"{r['auc_gap']:>8} {r['acc_comp']:>9}")
    print(f"\nFull results saved → {summary_path}")


# grid definition
# each dict is one run. lambda_pull == lambda_push keeps the two terms symmetric.
# warmup_epochs must be < train.epochs in base_config.yaml.

GRID = [
    # vary lambda (warmup fixed at 3)
    {"lambda_pull": 0.01, "lambda_push": 0.01, "warmup_epochs": 3},
    {"lambda_pull": 0.1,  "lambda_push": 0.1,  "warmup_epochs": 3},  # current
    {"lambda_pull": 0.5,  "lambda_push": 0.5,  "warmup_epochs": 3},

    # vary warmup (lambda fixed at 0.1)
    {"lambda_pull": 0.1,  "lambda_push": 0.1,  "warmup_epochs": 1},
    {"lambda_pull": 0.1,  "lambda_push": 0.1,  "warmup_epochs": 5},
]


if __name__ == "__main__":
    run_tuning("configs/base_config.yaml", GRID)
import torch
import os
import csv
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from .dataset import StandardDataset
from .models import CustomEfficientNetB0
from utils.config_loader import load_config

def evaluate_and_log(model_type, weight_path, test_csv, set_name, cfg, device):
    """
    runs inference and returns key metrics
    """
    # initialise and load the model
    model = CustomEfficientNetB0(cfg).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # setup data
    ds = StandardDataset(test_csv, cfg)
    loader = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False)

    all_preds = []
    all_labels = []

    # inference loop
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs, _ = model(imgs)

            # efficient net outputs probability via sigmoid
            all_preds.extend(outputs.cpu().view(-1).tolist())
            all_labels.extend(labels.tolist())

    # calculate metrics
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, binary_preds, output_dict=True, zero_division=0)
    
    return {
        "model": model_type,
        "datetime": None,
        "dataset": cfg['data']['dataset'],
        "test_set": set_name,
        "accuracy": round(acc, 4),
        "f1_score": round(report['macro avg']['f1-score'], 4),
        "auc": round(auc, 4),
        "weight_file": os.path.basename(weight_path)
    }

def run_evaluation(config_path, baseline_weight, novel_weight):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg['model']['backbone']
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # define test sets
    dataset_name = cfg['data']['dataset']
    test_sets = {
        "Set_C_RAW": f"data/manifests/{dataset_name}/test/test_raw.csv",
        "Set_D_COMP": f"data/manifests/{dataset_name}/test/test_compressed.csv"
    }

    results_table = []

    # evaluating both models on both sets
    for m_type, w_path in [("baseline", baseline_weight), ("novel", novel_weight)]:
        for s_name, s_path in test_sets.items():
            print(f"Evaluating {m_type} on {s_name}...")
            metrics = evaluate_and_log(m_type, w_path, s_path, s_name, cfg, device)
            metrics["datetime"] = ts
            results_table.append(metrics)

    # saving logs
    log_dir = os.path.join("results", "testing", "logs", backbone)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_results_{ts}.csv")
    
    df = pd.DataFrame(results_table)
    df.to_csv(log_file, index=False)
    
    # print results
    print(f"\n Evaluation Results for {backbone}")
    print("\n")
    print(df[['model', 'test_set', 'accuracy', 'f1_score', 'auc']].to_string(index=False))
    print("\n")
    print(f"Update: logs saved to {log_file}")

if __name__ == "__main__":
    B_WEIGHT = "results/training/weights/efficientnet_b0/baseline/baseline_20260412_133413.pth"
    N_WEIGHT = "results/training/weights/efficientnet_b0/novel/novel_20260412_133413.pth"
    
    run_evaluation("configs/base_config.yaml", B_WEIGHT, N_WEIGHT)
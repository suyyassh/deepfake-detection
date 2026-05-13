import torch
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from src.dataset import StandardDataset
from src.models import UniversalBackbone
from utils.config_loader import load_config

def evaluate_and_log(model_type, weight_path, test_csv, set_name, cfg, device):
    """
    runs inference on a specific test set and returns key aggregate and method-specific metrics.
    """
    
    # initialise the model 
    model = UniversalBackbone(cfg).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    # setup the standard dataset
    ds = StandardDataset(test_csv, cfg)
    loader = DataLoader(ds, batch_size=cfg['train']['batch_size_base'], shuffle=False)

    all_preds = []
    all_labels = []
    all_methods = []

    # inference loop
    with torch.no_grad():
        for batch in loader:
            imgs, labels, methods = batch[0].to(device), batch[1], batch[2]
            
            # forward pass
            outputs, _ = model(imgs)

            # convert logits to probabilities
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().view(-1).tolist())
            all_labels.extend(labels.tolist())
            all_methods.extend(methods)

    # calculate aggregate metrics
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, binary_preds, output_dict=True, zero_division=0)
    
    # build base metrics dict
    metrics = {
        "model": model_type,
        "datetime": None,
        "dataset": cfg['data']['dataset'],
        "test_set": set_name,
        "accuracy": round(acc, 4),
        "f1_score": round(report['macro avg']['f1-score'], 4),
        "auc": round(auc, 4),
        "weight_file": os.path.basename(weight_path)
    }

    # calculate accuracy
    df_eval = pd.DataFrame({
        'label': all_labels, 
        'pred': binary_preds, 
        'method': all_methods
    })
    
    for method in df_eval['method'].unique():
        method_data = df_eval[df_eval['method'] == method]
        # prevent division by 0
        if len(method_data) > 0:
            method_acc = accuracy_score(method_data['label'], method_data['pred'])
            metrics[f"acc_{method}"] = round(method_acc, 4)

    return metrics

def run_evaluation(config_path, baseline_weight, novel_weight):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg['model']['backbone']
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset_name = cfg['data']['dataset']
    
    # Verify these paths match your actual directory structure!
    test_sets = {
        "RAW": f"data/manifests/{dataset_name}/test/test_raw.csv",
        "Compressed": f"data/manifests/{dataset_name}/test/test_compressed.csv"
    }

    results_table = []

    print("Update: starting evaluation pipeline")

    # evaluate both models on both test sets
    for m_type, w_path in [("Baseline", baseline_weight), ("Novel", novel_weight)]:
        if not os.path.exists(w_path):
            print(f"Error: could not find weight file for {m_type}: {w_path}")
            continue
            
        for s_name, s_path in test_sets.items():
            print(f"Update: evaluating {m_type} model on {s_name} images")
            metrics = evaluate_and_log(m_type, w_path, s_path, s_name, cfg, device)
            metrics["datetime"] = ts
            results_table.append(metrics)

    if not results_table:
        print("Update: evaluation failed, please check file paths")
        return

    # saving to csv
    log_dir = os.path.join("results", "testing", "logs", backbone)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_results_{ts}.csv")
    
    df = pd.DataFrame(results_table)
    
    # ordering the columns
    cols = ['model', 'test_set', 'accuracy', 'f1_score', 'auc']
    method_cols = [c for c in df.columns if c.startswith('acc_')]
    final_cols = cols + method_cols + ['dataset', 'weight_file', 'datetime']
    df = df[[c for c in final_cols if c in df.columns]] 
    
    df.to_csv(log_file, index=False)
    
    # Print the final thesis table
    print(f"Final Evaluation Results ({backbone})")
    print("\n Aggregate Metrics ")
    print(df[['model', 'test_set', 'accuracy', 'auc']].to_string(index=False))
    
    print("\n Method Breakdown (Accuracy)")
    print(df[['model', 'test_set'] + method_cols].to_string(index=False))
    
    print(f"Success: logs saved to {log_file}")

if __name__ == "__main__":    
    B_WEIGHT = "results/training/weights/efficientnet_b0/baseline/baseline_20260512_160927.pth"
    N_WEIGHT = "results/training/weights/efficientnet_b0/novel/novel_20260512_160927.pth"
    
    run_evaluation("configs/base_config.yaml", B_WEIGHT, N_WEIGHT)
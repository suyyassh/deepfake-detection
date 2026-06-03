import torch
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader

from src.dataset import StandardDataset
from src.models import UniversalBackbone
from utils.config_loader import load_config


def collect_predictions(model, csv_path, cfg, device):
    """
    runs the model over a manifest and returns (probs, labels, methods).
    shared by both threshold calibration and final evaluation, so inference
    is identical in both paths.
    """
    ds = StandardDataset(csv_path, cfg)
    loader = DataLoader(ds, batch_size=cfg['train']['batch_size_base'], shuffle=False)

    all_preds, all_labels, all_methods = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels, methods = batch[0].to(device), batch[1], batch[2]
            outputs, _ = model(imgs)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().view(-1).tolist())
            all_labels.extend(labels.tolist())
            all_methods.extend(methods)
    return all_preds, all_labels, all_methods


def find_threshold(model, val_csv, cfg, device):
    """
    finds the threshold that maximises Youden's J (tpr - fpr) on a HELD-OUT
    validation manifest. This manifest is never used as a test set, so the
    resulting threshold carries no test-set information.
    """
    all_preds, all_labels, _ = collect_predictions(model, val_csv, cfg, device)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

    # sklearn prepends an artificial threshold of +inf to the ROC curve (the
    # "classify everything negative" point). Youden's J is 0 there so argmax
    # normally avoids it, but a degenerate val set can still surface it; guard
    # against returning a non-finite threshold by falling back to 0.5.
    optimal_idx = (tpr - fpr).argmax()
    chosen = float(thresholds[optimal_idx])
    if not (0.0 < chosen < 1.0):
        print(f"Warning: calibrated threshold {chosen} out of range; falling back to 0.5")
        return 0.5
    return chosen


def metrics_from_preds(all_preds, all_labels, all_methods,
                       model_type, set_name, threshold, threshold_type,
                       cfg, weight_path):
    """
    computes aggregate and per-method metrics from precomputed predictions,
    applying the supplied decision threshold. AUC is threshold-free and so is
    identical across threshold types for a given (model, test_set).
    """
    binary_preds = [1 if p > threshold else 0 for p in all_preds]

    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, binary_preds, output_dict=True, zero_division=0)

    metrics = {
        "model": model_type,
        "datetime": None,
        "dataset": cfg['data']['dataset'],
        "test_set": set_name,
        "threshold_type": threshold_type,   # "fixed_0.5" (primary) or "val_calibrated" (secondary)
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(report['macro avg']['f1-score'], 4),
        "auc": round(auc, 4),
        "weight_file": os.path.basename(weight_path)
    }

    df_eval = pd.DataFrame({'label': all_labels, 'pred': binary_preds, 'method': all_methods})
    for method in df_eval['method'].unique():
        method_data = df_eval[df_eval['method'] == method]
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

    test_sets = {
        "RAW": f"data/manifests/{dataset_name}/test/test_raw.csv",
        "Compressed": f"data/manifests/{dataset_name}/test/test_compressed.csv"
    }

    # HELD-OUT calibration manifest. This is the baseline validation split, which
    # points to RAW (c23) frames and is disjoint from every test set above.
    # Calibrating here, NOT on a test set, is what keeps the reported accuracy honest.
    calib_csv = f"data/manifests/{dataset_name}/baseline/val.csv"
    if not os.path.exists(calib_csv):
        print(f"Error: calibration manifest not found: {calib_csv}")
        return

    results_table = []
    print("Update: starting evaluation pipeline")

    for m_type, w_path in [("Baseline", baseline_weight), ("Novel", novel_weight)]:
        if not os.path.exists(w_path):
            print(f"Error: could not find weight file for {m_type}: {w_path}")
            continue

        # load the model ONCE per model
        model = UniversalBackbone(cfg).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
        model.eval()

        # freeze a single validation-calibrated threshold for this model
        val_threshold = find_threshold(model, calib_csv, cfg, device)
        print(f"Update: {m_type} val-calibrated threshold (frozen): {val_threshold:.4f}")

        # evaluate on each test set at BOTH thresholds, reusing one inference pass
        for s_name, s_path in test_sets.items():
            if not os.path.exists(s_path):
                print(f"Error: test manifest not found: {s_path}")
                continue
            print(f"Update: evaluating {m_type} on {s_name}")
            preds, labels, methods = collect_predictions(model, s_path, cfg, device)

            # PRIMARY: fixed 0.5 (matches report Section 7.1)
            m_fixed = metrics_from_preds(preds, labels, methods, m_type, s_name,
                                         0.5, "fixed_0.5", cfg, w_path)
            m_fixed["datetime"] = ts
            results_table.append(m_fixed)

            # SECONDARY: frozen validation-calibrated threshold (Section 9.2 experiment)
            m_calib = metrics_from_preds(preds, labels, methods, m_type, s_name,
                                         val_threshold, "val_calibrated", cfg, w_path)
            m_calib["datetime"] = ts
            results_table.append(m_calib)

    if not results_table:
        print("Update: evaluation failed, please check file paths")
        return

    log_dir = os.path.join("results", "testing", "logs", backbone)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_results_{ts}.csv")

    df = pd.DataFrame(results_table)

    cols = ['model', 'test_set', 'threshold_type', 'threshold',
            'accuracy', 'f1_score', 'auc']
    method_cols = [c for c in df.columns if c.startswith('acc_')]
    final_cols = cols + method_cols + ['dataset', 'weight_file', 'datetime']
    df = df[[c for c in final_cols if c in df.columns]]
    df.to_csv(log_file, index=False)

    print(f"\nFinal Evaluation Results ({backbone})")

    print("\n Primary metrics (fixed 0.5 threshold)")
    primary = df[df['threshold_type'] == "fixed_0.5"]
    print(primary[['model', 'test_set', 'accuracy', 'f1_score', 'auc']].to_string(index=False))

    print("\n Secondary metrics (validation-calibrated threshold)")
    secondary = df[df['threshold_type'] == "val_calibrated"]
    print(secondary[['model', 'test_set', 'threshold', 'accuracy', 'f1_score', 'auc']].to_string(index=False))

    print("\n Method breakdown — primary (fixed 0.5)")
    print(primary[['model', 'test_set'] + method_cols].to_string(index=False))

    print(f"\nSuccess: logs saved to {log_file}")

if __name__ == "__main__":
    B_WEIGHT = ""         
    N_WEIGHT = ""         

    run_evaluation("configs/base_config.yaml", B_WEIGHT, N_WEIGHT)
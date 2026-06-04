# evaluates the baseline_lq model (trained on compressed images) on both
# the RAW and Compressed test sets, using the same dual-threshold protocol
# as src/testing.py.

import torch
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader

from src.dataset import StandardDataset
from src.models import UniversalBackbone
from utils.config_loader import load_config, validate_config


def collect_predictions(model, csv_path, cfg, device):
    """
    runs the model over a manifest and returns (probs, labels, methods).
    """
    ds     = StandardDataset(csv_path, cfg)
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
    finds Youden's-J-optimal threshold on the held-out validation manifest.
    falls back to 0.5 if the calibrated value is out of (0, 1).
    """
    preds, labels, _ = collect_predictions(model, val_csv, cfg, device)
    fpr, tpr, thresholds = roc_curve(labels, preds)
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
    computes aggregate and per-method metrics at the given threshold.
    """
    binary_preds = [1 if p > threshold else 0 for p in all_preds]

    acc    = accuracy_score(all_labels, binary_preds)
    auc    = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, binary_preds,
                                   output_dict=True, zero_division=0)

    metrics = {
        "model":          model_type,
        "datetime":       None,
        "dataset":        cfg['data']['dataset'],
        "test_set":       set_name,
        "threshold_type": threshold_type,
        "threshold":      round(threshold, 4),
        "accuracy":       round(acc, 4),
        "f1_score":       round(report['macro avg']['f1-score'], 4),
        "auc":            round(auc, 4),
        "weight_file":    os.path.basename(weight_path),
    }

    df_eval = pd.DataFrame({'label': all_labels,
                            'pred':  binary_preds,
                            'method': all_methods})
    for method in df_eval['method'].unique():
        subset = df_eval[df_eval['method'] == method]
        if len(subset) > 0:
            metrics[f"acc_{method}"] = round(
                accuracy_score(subset['label'], subset['pred']), 4)

    return metrics


def run_evaluation(config_path, lq_weight):
    cfg         = load_config(config_path)
    validate_config(cfg)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone    = cfg['model']['backbone']
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset     = cfg['data']['dataset']

    test_sets = {
        "RAW":        f"data/manifests/{dataset}/test/test_raw.csv",
        "Compressed": f"data/manifests/{dataset}/test/test_compressed.csv",
    }

    # held-out calibration set: baseline_lq/val.csv (compressed images,
    # disjoint from both test sets)
    calib_csv = f"data/manifests/{dataset}/baseline_lq/val.csv"
    if not os.path.exists(calib_csv):
        print(f"Error: calibration manifest not found: {calib_csv}")
        return

    if not os.path.exists(lq_weight):
        print(f"Error: weight file not found: {lq_weight}")
        return

    # load model once
    model = UniversalBackbone(cfg).to(device)
    model.load_state_dict(torch.load(lq_weight, map_location=device,
                                     weights_only=True))
    model.eval()

    # freeze calibration threshold from val set
    val_threshold = find_threshold(model, calib_csv, cfg, device)
    print(f"Update: baseline_lq val-calibrated threshold (frozen): {val_threshold:.4f}")

    results_table = []
    print("Update: starting baseline_lq evaluation")

    for s_name, s_path in test_sets.items():
        if not os.path.exists(s_path):
            print(f"Error: test manifest not found: {s_path}")
            continue
        print(f"Update: evaluating baseline_lq on {s_name}")
        preds, labels, methods = collect_predictions(model, s_path, cfg, device)

        # primary: fixed 0.5
        m_fixed = metrics_from_preds(preds, labels, methods,
                                     "Baseline_LQ", s_name,
                                     0.5, "fixed_0.5", cfg, lq_weight)
        m_fixed["datetime"] = ts
        results_table.append(m_fixed)

        # secondary: val-calibrated
        m_calib = metrics_from_preds(preds, labels, methods,
                                     "Baseline_LQ", s_name,
                                     val_threshold, "val_calibrated",
                                     cfg, lq_weight)
        m_calib["datetime"] = ts
        results_table.append(m_calib)

    if not results_table:
        print("Update: evaluation failed — check file paths")
        return

    # save results
    log_dir = os.path.join("results", "testing", "logs", backbone, "baseline_lq")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_results_{ts}.csv")

    df = pd.DataFrame(results_table)
    cols        = ['model', 'test_set', 'threshold_type', 'threshold',
                   'accuracy', 'f1_score', 'auc']
    method_cols = [c for c in df.columns if c.startswith('acc_')]
    final_cols  = cols + method_cols + ['dataset', 'weight_file', 'datetime']
    df = df[[c for c in final_cols if c in df.columns]]
    df.to_csv(log_file, index=False)

    # print summary
    print(f"\nFinal Evaluation Results — Baseline LQ ({backbone})")

    print("\n Primary metrics (fixed 0.5 threshold)")
    primary = df[df['threshold_type'] == "fixed_0.5"]
    print(primary[['model', 'test_set', 'accuracy', 'f1_score', 'auc']].to_string(index=False))

    print("\n Secondary metrics (validation-calibrated threshold)")
    secondary = df[df['threshold_type'] == "val_calibrated"]
    print(secondary[['model', 'test_set', 'threshold',
                      'accuracy', 'f1_score', 'auc']].to_string(index=False))

    print("\n Method breakdown — primary (fixed 0.5)")
    print(primary[['model', 'test_set'] + method_cols].to_string(index=False))

    print(f"\nSuccess: logs saved to {log_file}")


if __name__ == "__main__":
    LQ_WEIGHT = "results/training/weights/efficientnet_b0/baseline_lq/baseline_lq_20260604_012604.pth"

    run_evaluation("configs/base_config.yaml", LQ_WEIGHT)
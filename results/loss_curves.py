"""
Loss curve plotter

Usage (from repo root):
    python -m results.loss_curves \
        --baseline    results/training/logs/efficientnet_b0/baseline/logs_20260603_055220.csv \
        --novel       results/training/logs/efficientnet_b0/novel/logs_20260603_055220.csv \
        --baseline_lq results/training/logs/efficientnet_b0/baseline_lq/logs_20260604_012604.csv

Each argument is optional — pass only the ones you have.
Plots are saved to results/plots/.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

OUT_DIR = "results/plots"

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GRAY   = "#8C8C8C"


def stem(csv_path):
    """
    extracts the timestamp from a log filename.
    works for both logs_<ts>.csv and baseline_lq_logs_<ts>.csv by
    taking the last two underscore-separated tokens (YYYYMMDD_HHMMSS).
    """
    base  = os.path.splitext(os.path.basename(csv_path))[0]
    parts = base.split('_')
    return f"{parts[-2]}_{parts[-1]}"


def add_warmup_band(ax, df, label=True):
    """shades epochs where pull and push are both zero (frozen warmup)."""
    if "train_pull" not in df.columns:
        return
    warmup_mask   = (df["train_pull"] == 0) & (df["train_push"] == 0)
    warmup_epochs = df.loc[warmup_mask, "epoch"]
    if warmup_epochs.empty:
        return
    last_warmup = warmup_epochs.max()
    ax.axvspan(0.5, last_warmup + 0.5, alpha=0.07, color=GRAY,
               label="warmup" if label else None)
    ax.axvline(last_warmup + 0.5, color=GRAY, linewidth=0.8, linestyle="--")


# ── shared plotter for logs with epoch / train_loss / val_loss ───────────────

def plot_simple(csv_path, title_label, out_prefix):
    """
    plots train vs val loss for baseline or baseline_lq logs — same format,
    same layout, different title and output filename.
    """
    df       = pd.read_csv(csv_path)
    ts       = stem(csv_path)
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_{ts}.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs  = df["epoch"]

    ax.plot(epochs, df["train_loss"], color=BLUE,   linewidth=1.8, label="Train loss")
    ax.plot(epochs, df["val_loss"],   color=ORANGE, linewidth=1.8,
            linestyle="--", label="Val loss")

    best_idx = df["val_loss"].idxmin()
    best_ep  = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_loss"]
    ax.scatter(best_ep, best_val, color=ORANGE, zorder=5, s=60)
    ax.annotate(f"  best val\n  ep {best_ep}  ({best_val:.4f})",
                xy=(best_ep, best_val), fontsize=8, color=ORANGE)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss",  fontsize=11)
    ax.set_title(f"{title_label} — Training & Validation Loss",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
    ax.grid(True, alpha=0.3)

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else ""
    fig.text(0.5, 0.01, f"dataset: {dataset}   |   run: {ts}",
             ha="center", fontsize=8, color=GRAY)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_baseline(csv_path):
    plot_simple(csv_path, title_label="Baseline", out_prefix="baseline")


def plot_baseline_lq(csv_path):
    plot_simple(csv_path,
                title_label="Baseline LQ (trained on compressed images)",
                out_prefix="baseline_lq")


# ── novel plotter — two-panel with component breakdown ───────────────────────

def plot_novel(csv_path):
    df       = pd.read_csv(csv_path)
    ts       = stem(csv_path)
    out_path = os.path.join(OUT_DIR, f"novel_{ts}.png")

    fig    = plt.figure(figsize=(10, 8))
    gs     = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4], hspace=0.38)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    epochs = df["epoch"]

    # top panel: total loss
    ax_top.plot(epochs, df["train_loss"], color=BLUE,   linewidth=1.8,
                label="Train loss (total)")
    ax_top.plot(epochs, df["val_loss"],   color=ORANGE, linewidth=1.8,
                linestyle="--", label="Val loss (total)")

    best_idx = df["val_loss"].idxmin()
    best_ep  = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_loss"]
    ax_top.scatter(best_ep, best_val, color=ORANGE, zorder=5, s=60)
    ax_top.annotate(f"  best val\n  ep {best_ep}  ({best_val:.4f})",
                    xy=(best_ep, best_val), fontsize=8, color=ORANGE)

    add_warmup_band(ax_top, df)
    ax_top.set_xlabel("Epoch", fontsize=10)
    ax_top.set_ylabel("Loss",  fontsize=10)
    ax_top.set_title("Novel — Total Loss", fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=8)
    ax_top.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
    ax_top.grid(True, alpha=0.3)

    # bottom panel: component breakdown
    components = [
        ("train_bce",  "val_bce",  "BCE",  BLUE,  BLUE),
        ("train_pull", "val_pull", "Pull", GREEN, GREEN),
        ("train_push", "val_push", "Push", RED,   RED),
    ]
    for train_col, val_col, label, t_col, v_col in components:
        if train_col not in df.columns:
            continue
        ax_bot.plot(epochs, df[train_col], color=t_col, linewidth=1.5,
                    label=f"Train {label}")
        ax_bot.plot(epochs, df[val_col],   color=v_col, linewidth=1.5,
                    linestyle="--", label=f"Val {label}", alpha=0.75)

    add_warmup_band(ax_bot, df, label=False)
    ax_bot.set_xlabel("Epoch", fontsize=10)
    ax_bot.set_ylabel("Loss",  fontsize=10)
    ax_bot.set_title("Novel — Loss Component Breakdown",
                     fontsize=12, fontweight="bold")
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
    ax_bot.grid(True, alpha=0.3)

    warmup_patch = Patch(facecolor=GRAY, alpha=0.15, label="warmup phase")
    for ax in [ax_top, ax_bot]:
        handles, labels = ax.get_legend_handles_labels()
        if "warmup" not in labels:
            handles.append(warmup_patch)
        ax.legend(handles=handles, fontsize=8, ncol=2)

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else ""
    fig.text(0.5, 0.005, f"dataset: {dataset}   |   run: {ts}",
             ha="center", fontsize=8, color=GRAY)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves.")
    parser.add_argument("--baseline",    type=str, default=None,
                        help="Path to baseline training log CSV")
    parser.add_argument("--novel",       type=str, default=None,
                        help="Path to novel training log CSV")
    parser.add_argument("--baseline_lq", type=str, default=None,
                        help="Path to baseline_lq training log CSV")
    args = parser.parse_args()

    if not any([args.baseline, args.novel, args.baseline_lq]):
        parser.error("Provide at least one of --baseline, --novel, --baseline_lq.")

    for path, fn in [
        (args.baseline,    plot_baseline),
        (args.novel,       plot_novel),
        (args.baseline_lq, plot_baseline_lq),
    ]:
        if path is None:
            continue
        if not os.path.isfile(path):
            print(f"Error: log not found: {path}")
        else:
            fn(path)


if __name__ == "__main__":
    main()
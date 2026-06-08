import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ARCH_RESULTS = {
    "Xception":        "results/testing/logs/xception/test_results_20260607_114948.csv",
    "DenseNet121":     "results/testing/logs/densenet121/test_results_20260607_163723.csv",
}

OUT_DIR = "results/plots"

RAW_COLOR  = "#4C72B0"
COMP_COLOR = "#DD8452"

def load_metric(csv_path, model_name, test_set, metric):
    """
    pull a single metric (accuracy or auc) for a given model and test set
    from the fixed_0.5 threshold row of a test-results CSV.
    """
    df = pd.read_csv(csv_path)
    row = df[
        (df["model"].str.lower() == model_name.lower())
        & (df["test_set"].str.lower() == test_set.lower())
        & (df["threshold_type"] == "fixed_0.5")
    ]
    if row.empty:
        raise ValueError(
            f"No '{model_name}' / '{test_set}' / fixed_0.5 row found in {csv_path}"
        )
    return float(row.iloc[0][metric])


def grouped_bar(arch_names, raw_vals, comp_vals, ylabel, title, out_path,
                as_percent=True, ymax=None):
    """
    draws a grouped bar chart: one pair of bars (RAW, Compressed) per architecture.
    """
    x = np.arange(len(arch_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.8))

    scale = 100.0 if as_percent else 1.0
    raw_scaled = [v * scale for v in raw_vals]
    comp_scaled = [v * scale for v in comp_vals]

    bars_raw = ax.bar(x - width / 2, raw_scaled, width,
                      label="RAW (c23)", color=RAW_COLOR)
    bars_comp = ax.bar(x + width / 2, comp_scaled, width,
                       label="Compressed", color=COMP_COLOR)

    # value labels on each bar
    for bars in (bars_raw, bars_comp):
        for b in bars:
            h = b.get_height()
            label = f"{h:.1f}" if as_percent else f"{h:.3f}"
            ax.annotate(label,
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    arch_names = list(ARCH_RESULTS.keys())

    # Baseline accuracy gap (RAW vs Compressed)
    base_raw, base_comp = [], []
    for name, path in ARCH_RESULTS.items():
        base_raw.append(load_metric(path, "Baseline", "RAW", "accuracy"))
        base_comp.append(load_metric(path, "Baseline", "Compressed", "accuracy"))

    grouped_bar(
        arch_names, base_raw, base_comp,
        ylabel="Accuracy (%)",
        title="Baseline accuracy: RAW vs compressed",
        out_path=os.path.join(OUT_DIR, "architecture_accuracy_gap.png"),
        as_percent=True, ymax=110,
    )

    # Novel model AUC (RAW vs Compressed)
    novel_raw_auc, novel_comp_auc = [], []
    for name, path in ARCH_RESULTS.items():
        novel_raw_auc.append(load_metric(path, "Novel", "RAW", "auc"))
        novel_comp_auc.append(load_metric(path, "Novel", "Compressed", "auc"))

    grouped_bar(
        arch_names, novel_raw_auc, novel_comp_auc,
        ylabel="AUC",
        title="Novel model AUC: RAW vs compressed",
        out_path=os.path.join(OUT_DIR, "architecture_novel_auc.png"),
        as_percent=False, ymax=1.0,
    )


if __name__ == "__main__":
    main()
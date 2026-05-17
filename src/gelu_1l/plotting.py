from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gelu_1l.artifacts import ensure_dir


def plot_activation_histogram(histogram: pd.DataFrame, output_path: Path, feature_id: int) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(histogram["bin_left"], histogram["count"], width=histogram["bin_right"] - histogram["bin_left"], align="edge")
    ax.set_title(f"Feature {feature_id} activation histogram")
    ax.set_xlabel("Activation")
    ax.set_ylabel("Token count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_validation_summary(summary: pd.DataFrame, output_path: Path, feature_id: int) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary))
    ax.bar([i - 0.18 for i in x], summary["ablation_delta_loss"], width=0.36, label="ablation - original")
    ax.bar([i + 0.18 for i in x], summary["boost_delta_loss"], width=0.36, label="boost - original")
    ax.set_xticks(list(x), summary["group"])
    ax.set_title(f"Feature {feature_id} causal intervention")
    ax.set_ylabel("Mean loss delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

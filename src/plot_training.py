"""Plot classifier training curves."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSIFIER_ACCURACY_PLOT, CLASSIFIER_HISTORY_PATH, CLASSIFIER_LOSS_PLOT, PLOTS_DIR


def plot_training() -> None:
    if not CLASSIFIER_HISTORY_PATH.is_file():
        raise FileNotFoundError(f"Missing {CLASSIFIER_HISTORY_PATH}; train classifier first.")
    df = pd.read_csv(CLASSIFIER_HISTORY_PATH)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Classifier loss (pattern-family proxy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(CLASSIFIER_LOSS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_accuracy"], label="train")
    ax.plot(df["epoch"], df["val_accuracy"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pattern-family accuracy")
    ax.set_title("Classifier accuracy (pattern-family proxy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(CLASSIFIER_ACCURACY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    plot_training()
    print(f"Saved {CLASSIFIER_LOSS_PLOT} and {CLASSIFIER_ACCURACY_PLOT}")


if __name__ == "__main__":
    main()

"""
Visualize leakage-safe split distributions and sample grids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HANDLOOM_UNLABELED_PATH,
    PLOTS_DIR,
    PROJECT_ROOT,
    SAMPLE_GRID_PLOT,
    SPLIT_DISTRIBUTION_PLOT,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)


def load_splits() -> pd.DataFrame:
    frames = []
    for path, split in (
        (TRAIN_MANIFEST_PATH, "train"),
        (VAL_MANIFEST_PATH, "val"),
        (TEST_MANIFEST_PATH, "test"),
    ):
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_split_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    split_counts = df["split"].value_counts().reindex(["train", "val", "test"])
    axes[0].bar(split_counts.index.astype(str), split_counts.values, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Split counts")
    axes[0].set_ylabel("Images")

    fam = pd.crosstab(df["pattern_family"], df["split"])
    fam = fam.reindex(index=["Banarasi", "Bandhani", "Ikat", "Pichwai"], columns=["train", "val", "test"])
    fam.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Pattern family by split")
    axes[1].set_ylabel("Images")

    src = pd.crosstab(df["dataset_source"], df["split"])
    src.plot(kind="bar", ax=axes[2], rot=0)
    axes[2].set_title("Dataset source by split")
    axes[2].set_ylabel("Images")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPLIT_DISTRIBUTION_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, per_family: int = 4) -> None:
    families = ["Banarasi", "Bandhani", "Ikat", "Pichwai"]
    fig, axes = plt.subplots(len(families), per_family, figsize=(10, 8))
    fig.suptitle("Random samples by pattern family (proxy label)", fontsize=12)

    for r, fam in enumerate(families):
        subset = df[df["pattern_family"] == fam].sample(n=min(per_family, (df["pattern_family"] == fam).sum()), random_state=42)
        for c in range(per_family):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(fam, rotation=0, labelpad=36, va="center")
            if c >= len(subset):
                continue
            path = PROJECT_ROOT / subset.iloc[c]["image_path"]
            try:
                with Image.open(path) as im:
                    ax.imshow(im.convert("RGB"))
            except OSError:
                ax.text(0.5, 0.5, "error", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(SAMPLE_GRID_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    df = load_splits()
    plot_split_distribution(df)
    plot_sample_grid(df)
    handloom_n = len(pd.read_csv(HANDLOOM_UNLABELED_PATH)) if HANDLOOM_UNLABELED_PATH.is_file() else 0
    print(f"Saved {SPLIT_DISTRIBUTION_PLOT}")
    print(f"Saved {SAMPLE_GRID_PLOT}")
    print(f"Handloom unlabeled (not in split plots): {handloom_n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset splits")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

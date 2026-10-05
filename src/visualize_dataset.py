"""
Visual dataset report: contact sheet by pattern family and dimension overview.
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
    CLEAN_MANIFEST_PATH,
    DATASET_OVERVIEW_PLOT,
    DIMENSION_PLOT,
    MASTER_DATASET_PATH,
    PLOTS_DIR,
    PROJECT_ROOT,
)

THUMB = 128
SAMPLES_PER_GROUP = 6


def load_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except OSError:
        return None


def make_contact_sheet(manifest: pd.DataFrame) -> None:
    groups = [
        ("Banarasi", manifest["pattern_family"] == "Banarasi"),
        ("Bandhani", manifest["pattern_family"] == "Bandhani"),
        ("Ikat", manifest["pattern_family"] == "Ikat"),
        ("Pichwai", manifest["pattern_family"] == "Pichwai"),
        ("Handloom", manifest["dataset_source"] == "handloom"),
    ]

    fig, axes = plt.subplots(len(groups), SAMPLES_PER_GROUP, figsize=(14, 10))
    fig.suptitle("Dataset overview — representative samples", fontsize=14)

    for row_idx, (title, mask) in enumerate(groups):
        subset = manifest[mask].sort_values("image_path").head(SAMPLES_PER_GROUP)
        for col in range(SAMPLES_PER_GROUP):
            ax = axes[row_idx, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(title, fontsize=10, rotation=0, labelpad=40, va="center")
            if col >= len(subset):
                continue
            rec = subset.iloc[col]
            img_path = PROJECT_ROOT / rec["image_path"]
            img = load_image(img_path)
            if img is None:
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
                continue
            img = img.resize((THUMB, THUMB))
            ax.imshow(img)
            if col == 0:
                name = Path(rec["image_path"]).name
                short = name[:18] + "..." if len(name) > 18 else name
                ax.set_title(short, fontsize=7)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DATASET_OVERVIEW_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_dimension_plot(manifest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, source, title in zip(
        axes,
        ["indian_patterns", "handloom"],
        ["Indian patterns (640×640 export)", "Handloom / DeepLure"],
    ):
        sub = manifest[manifest["dataset_source"] == source]
        ax.scatter(sub["width"], sub["height"], alpha=0.5, s=12)
        ax.set_xlabel("width (px)")
        ax.set_ylabel("height (px)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(DIMENSION_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    path = CLEAN_MANIFEST_PATH if CLEAN_MANIFEST_PATH.is_file() else MASTER_DATASET_PATH
    manifest = pd.read_csv(path)
    make_contact_sheet(manifest)
    make_dimension_plot(manifest)
    print(f"Saved {DATASET_OVERVIEW_PLOT}")
    print(f"Saved {DIMENSION_PLOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual dataset report")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

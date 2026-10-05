"""PCA/t-SNE visualization of test embeddings (pattern-family proxy labels)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EMBEDDINGS_DIR,
    LABEL_TO_PATTERN_FAMILY,
    PLOTS_DIR,
    TEST_EMBEDDINGS_PCA_PLOT,
    TEST_EMBEDDINGS_TSNE_PLOT,
)


def plot_pca(embs: np.ndarray, labels: np.ndarray) -> None:
    from sklearn.decomposition import PCA

    proj = PCA(n_components=2, random_state=42).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in sorted(set(labels)):
        mask = labels == lab
        name = LABEL_TO_PATTERN_FAMILY.get(int(lab), str(lab))
        ax.scatter(proj[mask, 0], proj[mask, 1], s=12, alpha=0.7, label=name)
    ax.set_title("Test embeddings PCA (pattern family — proxy label)")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TEST_EMBEDDINGS_PCA_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(embs: np.ndarray, labels: np.ndarray, max_samples: int = 400) -> None:
    from sklearn.manifold import TSNE

    n = embs.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        embs = embs[idx]
        labels = labels[idx]
    proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embs) - 1)).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in sorted(set(labels)):
        mask = labels == lab
        name = LABEL_TO_PATTERN_FAMILY.get(int(lab), str(lab))
        ax.scatter(proj[mask, 0], proj[mask, 1], s=12, alpha=0.7, label=name)
    ax.set_title("Test embeddings t-SNE (pattern family — proxy label)")
    ax.legend(markerscale=2, fontsize=8)
    fig.savefig(TEST_EMBEDDINGS_TSNE_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_visualization() -> None:
    embs = np.load(EMBEDDINGS_DIR / "test_embeddings.npy")
    meta = pd.read_csv(EMBEDDINGS_DIR / "test_metadata.csv")
    labels = meta["label"].values.astype(int)
    plot_pca(embs, labels)
    plot_tsne(embs, labels)
    print(f"Saved {TEST_EMBEDDINGS_PCA_PLOT}")
    print(f"Saved {TEST_EMBEDDINGS_TSNE_PLOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_visualization()


if __name__ == "__main__":
    main()

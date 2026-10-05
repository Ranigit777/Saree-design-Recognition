"""
Gallery index management for Saree Design Recognition.
Provides high-level indexing, querying, and persistent storage of gallery embeddings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    PROJECT_ROOT,
    TRAIN_MANIFEST_PATH,
)
from loaders import get_device
from retrieval import SareeRetrievalEngine


class SareeGallery:
    """Manages an indexed gallery of saree design embeddings."""

    def __init__(
        self,
        embeddings_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        engine: SareeRetrievalEngine | None = None,
    ) -> None:
        emb_p = Path(embeddings_path) if embeddings_path else EMBEDDINGS_DIR / "train_embeddings.npy"
        meta_p = Path(metadata_path) if metadata_path else EMBEDDINGS_DIR / "train_metadata.csv"

        self.engine = engine or SareeRetrievalEngine(gallery_emb_path=emb_p, gallery_meta_path=meta_p)
        self.embeddings = self.engine.gallery_embeddings
        self.metadata = self.engine.gallery_metadata

    def __len__(self) -> int:
        return len(self.embeddings) if self.embeddings is not None else 0

    def search(
        self,
        query: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 5,
        exclude_query: bool = True,
    ) -> list[dict[str, Any]]:
        return self.engine.search(query, top_k=top_k, exclude_query_path=exclude_query)

    def get_summary(self) -> dict[str, Any]:
        if self.metadata is None or self.embeddings is None:
            return {"status": "empty"}
        return {
            "total_items": len(self.embeddings),
            "embedding_dimension": int(self.embeddings.shape[1]),
            "pattern_family_counts": self.metadata["pattern_family"].value_counts().to_dict(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Saree Gallery CLI")
    parser.add_argument("--info", action="store_true", help="Print gallery summary info")
    parser.add_argument("--query", type=str, default=None, help="Image path to query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    gallery = SareeGallery()
    if args.info or not args.query:
        print("\nSaree Gallery Summary:")
        print(json.dumps(gallery.get_summary(), indent=2))
        return

    results = gallery.search(args.query, top_k=args.top_k)
    print(f"\nTop {args.top_k} results for: {args.query}")
    print("-" * 65)
    for r in results:
        print(f"Rank {r['rank']} | Sim: {r['similarity']:.4f} | Family: {r['pattern_family']:<10} | Path: {r['image_path']}")
    print("-" * 65)


if __name__ == "__main__":
    main()

"""
Gallery/query retrieval for saree images using cosine similarity on L2-normalized embeddings.

Proxy supervision only — retrieves by proxy pattern family; NOT fine-grained design IDs.
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
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform
from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    LABEL_TO_PATTERN_FAMILY,
    PATTERN_FAMILY_TO_LABEL,
    PROJECT_ROOT,
    RETRIEVAL_METRICS_PATH,
    RETRIEVAL_RESULTS_PATH,
    RESULTS_DIR,
)
from loaders import get_device
from metrics_utils import pattern_family_retrieval_metrics
from model import SareeEmbeddingModel


def retrieve_top_k(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_paths: list[str],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Rank gallery by cosine similarity to query embedding."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    q = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
    g = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = (q @ g.T).flatten()
    order = np.argsort(-sim)[:k]
    return [(gallery_paths[i], float(sim[i])) for i in order]


def evaluate_retrieval(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    query_paths: list[str],
    gallery_emb: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_paths: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Evaluate retrieval metrics (Recall@K, MRR) across query set."""
    g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8)
    q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    sim = q @ g.T
    # mask exact same image path if present in gallery
    for i, qp in enumerate(query_paths):
        for j, gp in enumerate(gallery_paths):
            if qp == gp:
                sim[i, j] = -np.inf
    metrics = pattern_family_retrieval_metrics(query_labels, gallery_labels, sim, ks=ks)
    for k in ks:
        key = f"pattern_family_recall_at_{k}"
        if key in metrics:
            metrics[f"recall_at_{k}"] = metrics[key]
    return metrics


class SareeRetrievalEngine:
    """
    End-to-end retrieval engine:
    Given a query saree image -> extract embedding -> search gallery -> rank by cosine similarity.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        gallery_emb_path: Path | str | None = None,
        gallery_meta_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.transform = build_eval_transform()
        self.model = self._load_model(model_path)
        self.gallery_embeddings: np.ndarray | None = None
        self.gallery_metadata: pd.DataFrame | None = None

        emb_p = Path(gallery_emb_path) if gallery_emb_path else EMBEDDINGS_DIR / "train_embeddings.npy"
        meta_p = Path(gallery_meta_path) if gallery_meta_path else EMBEDDINGS_DIR / "train_metadata.csv"
        if emb_p.is_file() and meta_p.is_file():
            self.load_gallery(emb_p, meta_p)

    def _load_model(self, model_path: Path | str | None) -> SareeEmbeddingModel:
        p = Path(model_path) if model_path else BEST_METRIC_MODEL_PATH
        if not p.is_file():
            raise FileNotFoundError(f"Metric checkpoint not found at: {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        emb_dim = int(cfg.get("embedding_dim", 256))
        dropout = float(cfg.get("dropout", 0.2))
        model = SareeEmbeddingModel(
            embedding_dim=emb_dim,
            pretrained=False,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def load_gallery(self, emb_path: Path, meta_path: Path) -> None:
        self.gallery_embeddings = np.load(emb_path)
        # Normalize gallery
        norms = np.linalg.norm(self.gallery_embeddings, axis=1, keepdims=True) + 1e-8
        self.gallery_embeddings = self.gallery_embeddings / norms
        self.gallery_metadata = pd.read_csv(meta_path)

    @torch.no_grad()
    def extract_embedding(
        self, image_input: Union[str, Path, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Extract 256-D L2-normalized embedding for an image."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.is_file():
                raise FileNotFoundError(f"Query image does not exist: {p}")
            try:
                with Image.open(p) as im:
                    tensor = self.transform(im.convert("RGB")).unsqueeze(0).to(self.device)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Invalid image file at {p}") from exc
        elif isinstance(image_input, Image.Image):
            tensor = self.transform(image_input.convert("RGB")).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 3:
                tensor = image_input.unsqueeze(0).to(self.device)
            else:
                tensor = image_input.to(self.device)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        out = self.model(tensor, return_logits=False)
        emb = out["embedding"].cpu().numpy()
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norm).astype(np.float32)

    def search(
        self,
        query: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
        top_k: int = 5,
        exclude_query_path: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search gallery for best matches to query.
        Returns ranked list of dictionaries with rank, image_path, pattern_family, similarity.
        """
        if self.gallery_embeddings is None or self.gallery_metadata is None:
            raise RuntimeError("Gallery has not been loaded. Call load_gallery first.")

        if isinstance(query, np.ndarray):
            q_emb = query.reshape(1, -1)
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
            query_path_str = None
        else:
            q_emb = self.extract_embedding(query)
            query_path_str = str(query) if isinstance(query, (str, Path)) else None

        sims = (q_emb @ self.gallery_embeddings.T).flatten()

        # Optionally mask out query if it is in gallery
        if exclude_query_path and query_path_str:
            for j, p in enumerate(self.gallery_metadata["image_path"]):
                if str(p) == query_path_str or Path(p).name == Path(query_path_str).name:
                    sims[j] = -np.inf

        top_indices = np.argsort(-sims)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.gallery_metadata.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "image_path": str(row["image_path"]),
                    "pattern_family": row.get("pattern_family", "unknown"),
                    "similarity": round(float(sims[idx]), 4),
                    "original_split": row.get("original_split", ""),
                    "dataset_source": row.get("dataset_source", ""),
                }
            )
        return results


def run_retrieval_evaluation() -> dict:
    test_emb = np.load(EMBEDDINGS_DIR / "test_embeddings.npy")
    train_emb = np.load(EMBEDDINGS_DIR / "train_embeddings.npy")
    test_meta = pd.read_csv(EMBEDDINGS_DIR / "test_metadata.csv")
    train_meta = pd.read_csv(EMBEDDINGS_DIR / "train_metadata.csv")

    q_labels = (
        test_meta["label"].values
        if "label" in test_meta
        else test_meta["pattern_family"].map(PATTERN_FAMILY_TO_LABEL).values
    )
    g_labels = (
        train_meta["label"].values
        if "label" in train_meta
        else train_meta["pattern_family"].map(PATTERN_FAMILY_TO_LABEL).values
    )

    metrics = evaluate_retrieval(
        test_emb,
        q_labels,
        test_meta["image_path"].tolist(),
        train_emb,
        g_labels,
        train_meta["image_path"].tolist(),
        ks=(1, 3, 5, 10),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results = {
        "task": "Pattern-Family Proxy Retrieval",
        "status": "completed",
        "protocol": "test_queries_vs_train_gallery",
        "num_test_queries": int(len(test_meta)),
        "gallery_size": int(len(train_meta)),
        "query_self_exclusion": True,
        "similarity_metric": "cosine_similarity",
        "proxy_labels_used": ["Banarasi", "Bandhani", "Ikat", "Pichwai"],
        "metrics": {
            "Recall@1": metrics["recall_at_1"],
            "Recall@3": metrics["recall_at_3"],
            "Recall@5": metrics["recall_at_5"],
            "Recall@10": metrics["recall_at_10"],
            "MRR": metrics["mrr"],
        },
        "scientific_disclaimer": (
            "These metrics represent Pattern-Family Proxy Retrieval across 4 coarse style families. "
            "They do NOT represent fine-grained saree design recognition or authentic CASE color invariance, "
            "because the dataset lacks verified design IDs and ground-truth colorway annotations."
        ),
    }
    with open(RETRIEVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Saree retrieval CLI")
    parser.add_argument("--query", type=str, default=None, help="Path to query image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches")
    parser.add_argument("--gallery-emb", type=str, default=None, help="Gallery embeddings .npy")
    parser.add_argument("--gallery-meta", type=str, default=None, help="Gallery metadata .csv")
    parser.add_argument("--evaluate", action="store_true", help="Run test retrieval evaluation")
    args = parser.parse_args()

    if args.evaluate or (not args.query and not args.evaluate):
        metrics = run_retrieval_evaluation()
        print(json.dumps(metrics, indent=2))
        return

    engine = SareeRetrievalEngine(
        gallery_emb_path=args.gallery_emb,
        gallery_meta_path=args.gallery_meta,
    )
    results = engine.search(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query}")
    print(f"Top {args.top_k} Retrieval Results:")
    print("-" * 65)
    for res in results:
        print(
            f"Rank {res['rank']} | Sim: {res['similarity']:.4f} | "
            f"Family: {res['pattern_family']:<10} | Path: {res['image_path']}"
        )
    print("-" * 65)


if __name__ == "__main__":
    main()
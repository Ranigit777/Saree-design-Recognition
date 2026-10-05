"""Classification and retrieval metrics (pattern-family proxy task)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_prec, per_rec, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3]
    )
    return {
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "per_class_precision": per_prec.tolist(),
        "per_class_recall": per_rec.tolist(),
        "per_class_f1": per_f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Banarasi", "Bandhani", "Ikat", "Pichwai"], zero_division=0
        ),
    }


def pattern_family_retrieval_metrics(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    similarity: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 4, 5, 10),
) -> dict:
    """
    similarity: [num_queries, num_gallery] cosine scores (higher is better).
    Excludes self-matches when gallery paths equal query paths via diagonal mask externally.
    """
    n_q = similarity.shape[0]
    ranks: list[int] = []
    recall_hits = {k: 0 for k in ks}
    mrr_sum = 0.0

    for i in range(n_q):
        scores = similarity[i].copy()
        order = np.argsort(-scores)
        gt = query_labels[i]
        rank = None
        for r, j in enumerate(order, start=1):
            if gallery_labels[j] == gt:
                rank = r
                break
        if rank is None:
            continue
        ranks.append(rank)
        mrr_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1

    result = {
        "task": "pattern_family_proxy_retrieval",
        "num_queries": int(n_q),
        "mrr": float(mrr_sum / n_q) if n_q else 0.0,
    }
    for k in ks:
        result[f"pattern_family_recall_at_{k}"] = float(recall_hits[k] / n_q) if n_q else 0.0
    return result

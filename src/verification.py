"""
Verification module for saree images (proxy pattern-family verification).

Proxy supervision only — tests whether two images share the same pattern family;
NOT fine-grained design verification.

Threshold selection is performed strictly on VALIDATION data.
The chosen threshold is then evaluated frozen on the TEST set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform
from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TEST_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    VERIFICATION_METRICS_PATH,
    VERIFICATION_RESULTS_PATH,
    VERIFICATION_ROC_PLOT,
    VERIFICATION_THRESHOLD_PATH,
    VERIFICATION_THRESHOLD_PLOT,
)
from loaders import get_device
from model import SareeEmbeddingModel


def generate_pairs_from_manifest(
    manifest_csv: Path | str,
    embeddings_npy: Path | str,
    num_pairs_per_class: int = 250,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate balanced positive and negative image pairs.
    Prevents data leakage by ensuring augmentation siblings / exact duplicates
    are never paired together.

    Returns:
        emb_pairs_1: [N, D] embeddings for first image
        emb_pairs_2: [N, D] embeddings for second image
        pair_metadata: DataFrame with details for each pair (pair_type, is_same, img1, img2, etc.)
    """
    df = pd.read_csv(manifest_csv).reset_index(drop=True)
    embeddings = np.load(embeddings_npy)
    # Ensure L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    rng = np.random.default_rng(seed)
    n = len(df)
    families = sorted(df["pattern_family"].unique())

    family_indices = {fam: df.index[df["pattern_family"] == fam].tolist() for fam in families}

    pairs_rows = []
    idx_pairs_pos = []
    idx_pairs_neg = []

    # 1. Positive pairs: same family, DIFFERENT augmentation_group and duplicate_group
    for fam, indices in family_indices.items():
        count = 0
        attempts = 0
        max_attempts = len(indices) * len(indices) * 2
        while count < num_pairs_per_class and attempts < max_attempts:
            attempts += 1
            i1, i2 = rng.choice(indices, size=2, replace=False)
            row1 = df.iloc[i1]
            row2 = df.iloc[i2]

            # Check augmentation and duplicate leakage
            aug1 = row1.get("augmentation_group")
            aug2 = row2.get("augmentation_group")
            if pd.notna(aug1) and pd.notna(aug2) and aug1 == aug2:
                continue

            dup1 = row1.get("duplicate_group")
            dup2 = row2.get("duplicate_group")
            if pd.notna(dup1) and pd.notna(dup2) and dup1 == dup2:
                continue

            idx_pairs_pos.append((i1, i2))
            pairs_rows.append(
                {
                    "idx1": i1,
                    "idx2": i2,
                    "image1": row1["image_path"],
                    "image2": row2["image_path"],
                    "family1": fam,
                    "family2": fam,
                    "is_same": 1,
                    "pair_type": "positive",
                }
            )
            count += 1

    total_pos = len(idx_pairs_pos)

    # 2. Negative pairs: different families
    neg_count = 0
    attempts = 0
    max_attempts = total_pos * 10
    while neg_count < total_pos and attempts < max_attempts:
        attempts += 1
        fam1, fam2 = rng.choice(families, size=2, replace=False)
        i1 = rng.choice(family_indices[fam1])
        i2 = rng.choice(family_indices[fam2])
        row1 = df.iloc[i1]
        row2 = df.iloc[i2]

        idx_pairs_neg.append((i1, i2))
        pairs_rows.append(
            {
                "idx1": i1,
                "idx2": i2,
                "image1": row1["image_path"],
                "image2": row2["image_path"],
                "family1": fam1,
                "family2": fam2,
                "is_same": 0,
                "pair_type": "negative",
            }
        )
        neg_count += 1

    all_pairs = idx_pairs_pos + idx_pairs_neg
    indices1 = [p[0] for p in all_pairs]
    indices2 = [p[1] for p in all_pairs]

    embs1 = embeddings[indices1]
    embs2 = embeddings[indices2]
    meta_df = pd.DataFrame(pairs_rows)
    return embs1, embs2, meta_df


def compute_pair_similarities(embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for paired L2-normalized embeddings."""
    return np.sum(embs1 * embs2, axis=1)


def select_verification_threshold(
    val_manifest: Path = VAL_MANIFEST_PATH,
    val_embeddings: Path = EMBEDDINGS_DIR / "val_embeddings.npy",
    num_thresholds: int = 400,
) -> dict[str, Any]:
    """
    Select optimal verification threshold using ONLY validation data.
    Sweeps cosine similarity thresholds and picks the one maximizing F1 score.
    Also calculates the Equal Error Rate (EER) threshold.
    """
    print("[Verification] Generating validation pairs (leakage-safe)...")
    embs1, embs2, meta_df = generate_pairs_from_manifest(
        val_manifest, val_embeddings, num_pairs_per_class=200, seed=42
    )
    sims = compute_pair_similarities(embs1, embs2)
    y_true = meta_df["is_same"].values

    thresholds = np.linspace(-0.5, 1.0, num_thresholds)
    best_thresh = 0.0
    best_f1 = -1.0
    best_acc = 0.0
    best_metrics = {}

    curve_data = []
    eer_thresh = 0.0
    min_diff = float("inf")

    for t in thresholds:
        y_pred = (sims >= t).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # FPR and FNR for EER
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        if abs(fpr - fnr) < min_diff:
            min_diff = abs(fpr - fnr)
            eer_thresh = float(t)

        curve_data.append(
            {
                "threshold": float(t),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_acc = acc
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
            }

    # Validation ROC-AUC
    val_auc = float(roc_auc_score(y_true, sims))

    # Plot threshold selection curve
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    thresh_arr = [c["threshold"] for c in curve_data]
    ax.plot(thresh_arr, [c["f1"] for c in curve_data], label="F1 Score", color="blue", lw=2)
    ax.plot(thresh_arr, [c["accuracy"] for c in curve_data], label="Accuracy", color="green", lw=1.5)
    ax.plot(thresh_arr, [c["precision"] for c in curve_data], label="Precision", color="orange", ls="--")
    ax.plot(thresh_arr, [c["recall"] for c in curve_data], label="Recall", color="purple", ls="--")
    ax.axvline(best_thresh, color="red", ls=":", label=f"Selected Threshold: {best_thresh:.3f}")
    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Validation Threshold Selection (Proxy Pattern Family)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(VERIFICATION_THRESHOLD_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    threshold_result = {
        "status": "success",
        "protocol": "validation_only_threshold_selection",
        "selected_threshold": round(best_thresh, 4),
        "selection_metric": "max_f1_score",
        "equal_error_rate_threshold": round(eer_thresh, 4),
        "validation_pairs_count": len(meta_df),
        "validation_positive_pairs": int((y_true == 1).sum()),
        "validation_negative_pairs": int((y_true == 0).sum()),
        "validation_metrics_at_selected_threshold": {
            "f1_score": round(best_f1, 4),
            "accuracy": round(best_acc, 4),
            "precision": round(best_metrics["precision"], 4),
            "recall": round(best_metrics["recall"], 4),
            "fpr": round(best_metrics["fpr"], 4),
            "fnr": round(best_metrics["fnr"], 4),
            "roc_auc": round(val_auc, 4),
        },
        "similarity_distribution": {
            "positive_mean": float(sims[y_true == 1].mean()),
            "positive_std": float(sims[y_true == 1].std()),
            "negative_mean": float(sims[y_true == 0].mean()),
            "negative_std": float(sims[y_true == 0].std()),
        },
        "note": "Selected threshold was derived EXCLUSIVELY on validation pairs; test labels were never touched.",
    }

    with open(VERIFICATION_THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(threshold_result, f, indent=2)

    return threshold_result


def evaluate_test_verification(
    threshold: float,
    test_manifest: Path = TEST_MANIFEST_PATH,
    test_embeddings: Path = EMBEDDINGS_DIR / "test_embeddings.npy",
) -> dict[str, Any]:
    """
    Evaluate the fixed validation-selected threshold on the test set.
    """
    print(f"[Verification] Evaluating fixed threshold ({threshold:.4f}) on TEST set...")
    embs1, embs2, meta_df = generate_pairs_from_manifest(
        test_manifest, test_embeddings, num_pairs_per_class=200, seed=42
    )
    sims = compute_pair_similarities(embs1, embs2)
    y_true = meta_df["is_same"].values
    y_pred = (sims >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, sims))
    pr_auc = float(average_precision_score(y_true, sims))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Plot ROC curve on Test set
    fpr_curve, tpr_curve, _ = roc_curve(y_true, sims)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"Test ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, ls="--", label="Random Chance")
    ax.scatter([fpr], [rec], color="red", s=50, zorder=5, label=f"Fixed Threshold ({threshold:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Test Verification ROC Curve (Pattern Family Proxy)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(VERIFICATION_ROC_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = {
        "task": "pattern_family_proxy_verification",
        "fixed_threshold_used": round(threshold, 4),
        "threshold_source": "validation_set_max_f1",
        "test_pairs_count": len(meta_df),
        "test_positive_pairs": int((y_true == 1).sum()),
        "test_negative_pairs": int((y_true == 0).sum()),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "specificity": round(specificity, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        },
        "similarity_distribution": {
            "positive_mean": float(sims[y_true == 1].mean()),
            "positive_std": float(sims[y_true == 1].std()),
            "negative_mean": float(sims[y_true == 0].mean()),
            "negative_std": float(sims[y_true == 0].std()),
        },
        "limitations": [
            "Pairs are formed by coarse pattern family (Banarasi, Bandhani, Ikat, Pichwai), NOT fine-grained designs.",
            "Verification predicts whether two sarees belong to the same proxy motif family.",
            "Color invariance cannot be verified with this proxy setup since authentic same-design/different-color pairs do not exist in the dataset.",
        ],
    }

    with open(VERIFICATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


class SareeVerifier:
    """
    Inference-ready verifier:
    Given two saree images, extract embeddings, compute cosine similarity,
    and predict whether they match using the validation-calibrated threshold.
    """

    def __init__(
        self,
        threshold: float | None = None,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.transform = build_eval_transform()
        self.model = self._load_model(model_path)

        if threshold is not None:
            self.threshold = float(threshold)
        elif VERIFICATION_THRESHOLD_PATH.is_file():
            with open(VERIFICATION_THRESHOLD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.threshold = float(data.get("selected_threshold", 0.5))
        else:
            self.threshold = 0.5

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

    @torch.no_grad()
    def _embed(self, image_input: Union[str, Path, Image.Image, torch.Tensor]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.is_file():
                raise FileNotFoundError(f"Image not found: {p}")
            try:
                with Image.open(p) as im:
                    tensor = self.transform(im.convert("RGB")).unsqueeze(0).to(self.device)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Corrupted or invalid image: {p}") from exc
        elif isinstance(image_input, Image.Image):
            tensor = self.transform(image_input.convert("RGB")).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input.unsqueeze(0).to(self.device) if image_input.ndim == 3 else image_input.to(self.device)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        out = self.model(tensor, return_logits=False)
        emb = out["embedding"].cpu().numpy()
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norm).flatten()

    def verify(
        self,
        image1: Union[str, Path, Image.Image, torch.Tensor],
        image2: Union[str, Path, Image.Image, torch.Tensor],
    ) -> dict[str, Any]:
        """
        Verify if image1 and image2 share the same proxy pattern family.
        """
        emb1 = self._embed(image1)
        emb2 = self._embed(image2)
        sim = float(np.dot(emb1, emb2))
        is_same = bool(sim >= self.threshold)
        diff = sim - self.threshold

        # Confidence: scaled margin from threshold
        confidence = float(min(1.0, max(0.0, 0.5 + diff / 0.5)))

        return {
            "is_same": is_same,
            "prediction": "MATCH" if is_same else "NO MATCH",
            "cosine_similarity": round(sim, 4),
            "threshold": round(self.threshold, 4),
            "confidence": round(confidence, 4),
            "margin_from_threshold": round(diff, 4),
        }


def run_verification_pipeline() -> dict[str, Any]:
    """Execute complete validation threshold selection and test evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresh_info = select_verification_threshold()
    test_info = evaluate_test_verification(thresh_info["selected_threshold"])
    val_m = thresh_info["validation_metrics_at_selected_threshold"]
    test_m = test_info["metrics"]
    combined = {
        "task": "Pattern-Family Proxy Verification",
        "status": "completed",
        "threshold_selection_protocol": "validation_set_only",
        "threshold_selection_source": "outputs/results/val_manifest.csv",
        "validation_results": {
            "best_threshold": thresh_info["selected_threshold"],
            "equal_error_rate_threshold": thresh_info["equal_error_rate_threshold"],
            "accuracy": val_m["accuracy"],
            "precision": val_m["precision"],
            "recall": val_m["recall"],
            "f1": val_m["f1_score"],
            "fpr": val_m["fpr"],
            "fnr": val_m["fnr"],
            "roc_auc": val_m["roc_auc"],
        },
        "test_results_with_fixed_threshold": {
            "fixed_threshold_used": test_info["fixed_threshold_used"],
            "total_test_pairs": test_info["test_pairs_count"],
            "positive_test_pairs": test_info["test_positive_pairs"],
            "negative_test_pairs": test_info["test_negative_pairs"],
            "accuracy": test_m["accuracy"],
            "precision": test_m["precision"],
            "recall": test_m["recall"],
            "f1": test_m["f1_score"],
            "specificity": test_m["specificity"],
            "fpr": test_m["false_positive_rate"],
            "fnr": test_m["false_negative_rate"],
            "roc_auc": test_m["roc_auc"],
            "pr_auc": test_m["pr_auc"],
            "confusion_matrix": {
                "true_positives": test_m["true_positives"],
                "true_negatives": test_m["true_negatives"],
                "false_positives": test_m["false_positives"],
                "false_negatives": test_m["false_negatives"],
            },
        },
        "pair_definitions": {
            "SAME": "Images belong to the same proxy pattern family (e.g. Banarasi-Banarasi)",
            "DIFFERENT": "Images belong to different proxy pattern families (e.g. Banarasi-Ikat)",
        },
        "scientific_disclaimer": (
            "Threshold was selected strictly on validation data (max F1) and evaluated unchanged on test pairs. "
            "This evaluation tests pattern-family proxy discrimination, NOT fine-grained design verification "
            "or authentic color-invariance, due to dataset labeling limitations."
        ),
    }
    with open(VERIFICATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    return {
        "threshold_selection": thresh_info,
        "test_verification": test_info,
        "combined_report": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verification CLI")
    parser.add_argument("--image1", type=str, default=None, help="Path to first saree image")
    parser.add_argument("--image2", type=str, default=None, help="Path to second saree image")
    parser.add_argument("--threshold", type=float, default=None, help="Custom similarity threshold")
    parser.add_argument("--run-evaluation", action="store_true", help="Run full threshold selection and test eval")
    args = parser.parse_args()

    if args.run_evaluation or (not args.image1 and not args.image2):
        res = run_verification_pipeline()
        print("\nVerification Evaluation Complete:")
        print(f"  Selected Validation Threshold: {res['threshold_selection']['selected_threshold']}")
        print(f"  Validation F1: {res['threshold_selection']['validation_metrics_at_selected_threshold']['f1_score']:.4f}")
        print(f"  Test Accuracy: {res['test_verification']['metrics']['accuracy']:.4f}")
        print(f"  Test F1:       {res['test_verification']['metrics']['f1_score']:.4f}")
        print(f"  Test ROC-AUC:  {res['test_verification']['metrics']['roc_auc']:.4f}")
        return

    if not args.image1 or not args.image2:
        parser.error("Both --image1 and --image2 are required for pairwise verification.")

    verifier = SareeVerifier(threshold=args.threshold)
    out = verifier.verify(args.image1, args.image2)
    print("\nPairwise Saree Verification Result:")
    print("-" * 45)
    print(f"Image 1:    {args.image1}")
    print(f"Image 2:    {args.image2}")
    print(f"Decision:   {out['prediction']}")
    print(f"Similarity: {out['cosine_similarity']:.4f}")
    print(f"Threshold:  {out['threshold']:.4f}")
    print(f"Confidence: {out['confidence']:.4f}")
    print("-" * 45)


if __name__ == "__main__":
    main()

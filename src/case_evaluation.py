"""
CASE (Color-Invariant Saree Design Recognition) Evaluation Framework.

Designed to accept future verified:
- fine-grained design IDs (design_id)
- verified colorway IDs (colorway_id)
- same-design / different-colorway pairs
- same-colorway / different-design distractors

Evaluates:
1. Cross-Colorway Same-Design Retrieval (Protocol 1)
2. Color-Distractor Discrimination (Protocol 2 - True CASE Test)
3. Hard-Pair Verification (Protocol 3)
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform
from config import (
    BEST_METRIC_MODEL_PATH,
    CASE_EVALUATION_PATH,
    CASE_EVALUATION_PROTOCOL_PATH,
    CASE_MOCK_METRICS_PATH,
    MASTER_DATASET_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from loaders import get_device
from model import SareeEmbeddingModel


REQUIRED_CASE_COLUMNS = {"image_path", "design_id", "colorway_id"}


def validate_case_manifest(df: pd.DataFrame) -> None:
    """Validate that the manifest meets the CASE specification."""
    missing = REQUIRED_CASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CASE manifest missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_CASE_COLUMNS}"
        )
    if df["design_id"].isna().any():
        raise ValueError("CASE manifest contains null design_id values.")
    if df["colorway_id"].isna().any():
        raise ValueError("CASE manifest contains null colorway_id values.")


def evaluate_cross_colorway_retrieval(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """
    Protocol 1: Cross-Colorway Same-Design Retrieval.
    For each query (design D_i, color C_a), rank all other gallery items.
    Valid targets are (D_i, C_b) where b != a (different colorway of same design).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embs = embeddings / norms
    sim_matrix = embs @ embs.T
    np.fill_diagonal(sim_matrix, -np.inf)

    n = len(df)
    hits = {k: 0 for k in top_ks}
    mrr_sum = 0.0
    valid_query_count = 0

    design_ids = df["design_id"].values
    color_ids = df["colorway_id"].values

    for i in range(n):
        q_design = design_ids[i]
        q_color = color_ids[i]

        # Valid targets: same design, DIFFERENT colorway
        target_mask = (design_ids == q_design) & (color_ids != q_color)
        if not target_mask.any():
            # Query has no cross-colorway counterpart
            continue

        valid_query_count += 1
        # Exclude same image
        scores = sim_matrix[i].copy()
        order = np.argsort(-scores)

        first_hit_rank = None
        for r, candidate_idx in enumerate(order, start=1):
            if target_mask[candidate_idx]:
                first_hit_rank = r
                break

        if first_hit_rank is not None:
            mrr_sum += 1.0 / first_hit_rank
            for k in top_ks:
                if first_hit_rank <= k:
                    hits[k] += 1

    results = {
        "valid_cross_colorway_queries": valid_query_count,
        "mrr": round(float(mrr_sum / valid_query_count), 4) if valid_query_count else 0.0,
    }
    for k in top_ks:
        results[f"recall_at_{k}"] = (
            round(float(hits[k] / valid_query_count), 4) if valid_query_count else 0.0
        )
    return results


def evaluate_color_distractor_discrimination(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Protocol 2: The Core CASE Discrimination Test.
    Given query: (Design D_i, Color C_a)
    Compares:
      S_match:   sim to (Design D_i, Color C_b)  [Same design, different color]
      S_distract: sim to (Design D_j, Color C_a)  [Different design, SAME color]

    A truly color-invariant system will score S_match > S_distract.
    If the system is color-dependent, S_distract > S_match because the colors match!
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embs = embeddings / norms
    rng = np.random.default_rng(seed)

    design_ids = df["design_id"].values
    color_ids = df["colorway_id"].values
    n = len(df)

    triplet_count = 0
    correct_discrimination = 0
    margins = []

    for i in range(n):
        q_d = design_ids[i]
        q_c = color_ids[i]

        # 1. Match candidates: same design, different color
        match_indices = np.where((design_ids == q_d) & (color_ids != q_c))[0]
        # 2. Distractor candidates: different design, SAME color
        distractor_indices = np.where((design_ids != q_d) & (color_ids == q_c))[0]

        if len(match_indices) == 0 or len(distractor_indices) == 0:
            continue

        # Sample up to 3 distractors per valid match
        for m_idx in match_indices:
            sample_dists = rng.choice(
                distractor_indices, size=min(3, len(distractor_indices)), replace=False
            )
            sim_match = float(np.dot(embs[i], embs[m_idx]))
            for d_idx in sample_dists:
                sim_distract = float(np.dot(embs[i], embs[d_idx]))
                triplet_count += 1
                diff = sim_match - sim_distract
                margins.append(diff)
                if diff > 0:
                    correct_discrimination += 1

    discrimination_rate = (
        float(correct_discrimination / triplet_count) if triplet_count > 0 else 0.0
    )
    mean_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "total_discrimination_triplets": triplet_count,
        "case_discrimination_rate": round(discrimination_rate, 4),
        "mean_design_over_color_margin": round(mean_margin, 4),
        "criterion": (
            "CASE Discrimination Rate measures the percentage of times the model correctly "
            "ranked the true design (in a different color) higher than an imposter design (in the exact same color)."
        ),
    }


def evaluate_hard_pair_verification(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Protocol 3: Verification on Hard Pairs.
    Positive pairs: Same design, different color (D_i, C_a) vs (D_i, C_b).
    Hard Negative pairs: Different design, SAME color (D_i, C_a) vs (D_j, C_a).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embs = embeddings / norms
    design_ids = df["design_id"].values
    color_ids = df["colorway_id"].values
    n = len(df)

    pairs_y = []
    pairs_sim = []

    # Positives
    for i in range(n):
        for j in range(i + 1, n):
            if design_ids[i] == design_ids[j] and color_ids[i] != color_ids[j]:
                pairs_y.append(1)
                pairs_sim.append(float(np.dot(embs[i], embs[j])))

    # Hard Negatives
    for i in range(n):
        for j in range(i + 1, n):
            if design_ids[i] != design_ids[j] and color_ids[i] == color_ids[j]:
                pairs_y.append(0)
                pairs_sim.append(float(np.dot(embs[i], embs[j])))

    if len(pairs_y) == 0 or sum(pairs_y) == 0 or sum(pairs_y) == len(pairs_y):
        return {
            "error": "Insufficient pairs to calculate verification metrics",
            "pairs_count": len(pairs_y),
        }

    y_true = np.array(pairs_y)
    sims = np.array(pairs_sim)
    y_pred = (sims >= threshold).astype(int)

    return {
        "total_pairs": len(y_true),
        "positive_pairs_diff_color": int((y_true == 1).sum()),
        "hard_negative_pairs_same_color": int((y_true == 0).sum()),
        "accuracy_at_threshold": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_at_threshold": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, sims)), 4),
    }


def run_case_evaluation(
    manifest_csv: Path | str,
    model_path: Path | str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run full CASE evaluation protocol on a verified manifest."""
    df = pd.read_csv(manifest_csv)
    validate_case_manifest(df)
    dev = device or get_device()
    p = Path(model_path) if model_path else BEST_METRIC_MODEL_PATH
    ckpt = torch.load(p, map_location=dev, weights_only=False)
    model = SareeEmbeddingModel(
        embedding_dim=ckpt.get("config", {}).get("embedding_dim", 256),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    tfm = build_eval_transform()
    embs_list = []
    for _, row in df.iterrows():
        p_img = PROJECT_ROOT / row["image_path"]
        with Image.open(p_img) as im:
            t = tfm(im.convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                out = model(t, return_logits=False)
                embs_list.append(out["embedding"].cpu().numpy())
    embs = np.concatenate(embs_list, axis=0)

    p1 = evaluate_cross_colorway_retrieval(df, embs)
    p2 = evaluate_color_distractor_discrimination(df, embs)
    p3 = evaluate_hard_pair_verification(df, embs)

    return {
        "task": "future_verified_case_evaluation",
        "dataset_rows": len(df),
        "protocol_1_cross_colorway_retrieval": p1,
        "protocol_2_color_distractor_discrimination": p2,
        "protocol_3_hard_pair_verification": p3,
    }


def write_case_unavailable_report() -> dict[str, Any]:
    """Official CASE status for the current project datasets (no verified GT)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "AIE-CASE Color-Invariant Saree Design Recognition",
        "CASE_STATUS": "not_available",
        "case_evaluation": "NOT AVAILABLE",
        "status": "not_available",
        "reason": "No verified design_id/color_id ground truth exists in the supplied datasets.",
        "supported_interface": {
            "expected_columns": [
                "image_a",
                "image_b",
                "design_id_a",
                "design_id_b",
                "color_id_a",
                "color_id_b",
            ],
            "alternate_manifest_columns": ["image_path", "design_id", "colorway_id"],
            "protocols": {
                "Protocol 1": "Cross-Colorway Same-Design Retrieval",
                "Protocol 2": "Color-Distractor Discrimination",
                "Protocol 3": "Hard-Pair Verification",
            },
        },
        "pipeline_readiness": (
            "src/case_evaluation.py can evaluate a verified CASE CSV when provided "
            "via --manifest. Mock schema tests are optional and must not be treated "
            "as CASE metrics on the real dataset."
        ),
        "metrics": "N/A",
    }
    if MASTER_DATASET_PATH.is_file():
        df = pd.read_csv(MASTER_DATASET_PATH)
        has_design = "design_id" in df.columns and df["design_id"].notna().any()
        report["dataset_audit"] = {
            "master_rows": int(len(df)),
            "verified_design_ids": bool(has_design),
            "verified_color_labels": False,
            "verified_same_design_different_color_pairs": False,
        }
    with open(CASE_EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
    """
    Synthetic mock validation test:
    Simulates a ground-truth verified dataset with 12 designs in 3 colorways
    to verify that the entire CASE framework runs without errors.
    """
    print("[CASE Framework] Running synthetic mock validation test...")
    rng = np.random.default_rng(42)
    n_designs = 12
    n_colors = 3
    dim = 256

    # Create synthetic orthogonal design prototypes and color biases
    design_prototypes = rng.normal(0, 1, (n_designs, dim))
    design_prototypes /= np.linalg.norm(design_prototypes, axis=1, keepdims=True)

    color_biases = rng.normal(0, 0.2, (n_colors, dim))

    rows = []
    mock_embs = []

    for d in range(n_designs):
        for c in range(n_colors):
            emb = design_prototypes[d] + color_biases[c] + rng.normal(0, 0.05, dim)
            emb /= np.linalg.norm(emb)
            rows.append(
                {
                    "image_path": f"mock_images/d{d}_c{c}.jpg",
                    "design_id": f"D{d:03d}",
                    "colorway_id": f"color_{c}",
                }
            )
            mock_embs.append(emb)

    mock_df = pd.DataFrame(rows)
    mock_embeddings = np.array(mock_embs)

    p1 = evaluate_cross_colorway_retrieval(mock_df, mock_embeddings)
    p2 = evaluate_color_distractor_discrimination(mock_df, mock_embeddings)
    p3 = evaluate_hard_pair_verification(mock_df, mock_embeddings)

    report = {
        "status": "verified_working",
        "framework": "CASE_Evaluation_Suite_v1",
        "mock_dataset_stats": {
            "num_images": len(mock_df),
            "num_unique_designs": n_designs,
            "num_unique_colorways": n_colors,
        },
        "protocol_1_cross_colorway_retrieval": p1,
        "protocol_2_color_distractor_discrimination": p2,
        "protocol_3_hard_pair_verification": p3,
        "note": "Framework successfully validated on mock data. Ready for future verified ground truth manifests.",
    }

    with open(CASE_MOCK_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def write_case_protocol_doc() -> None:
    doc = """# CASE Evaluation Protocol Specification

## Purpose
This document defines the exact evaluation protocols and data schema for **AIE-CASE (Color-Invariant Saree Design Recognition)** when verified fine-grained design and colorway ground truth becomes available.

---

## 1. Required Manifest Schema

Future verified datasets must be provided as a CSV file with the following columns:

| Column Name | Type | Description |
|---|---|---|
| `image_path` | string | Relative path to image within workspace |
| `design_id` | string/int | Verified unique identifier of the surface motif/design |
| `colorway_id` | string/int | Identifier of the specific color palette/dye combination |
| `split` | string | Optional: `train`, `val`, `test`, `gallery`, `query` |

---

## 2. Evaluation Protocols

### Protocol 1: Cross-Colorway Same-Design Retrieval
- **Query:** Image $(D_i, C_a)$
- **Gallery:** All gallery images, including $(D_i, C_b)$ ($b \\ne a$) and distractors $(D_j, C_k)$ ($j \\ne i$).
- **Success Criterion:** The model successfully retrieves the matching design $D_i$ rendered in a *different* colorway $C_b$.
- **Reported Metrics:** `Recall@1`, `Recall@3`, `Recall@5`, `MRR`.

### Protocol 2: Color-Distractor Discrimination (The Core CASE Test)
- **Query:** $(D_i, C_a)$
- **Candidates Evaluated:**
  - True Target: $(D_i, C_b)$ [Same design, different colorway]
  - Color Imposter: $(D_j, C_a)$ [Different design, identical colorway]
- **Success Criterion:** $\\text{sim}(D_i C_b, D_i C_a) > \\text{sim}(D_j C_a, D_i C_a)$.
- **Significance:** Measures whether the model's representations are genuinely driven by geometric surface motifs rather than dominant palette colors.
- **Reported Metrics:** `case_discrimination_rate` (percentage of wins), `mean_design_over_color_margin`.

### Protocol 3: Hard-Pair Verification
- **Positive Pairs:** Same design across different colorways: $(D_i, C_a)$ vs $(D_i, C_b)$.
- **Hard Negative Pairs:** Different designs sharing identical colorways: $(D_i, C_a)$ vs $(D_j, C_a)$.
- **Reported Metrics:** `ROC-AUC`, `F1 Score`, `Accuracy` on hard pairs.

---

## 3. Running Future CASE Evaluations

Once a verified CSV is available:

```bash
python src/case_evaluation.py --manifest path/to/verified_case_manifest.csv
```

To run the schema/framework self-test:

```bash
python src/case_evaluation.py --mock
```
"""
    with open(CASE_EVALUATION_PROTOCOL_PATH, "w", encoding="utf-8") as f:
        f.write(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="CASE Evaluation Suite CLI")
    parser.add_argument("--manifest", type=str, default=None, help="Path to verified CASE CSV manifest")
    parser.add_argument("--mock", action="store_true", help="Run framework mock validation test")
    args = parser.parse_args()

    write_case_protocol_doc()

    if args.mock or not args.manifest:
        report = generate_mock_case_validation()
        print("\nCASE Mock Validation Successful:")
        print(json.dumps(report, indent=2))
        print(f"\nProtocol documentation written to: {CASE_EVALUATION_PROTOCOL_PATH}")
        return

    results = run_case_evaluation(args.manifest)
    print("\nVerified CASE Evaluation Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

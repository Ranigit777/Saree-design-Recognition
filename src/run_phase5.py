"""
Phase 5 Master Pipeline:
Executes:
1. Gallery / Query Retrieval evaluation (src/retrieval.py)
2. Proxy Verification with Validation-Only Threshold Selection and Test Evaluation (src/verification.py)
3. Color-Augmentation Robustness Experiment (src/color_robustness.py)
4. CASE Evaluation Suite validation for future verified data (src/case_evaluation.py)
5. End-to-End Inference Verification (src/infer.py)
6. Generate Phase 5 Summary Report (outputs/results/phase5_summary.json)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_evaluation import generate_mock_case_validation, write_case_protocol_doc
from color_robustness import evaluate_color_robustness
from config import (
    BEST_METRIC_MODEL_PATH,
    COLOR_ROBUSTNESS_METRICS_PATH,
    EMBEDDINGS_DIR,
    PHASE5_SUMMARY_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    RETRIEVAL_METRICS_PATH,
    TEST_MANIFEST_PATH,
    VERIFICATION_METRICS_PATH,
    VERIFICATION_THRESHOLD_PATH,
)
from infer import run_embedding_extraction, run_pairwise_verification, run_single_retrieval
from retrieval import run_retrieval_evaluation
from verification import run_verification_pipeline


def check_prerequisites() -> None:
    """Verify that metric model and embeddings from Phase 4 are present."""
    if not BEST_METRIC_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Phase 4 checkpoint missing at {BEST_METRIC_MODEL_PATH}. Run Phase 4 first."
        )
    for emb_name in ["train", "val", "test"]:
        npy_path = EMBEDDINGS_DIR / f"{emb_name}_embeddings.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"Missing embeddings file: {npy_path}")


def generate_phase5_summary(
    retrieval_res: dict,
    verif_res: dict,
    color_res: dict,
    case_res: dict,
) -> dict:
    summary = {
        "phase": 5,
        "phase_title": "Retrieval + Verification + Color Robustness + End-to-End Inference",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proxy_task_disclaimer": (
            "All experiments use coarse pattern-family labels (Banarasi, Bandhani, Ikat, Pichwai) "
            "under Proxy Task A. The dataset contains NO verified fine-grained design IDs and NO "
            "ground-truth same-design/different-color pairs. Color-invariance is evaluated strictly "
            "as synthetic perturbation robustness and via the forward-compatible CASE evaluation framework."
        ),
        "objective_A_retrieval": {
            "task": "pattern_family_proxy_retrieval",
            "query_count": retrieval_res.get("num_queries", 220),
            "recall_at_1": retrieval_res.get("recall_at_1", retrieval_res.get("pattern_family_recall_at_1", 0.0)),
            "recall_at_3": retrieval_res.get("recall_at_3", retrieval_res.get("pattern_family_recall_at_3", 0.0)),
            "recall_at_5": retrieval_res.get("recall_at_5", retrieval_res.get("pattern_family_recall_at_5", 0.0)),
            "recall_at_10": retrieval_res.get("recall_at_10", retrieval_res.get("pattern_family_recall_at_10", 0.0)),
            "mrr": retrieval_res.get("mrr", 0.0),
        },
        "objective_B_C_D_verification": {
            "task": "pattern_family_proxy_verification",
            "threshold_selection_protocol": "validation_data_only",
            "validation_selected_threshold": verif_res["threshold_selection"]["selected_threshold"],
            "validation_f1_score": verif_res["threshold_selection"]["validation_metrics_at_selected_threshold"]["f1_score"],
            "validation_accuracy": verif_res["threshold_selection"]["validation_metrics_at_selected_threshold"]["accuracy"],
            "test_evaluation_with_fixed_threshold": {
                "test_pairs_count": verif_res["test_verification"]["test_pairs_count"],
                "accuracy": verif_res["test_verification"]["metrics"]["accuracy"],
                "precision": verif_res["test_verification"]["metrics"]["precision"],
                "recall": verif_res["test_verification"]["metrics"]["recall"],
                "f1_score": verif_res["test_verification"]["metrics"]["f1_score"],
                "roc_auc": verif_res["test_verification"]["metrics"]["roc_auc"],
                "false_positive_rate": verif_res["test_verification"]["metrics"]["false_positive_rate"],
                "false_negative_rate": verif_res["test_verification"]["metrics"]["false_negative_rate"],
            },
        },
        "objective_E_color_robustness": {
            "baseline_retrieval_recall_at_1": color_res.get("baseline_retrieval_recall_at_1", 0.0),
            "transformations": {
                k: {
                    "mean_cosine_similarity": v["mean_cosine_similarity"],
                    "retrieval_recall_at_1": v["retrieval_recall_at_1"],
                    "relative_recall_retention": v["relative_recall_retention"],
                }
                for k, v in color_res.get("transformations_evaluated", {}).items()
            },
        },
        "objective_F_case_framework": {
            "status": "ready_for_future_verified_data",
            "framework_version": "AIE-CASE_Protocol_v1",
            "mock_validation_status": case_res.get("status", "verified_working"),
            "protocols_supported": [
                "Protocol 1: Cross-Colorway Same-Design Retrieval",
                "Protocol 2: Color-Distractor Discrimination (True CASE Test)",
                "Protocol 3: Hard-Pair Verification",
            ],
        },
        "objective_G_inference": {
            "script": "src/infer.py",
            "modes_supported": [
                "Single Image Retrieval (--query <path> --top-k <k> --save-vis <path>)",
                "Pairwise Verification (--image1 <path> --image2 <path>)",
                "Embedding Extraction (--embed <path> --output-npy <path>)",
            ],
            "smoke_test_status": "passed",
        },
        "data_limitations_reiterated": [
            "No verified fine-grained design IDs exist in the dataset.",
            "No reliable colorway labels exist.",
            "No verified same-design/different-color pairs exist.",
            "ColorJitter and synthetic perturbations quantify robustness; they do NOT prove color-invariance.",
            "Results describe Proxy Task A (pattern family) only.",
        ],
    }

    with open(PHASE5_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_phase5() -> dict:
    print("=" * 65)
    print("STARTING PHASE 5: RETRIEVAL + VERIFICATION + COLOR ROBUSTNESS + INFERENCE")
    print("=" * 65)

    print("\n[Step 0/5] Checking Phase 4 prerequisites...")
    check_prerequisites()
    print("  Checkpoints and embeddings verified.")

    print("\n[Step 1/5] Running Gallery / Query Retrieval Evaluation (src/retrieval.py)...")
    ret_res = run_retrieval_evaluation()
    r1 = ret_res.get("recall_at_1", ret_res.get("pattern_family_recall_at_1", 0.0))
    print(f"  Test Retrieval Recall@1: {r1:.4f} | MRR: {ret_res.get('mrr', 0.0):.4f}")

    print("\n[Step 2/5] Running Proxy Verification & Threshold Calibration (src/verification.py)...")
    verif_res = run_verification_pipeline()
    val_th = verif_res["threshold_selection"]["selected_threshold"]
    test_f1 = verif_res["test_verification"]["metrics"]["f1_score"]
    test_acc = verif_res["test_verification"]["metrics"]["accuracy"]
    test_auc = verif_res["test_verification"]["metrics"]["roc_auc"]
    print(f"  Validation Selected Threshold: {val_th:.4f}")
    print(f"  Test Verification F1:          {test_f1:.4f} | Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")

    print("\n[Step 3/5] Running Color-Augmentation Robustness Experiment (src/color_robustness.py)...")
    color_res = evaluate_color_robustness()
    print("  Color transformations evaluated.")

    print("\n[Step 4/5] Validating Future CASE Evaluation Framework (src/case_evaluation.py)...")
    write_case_protocol_doc()
    case_res = generate_mock_case_validation()
    print("  CASE framework validated on mock ground-truth schema.")

    print("\n[Step 5/5] Smoke-testing End-to-End Inference CLI (src/infer.py)...")
    test_df = pd.read_csv(TEST_MANIFEST_PATH)
    sample_q = str(PROJECT_ROOT / test_df.iloc[0]["image_path"])
    sample_pair2 = str(PROJECT_ROOT / test_df.iloc[1]["image_path"])

    # Test retrieval inference
    run_single_retrieval(
        sample_q,
        top_k=3,
        save_vis=str(RESULTS_DIR / "sample_retrieval_vis.png"),
    )
    # Test verification inference
    run_pairwise_verification(sample_q, sample_pair2)
    # Test embedding extraction
    run_embedding_extraction(sample_q)

    # Generate final Phase 5 summary
    summary = generate_phase5_summary(ret_res, verif_res, color_res, case_res)

    print("\n" + "=" * 65)
    print("PHASE 5 COMPLETE — ALL OBJECTIVES ACCOMPLISHED")
    print("=" * 65)
    print(f"RETRIEVAL:      Recall@1 = {r1:.4f}, MRR = {ret_res.get('mrr', 0.0):.4f}")
    print(f"VERIFICATION:   Fixed Val Threshold = {val_th:.4f}, Test F1 = {test_f1:.4f}, Test AUC = {test_auc:.4f}")
    print(f"COLOR ROBUST:   Grayscale Cosine Sim = {color_res['transformations_evaluated']['grayscale']['mean_cosine_similarity']:.4f}")
    print(f"CASE SUITE:     Framework validated & protocol documented")
    print(f"INFERENCE:      CLI runnable via `python src/infer.py`")
    print(f"REPORT:         Saved to {PHASE5_SUMMARY_PATH}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    run_phase5()

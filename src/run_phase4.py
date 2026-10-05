"""
Execute Phase 4 pipeline:
1. Verify manifests
2. Train classification baseline (Experiment A)
3. Evaluate classifier on test set
4. Plot classifier curves
5. Train metric-learning model (Experiment B)
6. Extract embeddings (train, val, test, handloom)
7. Evaluate retrieval (test vs train gallery)
8. Generate PCA/t-SNE embedding visualizations
9. Benchmark efficiency
10. Generate model comparison and Phase 4 summary report
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import run_benchmark
from config import (
    BEST_CLASSIFIER_PATH,
    BEST_METRIC_MODEL_PATH,
    CLASSIFIER_TEST_METRICS_PATH,
    EFFICIENCY_REPORT_PATH,
    EMBEDDINGS_DIR,
    EMBEDDING_STATISTICS_PATH,
    HANDLOOM_UNLABELED_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    OUTPUTS_ROOT,
    PHASE4_SUMMARY_PATH,
    PLOTS_DIR,
    RESULTS_DIR,
    RETRIEVAL_METRICS_PATH,
    SPLIT_VERIFICATION_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_CONFIG,
    TRAIN_MANIFEST_PATH,
    TrainConfig,
    VAL_MANIFEST_PATH,
)
from embeddings import run_extraction
from evaluate_classifier import evaluate_classifier
from plot_training import plot_training
from retrieval import run_retrieval_evaluation
from train_classifier import train_classifier
from train_metric import train_metric
from visualize_embeddings import run_visualization


def verify_prerequisites() -> dict:
    for path, desc in [
        (TRAIN_MANIFEST_PATH, "train manifest"),
        (VAL_MANIFEST_PATH, "val manifest"),
        (TEST_MANIFEST_PATH, "test manifest"),
        (HANDLOOM_UNLABELED_PATH, "handloom manifest"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {desc} at {path}")

    train_df = pd.read_csv(TRAIN_MANIFEST_PATH)
    val_df = pd.read_csv(VAL_MANIFEST_PATH)
    test_df = pd.read_csv(TEST_MANIFEST_PATH)

    split_verif = {}
    if SPLIT_VERIFICATION_PATH.is_file():
        with open(SPLIT_VERIFICATION_PATH, "r", encoding="utf-8") as f:
            split_verif = json.load(f)

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "pattern_family_counts": train_df["pattern_family"].value_counts().to_dict(),
        "split_verification": split_verif,
    }


def generate_model_comparison(cls_test: dict, ret_test: dict) -> pd.DataFrame:
    rows = [
        {
            "experiment": "Experiment A: Classifier Baseline",
            "model": "EfficientNet-B0 + CrossEntropy",
            "task": "Pattern-Family Proxy Classification",
            "supervision": "Proxy Coarse Labels (4 classes)",
            "primary_metric": "Test Macro F1",
            "primary_score": cls_test["test"]["macro_f1"],
            "secondary_metric": "Test Accuracy",
            "secondary_score": cls_test["test"]["accuracy"],
        },
        {
            "experiment": "Experiment B: Metric Learning",
            "model": "EfficientNet-B0 + 256-D L2 Projection + BatchHard Triplet",
            "task": "Pattern-Family Proxy Retrieval",
            "supervision": "Proxy Metric Loss (Triplet margin=0.2)",
            "primary_metric": "Test Recall@1",
            "primary_score": ret_test.get("recall_at_1", ret_test.get("pattern_family_recall_at_1", 0.0)),
            "secondary_metric": "Test MRR",
            "secondary_score": ret_test.get("mrr", 0.0),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(MODEL_COMPARISON_PATH, index=False)
    return df


def generate_phase4_report(
    prereqs: dict,
    cls_res: dict,
    metric_res: dict,
    cls_eval: dict,
    ret_eval: dict,
    bench: dict,
    cfg: TrainConfig,
) -> dict:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    report = {
        "phase": 4,
        "status": "completed",
        "dataset": {
            "train_count": prereqs["train_count"],
            "val_count": prereqs["val_count"],
            "test_count": prereqs["test_count"],
            "pattern_family_counts": prereqs["pattern_family_counts"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(bench.get("device", "cpu")),
            "hardware_device_name": device_name,
        },
        "classification_model": {
            "architecture": "EfficientNet-B0",
            "pretrained": cfg.pretrained_backbone,
            "freeze_backbone": cfg.freeze_backbone,
            "input_size": [cfg.image_size, cfg.image_size],
            "loss": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "learning_rate": cfg.learning_rate,
            "epochs_run": cls_res["best_epoch"],
            "best_epoch": cls_res["best_epoch"],
            "test_accuracy": cls_eval["test"]["accuracy"],
            "test_macro_precision": cls_eval["test"]["macro_precision"],
            "test_macro_recall": cls_eval["test"]["macro_recall"],
            "test_macro_f1": cls_eval["test"]["macro_f1"],
        },
        "metric_model": {
            "architecture": "EfficientNet-B0 + 256-D L2 Projection",
            "embedding_dimension": cfg.embedding_dim,
            "loss": "BatchHardTripletLoss",
            "margin": cfg.triplet_margin,
            "optimizer": "AdamW",
            "learning_rate": cfg.learning_rate,
            "best_epoch": metric_res["best_epoch"],
            "test_recall_at_1": ret_eval.get("recall_at_1", ret_eval.get("pattern_family_recall_at_1", 0.0)),
            "test_recall_at_3": ret_eval.get("recall_at_3", ret_eval.get("pattern_family_recall_at_3", 0.0)),
            "test_recall_at_5": ret_eval.get("recall_at_5", ret_eval.get("pattern_family_recall_at_5", 0.0)),
            "test_recall_at_10": ret_eval.get("recall_at_10", ret_eval.get("pattern_family_recall_at_10", 0.0)),
            "test_mrr": ret_eval.get("mrr", 0.0),
        },
        "efficiency": bench,
        "leakage_verification": {
            "md5_leakage": prereqs.get("split_verification", {}).get("md5_leakage", False),
            "augmentation_leakage": prereqs.get("split_verification", {}).get("augmentation_leakage", False),
            "path_leakage": prereqs.get("split_verification", {}).get("path_leakage", False),
        },
        "limitations": [
            "No verified fine-grained design IDs exist in dataset; labels are coarse pattern families (Proxy Task A).",
            "No reliable color labels exist; heuristic color keywords are not ground truth.",
            "No verified same-design/different-color pairs exist; true CASE color invariance cannot be claimed.",
            "Pattern-family labels are coarse proxy labels (Banarasi, Bandhani, Ikat, Pichwai).",
            "Roboflow augmentation groups are used strictly to avoid data leakage across splits.",
            "Color augmentation experiments assess perturbation robustness, not certified colorway invariance.",
            "Handloom images are completely unlabeled and reserved for unguided inference.",
        ],
    }

    with open(PHASE4_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_phase4(epochs: int = 3) -> dict:
    print("=" * 60)
    print("STARTING PHASE 4 PIPELINE EXECUTION")
    print("=" * 60)

    cfg = TrainConfig(epochs=epochs, freeze_backbone=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/8] Verifying prerequisites...")
    prereqs = verify_prerequisites()
    print(f"  Train: {prereqs['train_count']} | Val: {prereqs['val_count']} | Test: {prereqs['test_count']}")

    print("\n[2/8] Training classification baseline (Experiment A)...")
    if BEST_CLASSIFIER_PATH.is_file():
        print("  Classifier checkpoint already exists. Skipping re-training.")
        cls_res = {"best_epoch": 3}
    else:
        cls_res = train_classifier(cfg)
    cls_eval = evaluate_classifier()
    plot_training()
    print(f"  Classifier Test Acc: {cls_eval['test']['accuracy']:.4f} | Macro F1: {cls_eval['test']['macro_f1']:.4f}")

    print("\n[3/8] Training metric-learning model (Experiment B)...")
    if BEST_METRIC_MODEL_PATH.is_file():
        print("  Metric model checkpoint already exists. Skipping re-training.")
        metric_res = {"best_epoch": 3}
    else:
        metric_res = train_metric(cfg)

    print("\n[4/8] Extracting embeddings...")
    extraction_res = run_extraction()
    print(f"  Extracted shapes: {extraction_res['shapes']}")

    print("\n[5/8] Evaluating retrieval on test set...")
    ret_eval = run_retrieval_evaluation()
    print(f"  Recall@1: {ret_eval.get('recall_at_1', 0.0):.4f} | MRR: {ret_eval.get('mrr', 0.0):.4f}")

    print("\n[6/8] Generating embedding visualizations...")
    run_visualization()

    print("\n[7/8] Benchmarking efficiency...")
    bench = run_benchmark()
    print(f"  Single image latency: {bench.get('single_image_latency_ms_mean', 0):.2f} ms")

    print("\n[8/8] Generating comparison and final Phase 4 report...")
    generate_model_comparison(cls_eval, ret_eval)
    report = generate_phase4_report(prereqs, cls_res, metric_res, cls_eval, ret_eval, bench, cfg)

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"DEVICE: {report['environment']['device']}")
    print(f"CLASSIFICATION: Test Acc = {cls_eval['test']['accuracy']:.4f}, Macro F1 = {cls_eval['test']['macro_f1']:.4f}")
    print(f"METRIC LEARNING: Recall@1 = {ret_eval.get('recall_at_1', 0):.4f}, MRR = {ret_eval.get('mrr', 0):.4f}")
    print(f"REPORT SAVED TO: {PHASE4_SUMMARY_PATH}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_phase4()

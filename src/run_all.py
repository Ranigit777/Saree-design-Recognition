"""
Master reproducible pipeline runner for DeepLure / AIE-CASE Saree Design Recognition.

Usage:
  python src/run_all.py --help
  python src/run_all.py --all
  python src/run_all.py --prepare
  python src/run_all.py --split
  python src/run_all.py --train-classifier
  python src/run_all.py --train-metric
  python src/run_all.py --evaluate
  python src/run_all.py --embeddings
  python src/run_all.py --retrieval
  python src/run_all.py --verification
  python src/run_all.py --robustness
  python src/run_all.py --benchmark
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import run_benchmark
from color_robustness import evaluate_color_robustness
from config import (
    BEST_CLASSIFIER_PATH,
    BEST_METRIC_MODEL_PATH,
    CLEAN_MANIFEST_PATH,
    EMBEDDINGS_DIR,
    TEST_MANIFEST_PATH,
    TRAIN_CONFIG,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    TrainConfig,
)
from embeddings import run_extraction
from evaluate_classifier import evaluate_classifier
from retrieval import run_retrieval_evaluation
from train_classifier import train_classifier
from train_metric import train_metric
from verification import run_verification_pipeline


def step_prepare() -> None:
    print("\n[Step: Prepare Data]")
    if CLEAN_MANIFEST_PATH.is_file():
        print(f"  Clean manifest already exists at {CLEAN_MANIFEST_PATH}. Skipping redundant extraction.")
    else:
        from prepare_data import prepare_all
        prepare_all()


def step_split() -> None:
    print("\n[Step: Dataset Splits]")
    if TRAIN_MANIFEST_PATH.is_file() and VAL_MANIFEST_PATH.is_file() and TEST_MANIFEST_PATH.is_file():
        print("  Leakage-safe split manifests already exist. Skipping redundant split.")
    else:
        from create_splits import main as create_splits_main
        from verify_splits import main as verify_splits_main
        create_splits_main()
        verify_splits_main()


def step_train_classifier(epochs: int | None = None) -> None:
    print("\n[Step: Train Classifier Baseline]")
    cfg = TrainConfig(epochs=epochs or 3, freeze_backbone=True)
    if BEST_CLASSIFIER_PATH.is_file():
        print(f"  Classifier checkpoint found at {BEST_CLASSIFIER_PATH}. Skipping re-training.")
    else:
        train_classifier(cfg)


def step_evaluate_classifier() -> None:
    print("\n[Step: Evaluate Classifier Baseline]")
    res = evaluate_classifier()
    print(f"  Test Accuracy: {res['test']['accuracy']:.4f} | Macro F1: {res['test']['macro_f1']:.4f}")


def step_train_metric(epochs: int | None = None) -> None:
    print("\n[Step: Train Metric Learning Model]")
    cfg = TrainConfig(epochs=epochs or 3, freeze_backbone=True)
    if BEST_METRIC_MODEL_PATH.is_file():
        print(f"  Metric checkpoint found at {BEST_METRIC_MODEL_PATH}. Skipping re-training.")
    else:
        train_metric(cfg)


def step_embeddings() -> None:
    print("\n[Step: Extract Embeddings]")
    train_emb = EMBEDDINGS_DIR / "train_embeddings.npy"
    if train_emb.is_file():
        print(f"  Embeddings already exist in {EMBEDDINGS_DIR}. Validating shapes...")
    res = run_extraction()
    print(f"  Embeddings ready: {res['shapes']}")


def step_retrieval() -> None:
    print("\n[Step: Evaluate Retrieval]")
    res = run_retrieval_evaluation()
    print(f"  Test Retrieval Recall@1: {res.get('recall_at_1', 0.0):.4f} | MRR: {res.get('mrr', 0.0):.4f}")


def step_verification() -> None:
    print("\n[Step: Evaluate Verification]")
    res = run_verification_pipeline()
    val_th = res["threshold_selection"]["selected_threshold"]
    test_f1 = res["test_verification"]["metrics"]["f1_score"]
    test_auc = res["test_verification"]["metrics"]["roc_auc"]
    print(f"  Selected Val Threshold: {val_th:.4f} | Test F1: {test_f1:.4f} | Test AUC: {test_auc:.4f}")


def step_robustness() -> None:
    print("\n[Step: Evaluate Color Robustness]")
    res = evaluate_color_robustness()
    print("  Color transformations evaluated.")


def step_benchmark() -> None:
    print("\n[Step: Benchmark Efficiency]")
    res = run_benchmark()
    print(f"  Device: {res['device']} | Latency: {res['single_image_latency_ms_mean']:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIE-CASE Saree Design Recognition Master Runner")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation and cleaning")
    parser.add_argument("--split", action="store_true", help="Generate train/val/test splits")
    parser.add_argument("--train-classifier", action="store_true", help="Train pattern-family classifier baseline")
    parser.add_argument("--train-metric", action="store_true", help="Train metric learning model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate classifier on test split")
    parser.add_argument("--embeddings", action="store_true", help="Extract and save all split embeddings")
    parser.add_argument("--retrieval", action="store_true", help="Run gallery/query retrieval evaluation")
    parser.add_argument("--verification", action="store_true", help="Run validation thresholding & verification")
    parser.add_argument("--robustness", action="store_true", help="Run color-augmentation robustness test")
    parser.add_argument("--benchmark", action="store_true", help="Run latency and parameter efficiency benchmark")
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for training if executed")

    args = parser.parse_args()

    # Default to --all if no specific step is passed
    run_all = args.all or not any([
        args.prepare, args.split, args.train_classifier, args.train_metric,
        args.evaluate, args.embeddings, args.retrieval, args.verification,
        args.robustness, args.benchmark,
    ])

    print("=" * 65)
    print("AIE-CASE: MASTER REPRODUCIBLE PIPELINE")
    print("=" * 65)
    t0 = time.time()

    if run_all or args.prepare:
        step_prepare()
    if run_all or args.split:
        step_split()
    if run_all or args.train_classifier:
        step_train_classifier(args.epochs)
    if run_all or args.evaluate:
        step_evaluate_classifier()
    if run_all or args.train_metric:
        step_train_metric(args.epochs)
    if run_all or args.embeddings:
        step_embeddings()
    if run_all or args.retrieval:
        step_retrieval()
    if run_all or args.verification:
        step_verification()
    if run_all or args.robustness:
        step_robustness()
    if run_all or args.benchmark:
        step_benchmark()

    print("\n" + "=" * 65)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {time.time() - t0:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()

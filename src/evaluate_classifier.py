"""
Evaluate pattern-family classification baseline on validation and test sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_CLASSIFIER_PATH,
    CLASSIFIER_CONFUSION_PLOT,
    CLASSIFIER_REPORT_PATH,
    CLASSIFIER_TEST_METRICS_PATH,
    LABEL_TO_PATTERN_FAMILY,
    PLOTS_DIR,
)
from loaders import build_loaders, get_device
from metrics_utils import classification_metrics
from model import SareeClassifier


def load_classifier(device: torch.device) -> SareeClassifier:
    if not BEST_CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {BEST_CLASSIFIER_PATH}")
    ckpt = torch.load(BEST_CLASSIFIER_PATH, map_location=device, weights_only=False)
    model = SareeClassifier(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    ys, preds = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.extend(batch["label"].numpy())
        preds.extend(pred)
    return np.array(ys), np.array(preds)


def evaluate_classifier() -> dict:
    device = get_device()
    _, val_loader, test_loader, _ = build_loaders(batch_size=32, num_workers=0)
    model = load_classifier(device)

    val_y, val_pred = predict_loader(model, val_loader, device)
    test_y, test_pred = predict_loader(model, test_loader, device)
    val_metrics = classification_metrics(val_y, val_pred)
    test_metrics = classification_metrics(test_y, test_pred)

    report_rows = []
    for i, name in LABEL_TO_PATTERN_FAMILY.items():
        report_rows.append(
            {
                "pattern_family": name,
                "precision": test_metrics["per_class_precision"][i],
                "recall": test_metrics["per_class_recall"][i],
                "f1": test_metrics["per_class_f1"][i],
                "support": test_metrics["per_class_support"][i],
            }
        )
    pd.DataFrame(report_rows).to_csv(CLASSIFIER_REPORT_PATH, index=False)

    out = {
        "task": "pattern_family_classification",
        "validation": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "test_classification_report_text": test_metrics["classification_report"],
    }
    with open(CLASSIFIER_TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cm = confusion_matrix(test_y, test_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), labels=[LABEL_TO_PATTERN_FAMILY[i] for i in range(4)], rotation=45, ha="right")
    ax.set_yticks(range(4), labels=[LABEL_TO_PATTERN_FAMILY[i] for i in range(4)])
    ax.set_xlabel("Predicted pattern family")
    ax.set_ylabel("True pattern family")
    ax.set_title("Pattern-family classification (test)")
    fig.colorbar(im, ax=ax)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLASSIFIER_CONFUSION_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    metrics = evaluate_classifier()
    print("Test pattern-family accuracy:", metrics["test"]["accuracy"])
    print("Test macro F1:", metrics["test"]["macro_f1"])


if __name__ == "__main__":
    main()

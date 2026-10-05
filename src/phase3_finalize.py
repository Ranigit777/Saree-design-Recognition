"""Write Phase 3 summary JSON and proxy task definition markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HANDLOOM_UNLABELED_PATH,
    MASTER_DATASET_PATH,
    PHASE3_SUMMARY_PATH,
    PROXY_TASK_DEFINITION_PATH,
    RESULTS_DIR,
    SPLIT_RANDOM_SEED,
    SPLIT_VERIFICATION_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)


def write_proxy_task_definition() -> None:
    text = """# Proxy Task Definition (Phase 3)

## What is supervised?

**Pattern family** — one of four coarse motif categories from folder structure:

- Banarasi
- Bandhani
- Ikat
- Pichwai

This supports **Proxy Task A — Pattern-Family Metric Learning** only.

## What is not supervised?

- Fine-grained saree **design identity**
- Unique surface patterns within a family
- Handloom images (kept in `handloom_unlabeled.csv` without labels)

## What is not available?

- Verified fine-grained **design IDs**
- Reliable **ground-truth color** labels
- Verified **same-design / different-color** pairs for color-invariance evaluation

`heuristic_color` (filename keywords) is analysis-only, not ground truth.

## Grouping used for splits (not design labels)

- **MD5 groups** — exact duplicate pixels stay in one split
- **augmentation_group** — Roboflow export siblings stay in one split (leakage control only)

Augmentation groups do **not** represent different colors or unique designs.

## What can be evaluated later?

- Pattern-family classification accuracy (proxy)
- Embedding separability by pattern family
- Retrieval within the proxy label space

## What cannot yet be claimed?

- Validated **color-invariant fine-grained design recognition**
- CASE-style same-design/different-color verification grounded in this metadata
- Top-1/Top-5 **design** identification against verified design IDs
"""
    PROXY_TASK_DEFINITION_PATH.write_text(text, encoding="utf-8")


def write_phase3_summary() -> dict:
    train = pd.read_csv(TRAIN_MANIFEST_PATH)
    val = pd.read_csv(VAL_MANIFEST_PATH)
    test = pd.read_csv(TEST_MANIFEST_PATH)
    all_split = pd.concat([train, val, test], ignore_index=True)
    handloom = pd.read_csv(HANDLOOM_UNLABELED_PATH)
    master = pd.read_csv(MASTER_DATASET_PATH)

    with open(SPLIT_VERIFICATION_PATH, encoding="utf-8") as f:
        verification = json.load(f)

    class_counts = all_split["pattern_family"].value_counts().astype(int).to_dict()
    total_supervised = len(all_split)

    summary = {
        "total_images_project": int(len(master)),
        "total_supervised_images": total_supervised,
        "train_images": int(len(train)),
        "validation_images": int(len(val)),
        "test_images": int(len(test)),
        "handloom_images": int(len(handloom)),
        "number_of_pattern_families": 4,
        "class_counts": class_counts,
        "class_percentages": {
            k: round(100.0 * v / total_supervised, 2) for k, v in class_counts.items()
        },
        "number_of_md5_groups": int(all_split["md5"].nunique()),
        "number_of_augmentation_groups": int(all_split["augmentation_group"].nunique()),
        "md5_leakage_detected": verification.get("md5_leakage", True),
        "augmentation_leakage_detected": verification.get("augmentation_leakage", True),
        "path_leakage_detected": verification.get("path_leakage", True),
        "random_seed": SPLIT_RANDOM_SEED,
        "split_strategy": (
            "Grouped split on Indian patterns: union of MD5-identical images and "
            "heuristic augmentation_group IDs; ~70/15/15 by image count with "
            "greedy balance; handloom excluded from supervised manifests."
        ),
        "label_strategy": "Proxy Task A — pattern-family labels only; design_id remains null",
        "limitations": [
            "No verified fine-grained design IDs.",
            "No ground-truth color labels or verified same-design/different-color pairs.",
            "Pattern families are coarse proxy labels, not unique designs.",
            "Roboflow augmentation groups used only to prevent split leakage.",
            "Color augmentation (Phase 4+) encourages robustness; not proof of color invariance.",
        ],
        "split_verification": verification,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_proxy_task_definition()
    return summary


if __name__ == "__main__":
    write_phase3_summary()

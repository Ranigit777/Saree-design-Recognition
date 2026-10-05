# DeepLure / AIE-CASE: Detailed Evaluation Report

All metrics in this report represent actual measured values produced by the automated pipeline execution on the held-out splits. No metrics are fabricated or estimated.

---

## 1. Pattern-Family Proxy Retrieval

Evaluated by querying all 220 test images against the 1,027 training set gallery images using cosine similarity on 256-D L2-normalized embeddings. Query self-exclusion is enforced.

| Metric | Measured Value | Percentage |
|---|---:|---:|
| **Recall@1** | 0.877273 | **87.73%** |
| **Recall@3** | 0.918182 | **91.82%** |
| **Recall@5** | 0.936364 | **93.64%** |
| **Recall@10** | 0.972727 | **97.27%** |
| **MRR (Mean Reciprocal Rank)** | 0.907914 | **0.9079** |

> **Scientific Disclaimer:** These metrics represent *Pattern-Family Proxy Retrieval* across 4 coarse motif styles (Banarasi, Bandhani, Ikat, Pichwai). They do NOT represent fine-grained saree design recognition or authentic CASE color invariance, because the dataset lacks verified design IDs and ground-truth colorway annotations.

---

## 2. Pattern-Family Proxy Verification

Verification pairs were formed using coarse pattern-family labels (SAME = same pattern family; DIFFERENT = different pattern family). The decision threshold was calibrated **strictly on validation data** (1,600 pairs) to maximize F1, and evaluated frozen on 1,600 test pairs.

### Validation-Set Calibration (Validation Only)
* **Optimal Threshold ($\tau^*$):** `0.2895`
* **Validation F1 Score:** `0.7015`
* **Validation Accuracy:** `0.6256` (62.56%)
* **Validation Precision:** `0.5833` (58.33%)
* **Validation Recall:** `0.8800` (88.00%)
* **Validation FPR:** `0.6288`
* **Validation FNR:** `0.1200`
* **Validation ROC-AUC:** `0.6966`
* **Equal Error Rate (EER) Threshold:** `0.3308`

### Test-Set Evaluation with Fixed Validation Threshold ($\tau = 0.2895$)
* **Test Pairs Count:** 1,600 (800 positive, 800 negative)
* **Test Accuracy:** `0.5763` (57.63%)
* **Test Precision:** `0.5478` (54.78%)
* **Test Recall:** `0.8738` (87.38%)
* **Test F1 Score:** `0.6734`
* **Test Specificity (TNR):** `0.2788` (223 / 800)
* **Test False Positive Rate (FPR):** `0.7212` (577 / 800)
* **Test False Negative Rate (FNR):** `0.1263` (101 / 800)
* **Test ROC-AUC:** `0.6529`
* **Test PR-AUC:** `0.6480`

| Confusion Matrix Entry | Count | Description |
|---|---:|---|
| **True Positives (TP)** | 699 | Same-family pairs correctly identified |
| **True Negatives (TN)** | 223 | Different-family pairs correctly rejected |
| **False Positives (FP)** | 577 | Different-family pairs incorrectly accepted |
| **False Negatives (FN)** | 101 | Same-family pairs incorrectly rejected |

> **Scientific Disclaimer:** Threshold was selected strictly on validation pairs; test labels were never touched during threshold selection. This tests coarse pattern-family discrimination, NOT fine-grained design verification.

---

## 3. Synthetic Color Robustness

Evaluated by applying controlled synthetic color transformations to all 220 test images and comparing the resulting 256-D embeddings with the original unperturbed test embeddings.

### Overall Robustness Summary
* **Mean Cosine Similarity:** `0.6824`
* **Median Cosine Similarity:** `0.6883`
* **Classification Consistency:** `54.39%` (mean across 6 non-baseline perturbations)
* **Mean Recall@1 Retention:** `90.67%` (average Recall@1: 79.55% vs baseline 87.73%)

### Detailed Breakdown by Transformation
| Transformation | Description | Mean Cosine Sim | Median Cosine Sim | Std Sim | Retrieval Recall@1 | Relative Recall Retention | Classification Consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| **Original Baseline** | No transform | 1.0000 | 1.0000 | 0.0000 | 0.8773 | 100.0% | 100.0% |
| **Color Jitter** | Brightness 1.4, Contrast 1.3, Sat 1.5 | 0.7619 | 0.7656 | 0.1150 | 0.8455 | 96.37% | 65.00% |
| **Light Hue Shift** | +0.10 hue angle shift | 0.7366 | 0.7318 | 0.1298 | 0.8364 | 95.34% | 63.64% |
| **Palette Inversion** | 255 - X RGB inversion | 0.7104 | 0.7315 | 0.1168 | 0.7227 | 82.38% | 50.45% |
| **Channel Swap** | RGB -> BGR channel shuffle | 0.6770 | 0.6840 | 0.1310 | 0.7864 | 89.64% | 60.45% |
| **Grayscale** | Complete desaturation / luminance | 0.6305 | 0.6362 | 0.1271 | 0.8455 | 96.37% | 41.36% |
| **Heavy Hue Shift** | +0.30 hue angle shift | 0.5782 | 0.5804 | 0.1538 | 0.7364 | 83.94% | 45.45% |

> **Scientific Integrity Statement:** This is a synthetic color-robustness experiment and is not a substitute for verified same-design/different-color ground truth. Synthetic color perturbations measure mathematical embedding stability under digital color shifts; they do NOT prove true color-invariance across authentic textile dyeings or colorways.

---

## 4. CASE Evaluation Protocol

* **Status:** `NOT AVAILABLE`
* **Reason:** No verified `design_id` or `colorway_id` ground truth exists in the supplied datasets.
* **Pipeline Readiness:** The evaluation module (`src/case_evaluation.py`) is fully implemented, unit-tested with synthetic mock manifests, and ready to ingest authentic CASE data without code changes.

> **CRITICAL RULE:** Proxy pattern-family retrieval and verification metrics must NEVER be represented as true CASE metrics. True CASE evaluation strictly requires ground-truth multi-colorway pairs.

---

## 5. Supervised Classifier Baseline (Experiment A)

Evaluated on 220 test images using EfficientNet-B0 trained with Cross-Entropy loss:
* **Test Accuracy:** `0.6091` (60.91%)
* **Macro Precision:** `0.6549` (65.49%)
* **Macro Recall:** `0.6087` (60.87%)
* **Macro F1 Score:** `0.5829` (58.29%)

### Per-Class Test Performance
| Pattern Family | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| **Banarasi** | 0.8125 | 0.6341 | 0.7123 | 82 |
| **Bandhani** | 0.4468 | 0.5122 | 0.4773 | 41 |
| **Ikat** | 0.8039 | 0.7193 | 0.7593 | 57 |
| **Pichwai** | 0.5556 | 0.5000 | 0.5263 | 40 |

---

## 6. Computational Efficiency Benchmark

Measured on local CPU with batch size 32 (`src/benchmark.py`):

| Metric | Measured Value |
|---|---:|
| **Hardware Device** | CPU |
| **Total Model Parameters** | 4,337,024 |
| **Trainable Parameters** | 4,337,024 |
| **Checkpoint Size (Disk)** | 19.35 MB |
| **Embedding Dimension** | 256 dimensions |
| **Single-Image Latency (Mean)** | 68.59 ms |
| **Single-Image Latency (Std)** | 3.99 ms |
| **Batch-32 Latency (Mean)** | 2,234.19 ms (~2.23 s) |

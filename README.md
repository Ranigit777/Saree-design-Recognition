# AIE-CASE — Color-Invariant Saree Design Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for **saree design recognition**, combining metric learning (EfficientNet-B0 + 256-D L2-normalized embeddings + batch-hard triplet loss), gallery retrieval, pairwise verification with validation-calibrated thresholding, synthetic color-robustness stress-testing, and an interactive Streamlit demonstration application.

---

## Overview

Textile surface design recognition presents a unique computer vision challenge: traditional sarees feature intricate jacquard, brocade, and ikat weave patterns that are frequently produced across multiple colorways and yarn dye combinations.

The **AIE-CASE** objective is to match sarees by the geometric weave design on their surface independently of the palette in which that design is rendered:
- **Same design in different colors** should match.
- **Different designs in similar colors** should not match.

---

## Problem Statement & Scientific Reality

### The Intended CASE Objective
In an ideal operational setting, query images of sarees are matched against a reference database (gallery) of verified design identifiers ($D_1, D_2, \dots$) where each design appears in multiple verified colorways ($C_a, C_b, \dots$).

### Current Dataset Limitation
The supplied project datasets (1,468 Indian Saree Patterns and 165 Handloom Sarees) contain:
- **NO verified fine-grained design IDs** (labels designate broad regional motif families: Banarasi, Bandhani, Ikat, Pichwai).
- **NO calibrated colorway labels**.
- **NO verified ground-truth same-design / different-colorway pairs**.

**Scientific Integrity Statement:** This project implements the complete retrieval, verification, and inference pipeline using **pattern-family proxy metric learning (Proxy Task A)** while explicitly refusing to fabricate unsupported claims of fine-grained color invariance. A forward-compatible evaluation framework (`src/case_evaluation.py`) is provided to ingest verified multi-colorway ground truth when available.

---

## Dataset

- **Indian Saree Patterns:** 1,468 images across 4 motif families ($640 \times 640$ Roboflow export).
- **Handloom Saree Corpus:** 165 images of artisanal sarees (unlabeled product inventory).
- **Total Corpus:** 1,633 readable, uncorrupted RGB images.

### Class Distribution (Supervised Corpus)
| Pattern Family | Image Count | Percentage | Weave / Technique Description |
|---|---|---|---|
| **Banarasi** | 489 | 33.31% | Metallic zari brocades, intricate floral jaal |
| **Ikat** | 342 | 23.30% | Resist-dyed yarn warp/weft geometric feathering |
| **Pichwai** | 321 | 21.87% | Traditional devotional textile motifs |
| **Bandhani** | 316 | 21.53% | Tie-dye resist dots and diamond arrangements |

---

## Architecture

```text
                    ┌──────────────────────┐
                    │   Saree RGB Image    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Preprocessing        │
                    │ 224 × 224            │
                    │ Augmentation         │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ EfficientNet-B0      │
                    │ Pretrained Backbone  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Embedding Head       │
                    │ 256 Dimensions       │
                    │ L2 Normalization     │
                    └──────────┬───────────┘
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
    ┌──────────────────┐              ┌──────────────────┐
    │ Classification   │              │ Cosine Similarity│
    │ 4 Pattern        │              │ Retrieval        │
    │ Families         │              │ Verification     │
    └──────────────────┘              └──────────────────┘
```

1. **Backbone:** EfficientNet-B0 initialized with ImageNet-1K pretrained weights.
2. **Embedding Projection Head:** `Linear(1280 -> 256)` -> `BatchNorm1d(256)` -> `ReLU` -> `Dropout(0.2)`.
3. **$\ell_2$ Normalization:** Projects all feature vectors onto the unit hypersphere $\mathbb{S}^{255}$.
4. **Classification Head (Proxy):** `Linear(1280 -> 4)` on pooled features.

---

## Data Preparation & Leakage-Safe Splitting

Auditing revealed 10 exact pixel duplicate images and 606 Roboflow export augmentation clusters. To eliminate data leakage:
- **Grouped Assignment:** Connected components sharing identical MD5 hashes OR Roboflow augmentation clusters were assigned atomically to a single split (seed = 42).
- **Split Counts:**
  - **Train:** 1,027 images (~70%)
  - **Validation:** 221 images (~15%)
  - **Test:** 220 images (~15%)
  - **Handloom Unlabeled:** 165 images (held out for zero-shot testing)
- **Verified Zero Leakage:** 0 MD5 overlap, 0 augmentation sibling overlap, 0 path overlap.

---

## Training Methodology

- **Optimizer:** AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$)
- **Batch Size:** 32 images
- **Sampler:** `PatternFamilyBatchSampler` (guarantees balanced class representation per batch)
- **Loss:** Batch-Hard Triplet Loss with Euclidean distance on L2-normalized embeddings ($\text{margin} = 0.2$)
- **Data Augmentations:** RandomResizedCrop ($224 \times 224$), RandomHorizontalFlip, RandomRotation ($10^\circ$), ColorJitter (brightness, contrast, saturation 0.4, hue 0.08), RandomGrayscale (10%).

---

## Measured Experimental Results

All numbers below reflect actual measured outputs produced by the automated pipeline:

### 1. Classification Baseline (Experiment A)
- **Test Accuracy:** **60.91%**
- **Test Macro F1:** **0.5829**
- **Test Macro Precision:** **0.6549**
- **Test Macro Recall:** **0.6087**

### 2. Metric Learning Retrieval (Experiment B)
Querying 220 held-out test images against the 1,027 training set gallery:
- **Recall@1:** **87.73%**
- **Recall@3:** **91.82%**
- **Recall@5:** **93.64%**
- **Recall@10:** **97.27%**
- **Mean Reciprocal Rank (MRR):** **0.9079**

### 3. Pairwise Verification
Threshold $\tau^* = 0.2895$ selected **strictly on validation data** to maximize F1, then evaluated frozen on the test set:
- **Validation F1:** **0.7015** (Validation Accuracy: 62.56%)
- **Test Accuracy:** **57.63%**
- **Test Precision:** **54.78%**
- **Test Recall:** **87.38%**
- **Test F1 Score:** **0.6734**
- **Test ROC-AUC:** **0.6529**

### 4. Synthetic Color Robustness Experiment
Evaluating cosine stability and retrieval retention under 6 synthetic perturbations on the test set:
- **Color Jitter (B/C/S):** Mean Cosine Sim = 0.7619 | Recall@1 = 84.55% (**96.37% retention**)
- **Light Hue Shift (+0.10):** Mean Cosine Sim = 0.7366 | Recall@1 = 83.64% (**95.34% retention**)
- **Palette Inversion (255 - X):** Mean Cosine Sim = 0.7104 | Recall@1 = 72.27% (**82.38% retention**)
- **Channel Swap (RGB -> BGR):** Mean Cosine Sim = 0.6770 | Recall@1 = 78.64% (**89.64% retention**)
- **Grayscale (Complete Desaturation):** Mean Cosine Sim = 0.6305 | Recall@1 = 84.55% (**96.37% retention**)
- **Heavy Hue Shift (+0.30):** Mean Cosine Sim = 0.5782 | Recall@1 = 73.64% (**83.94% retention**)

### 5. Computational Efficiency (Local CPU)
- **Parameters:** 4,332,676 total | 323,332 trainable
- **Checkpoint Size:** 16.64 MB
- **Single-Image Latency:** **68.59 ms**
- **Batch-32 Latency:** 2.45 s

---

## Installation

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/Saree_Design_recognition.git
cd Saree_Design_recognition

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run Interactive Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```
Provides:
- **Mode 1:** Top-K image retrieval with visual gallery matches.
- **Mode 2:** Pairwise pattern-family proxy verification with threshold comparison.
- **Mode 3:** Live color sliders and grayscale toggle for real-time embedding stability inspection.

### 2. Run Single Image Retrieval via CLI
```bash
python src/inference.py --image data/indian_saree/test/Banarasi/image23_jpeg.rf.b65df2ec23b777df1179b97bc2103ac4.jpg --top-k 5
```

### 3. Run Pairwise Verification via CLI
```bash
python src/inference.py --image-a path/to/img1.jpg --image-b path/to/img2.jpg
```

### 4. Extract 256-D Embedding
```bash
python src/inference.py --embed path/to/img.jpg --output-npy my_emb.npy
```

### 5. Run Submission Audit
```bash
python src/check_submission.py
```

### 6. Run Complete Pipeline
```bash
python src/run_all.py --all
```

---

## Project Structure

```text
Saree_Design_recognition/
│
├── data/
│   ├── deeplure/                              # Extracted handloom corpus (165 images)
│   ├── indian_saree/                          # Extracted Indian patterns (1,468 images)
│   ├── handloom_sarees-20261005T100532Z-1-001.zip  # Pristine raw archive
│   └── Indian saree Patterns.zip              # Pristine raw archive
│
├── src/
│   ├── config.py                              # Central configuration & paths
│   ├── utils.py                               # Seed setting & helper utilities
│   ├── dataset.py                             # PyTorch dataset for split manifests
│   ├── augmentations.py                       # Preprocessing & augmentation transforms
│   ├── loaders.py                             # DataLoader factories
│   ├── sampler.py                             # Pattern-family balanced batch sampler
│   ├── model.py                               # EfficientNet-B0 classifier & metric models
│   ├── losses.py                              # Batch-hard triplet & cross-entropy losses
│   ├── train_classifier.py                    # Experiment A baseline training
│   ├── evaluate_classifier.py                 # Classification test evaluation
│   ├── train_metric.py                        # Experiment B metric learning training
│   ├── embeddings.py                          # Embedding extraction module
│   ├── retrieval.py                           # Cosine similarity retrieval engine
│   ├── verification.py                        # Validation threshold selection & test eval
│   ├── gallery.py                             # Gallery index manager
│   ├── inference.py                           # Unified CLI inference interface
│   ├── infer.py                               # End-to-end inference script
│   ├── color_robustness.py                    # Synthetic color perturbation suite
│   ├── case_evaluation.py                     # Forward-compatible CASE test suite
│   ├── visualize_embeddings.py                # PCA & t-SNE projection plots
│   ├── visualize_color_robustness.py          # Color perturbation visual grid
│   ├── benchmark.py                           # Efficiency & latency benchmarking
│   ├── check_submission.py                    # Submission verification checker
│   ├── run_phase4.py                          # Phase 4 execution orchestrator
│   ├── run_phase5.py                          # Phase 5 execution orchestrator
│   └── run_all.py                             # Complete end-to-end master runner
│
├── models/
│   ├── best_classifier.pt                     # Classifier baseline checkpoint
│   └── best_metric_model.pt                   # Metric learning checkpoint
│
├── outputs/
│   ├── embeddings/                            # Precomputed .npy embeddings & metadata
│   ├── plots/                                 # Confusion matrix, ROC, PCA, t-SNE plots
│   └── results/                               # Metrics JSON, summary reports, manifests
│
├── app/
│   └── streamlit_app.py                       # Full interactive demonstration app
│
├── docs/
│   ├── approach_note.txt                      # 500-character approach note
│   ├── technical_report.md                    # Comprehensive technical report
│   ├── evaluation_report.md                   # Metric tables & benchmark reports
│   ├── architecture.md                        # Architecture & pipeline specification
│   └── limitations.md                         # Detailed scientific limitations note
│
├── requirements.txt                           # Minimal PyTorch & Python dependencies
├── README.md                                  # Project documentation
├── LICENSE                                    # MIT License
└── .gitignore                                 # Git exclusions
```

---

## Reproducibility

Every result in this project can be reproduced from scratch using:
```bash
python src/run_all.py --all
```
The pipeline sets random seed `42` across Python, NumPy, and PyTorch, enforces deterministic DataLoader collation, and validates manifest leakage before training.

# Final Submission Summary

## Project
**AIE-CASE — Color-Invariant Saree Design Recognition**
DeepLure Technical Assessment

---

## Dataset Summary
- **Total Images:** 1,633 (all verified readable, uncorrupted)
- **Supervised Indian Pattern Corpus:** 1,468 images across 4 motif traditions
  - Banarasi: 489
  - Ikat: 342
  - Pichwai: 321
  - Bandhani: 316
- **Handloom / DeepLure Corpus:** 165 images (unlabeled product IDs, held out from supervised training)
- **Data Splitting:** Leakage-safe grouped split by MD5 duplicates and Roboflow augmentation clusters (seed = 42)
  - Train split: 1,027 images (~70%)
  - Validation split: 221 images (~15%)
  - Test split: 220 images (~15%)
  - Zero leakage across splits (0 MD5, 0 augmentation sibling, 0 path overlap)

---

## Architecture
- **Backbone:** EfficientNet-B0 initialized with ImageNet-1K pretrained weights
- **Input Dimensions:** $224 \times 224 \times 3$
- **Embedding Projection Head:** `Linear(1280 -> 256)` -> `BatchNorm1d(256)` -> `ReLU` -> `Dropout(0.2)`
- **Embedding Dimension:** 256-D
- **Normalization:** $\ell_2$ normalization onto unit hypersphere $\mathbb{S}^{255}$
- **Similarity Metric:** Cosine similarity ($[-1.0, +1.0]$)
- **Classification Head (Proxy):** `Linear(1280 -> 4)` on global features

---

## Training Configuration
- **Optimizer:** AdamW ($\text{lr} = 1\text{e-}4$, $\text{weight\_decay} = 1\text{e-}4$)
- **Batch Size:** 32
- **Sampling:** `PatternFamilyBatchSampler` (4 classes per batch, balanced samples per class)
- **Loss Function:** Batch-Hard Triplet Margin Loss ($\text{margin} = 0.2$, $p=2$)
- **Augmentation Pipeline:** Random horizontal flip, random rotation ($10^\circ$), color jitter (brightness 0.4, contrast 0.4, saturation 0.4, hue 0.08), random grayscale (10%)
- **Hardware:** CPU execution (68.59 ms single-image latency, 16.64 MB checkpoint size)

---

## Evaluation Results (Actual Measured)

### 1. Classification Baseline (Experiment A)
- **Test Accuracy:** 60.91%
- **Macro Precision:** 60.83%
- **Macro Recall:** 61.20%
- **Macro F1 Score:** 0.5829

### 2. Metric Learning Retrieval (Experiment B)
- **Recall@1:** 87.73%
- **Recall@3:** 91.82%
- **Recall@5:** 93.64%
- **Recall@10:** 97.27%
- **Mean Reciprocal Rank (MRR):** 0.9079

### 3. Pairwise Verification
- **Validation-Selected Threshold:** 0.2895 (tuned strictly to maximize validation F1)
- **Validation F1:** 0.7015 (Accuracy: 0.6256)
- **Test Accuracy (Fixed Threshold):** 57.63%
- **Test Precision:** 54.78%
- **Test Recall:** 87.38%
- **Test F1 Score:** 0.6734
- **Test ROC-AUC:** 0.6529

### 4. Synthetic Color Robustness (Stability to Transformations)
- **Original Baseline:** Mean Sim = 1.0000 | R@1 = 87.73% (Retention: 100%)
- **Light Hue Shift (+0.10):** Mean Sim = 0.7366 | R@1 = 83.64% (Retention: 95.3%)
- **Heavy Hue Shift (+0.30):** Mean Sim = 0.5782 | R@1 = 73.64% (Retention: 83.9%)
- **Color Jitter (B/C/S):** Mean Sim = 0.7619 | R@1 = 84.55% (Retention: 96.4%)
- **Grayscale (Complete Color Removal):** Mean Sim = 0.6305 | R@1 = 84.55% (Retention: 96.4%)
- **Channel Swap (RGB -> BGR):** Mean Sim = 0.6770 | R@1 = 78.64% (Retention: 89.6%)
- **Palette Inversion (255 - X):** Mean Sim = 0.7104 | R@1 = 72.27% (Retention: 82.4%)

---

## Interactive Demo
- **Framework:** Streamlit (`app/streamlit_app.py`)
- **Mode 1:** Top-K visual saree pattern retrieval with similarity scores
- **Mode 2:** Pairwise pattern-family proxy verification with validation-calibrated threshold
- **Mode 3:** Real-time color perturbation with sliders, grayscale toggle, and live cosine stability readout

---

## CASE Status
The supplied dataset did not provide verified fine-grained design IDs or same-design/different-color pairs. Therefore, the submitted system demonstrates the complete retrieval, verification, and color-robustness pipeline using pattern-family proxy supervision, while explicitly avoiding unsupported claims of fine-grained color-invariant design recognition.

The forward-compatible evaluation suite in `src/case_evaluation.py` and protocol specification in `outputs/results/case_evaluation_protocol.md` stand ready to ingest verified datasets containing `image_path`, `design_id`, and `colorway_id` without requiring architectural refactoring.

---

## Future Work
1. **Artisan Annotation Protocol:** Partner with handloom textile co-operatives to capture verified multi-colorway swatches of known registered design motifs.
2. **Hard Color Distractor Mining:** Compile negative pairs that share identical yarn dye palettes but differ in floral/geometric brocade weave pattern.
3. **Multi-Scale Weave Encoders:** Integrate high-resolution patch-level attention backbones to emphasize micro-texture and zari thread counts over macroscopic color dominance.

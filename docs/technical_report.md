# DeepLure / AIE-CASE: Comprehensive Technical Report
**Color-Invariant Saree Design Recognition**

---

## Explicit Equivalence Distinctions

Before reviewing technical specifications, the following scientific boundaries are explicitly defined:

$$\mathbf{pattern\_family \ne design\_id}$$
$$\mathbf{augmentation\_group \ne design\_id}$$
$$\mathbf{synthetic\ color\ transformation \ne authentic\ colorway}$$

* Regional pattern families (Banarasi, Bandhani, Ikat, Pichwai) are broad cultural weave traditions, NOT unique garment design models.
* Roboflow augmentation groups are synthetic salt-and-pepper duplicate clusters used exclusively for data leakage prevention across splits, NOT design identifiers.
* Synthetic digital color transformations (hue shift, jitter, desaturation) test mathematical embedding perturbation stability, NOT artisan dyeings across distinct textile colorways.

---

## 1. Problem Statement
Textile surface recognition requires identifying garments by surface weave patterns, brocades, and motifs independently of the color palette in which they are rendered. Traditional sarees feature intricate motifs (paisley jaal, floral butti, geometric ikat borders) woven in diverse colorways.

## 2. CASE Objective
The core objective of Color-Invariant Saree Design Recognition (AIE-CASE) is:
* **Same design in different colorways** should match.
* **Different designs in similar colorways** should not match.

## 3. Dataset Sources
The project corpus consists of 1,633 images across two archives in `data/`:
1. **Indian Saree Patterns:** 1,468 images across 4 motif styles (Roboflow export, $640 \times 640$ RGB).
2. **Handloom Saree Corpus:** 165 images of artisan sarees (unlabeled product inventory, variable resolutions).
*All original archives in `data/*.zip` remain pristine and unmodified.*

## 4. Dataset Inspection
* Zero corrupted images: all 1,633 images open successfully via PIL.
* Metadata audit: No external ground-truth CSV/JSON annotations or colorway labels existed in source archives.
* Filenames: Handloom files (`h_img_*.jpg`) and Indian pattern files represent arbitrary catalog or export filenames, not fine-grained design identifiers.

## 5. Dataset Statistics
* Total images: 1,633
* Supervised subset: 1,468
  - **Banarasi:** 489 images (33.31%)
  - **Ikat:** 342 images (23.30%)
  - **Pichwai:** 321 images (21.87%)
  - **Bandhani:** 316 images (21.53%)
* Unsupervised subset: 165 Handloom images (held out for zero-shot testing)

## 6. Data Cleaning
The data pipeline (`src/prepare_data.py`) validates channels, color modes, and aspect ratios. All images are converted to RGB. No fabricated labels are introduced.

## 7. Duplicate Handling
MD5 hash auditing identified 10 duplicate images forming 5 distinct duplicate groups. Exact duplicates are indexed and constrained to reside entirely within single split partitions.

## 8. Leakage Prevention
Stem matching identified 606 Roboflow augmentation clusters (`.rf.<hash>`). Connected components combining MD5 duplicates and Roboflow siblings were grouped into indivisible split units to ensure zero information leakage.

## 9. Train / Validation / Test Split
Split assignment was executed via `src/create_splits.py` (seed = 42) using a greedy balance heuristic:
* **Train:** 1,027 images (~70%)
* **Validation:** 221 images (~15%)
* **Test:** 220 images (~15%)
* **Handloom:** 165 images (excluded from supervised manifests)
*Verification (`src/verify_splits.py`): 0 MD5 leakage, 0 augmentation leakage, 0 path overlap.*

## 10. Pattern-Family Proxy Task
Because fine-grained design IDs are absent, supervision is strictly restricted to **Proxy Task A (pattern-family learning)**. This validates metric learning pipelines without corrupting ground truth.

## 11. Model Architecture
The dual-branch architecture decouples categorical classification from metric representation:

```text
RGB Saree Image
       │
       ▼
224×224 Preprocessing
       │
       ▼
EfficientNet-B0 Backbone
       │
       ▼
Feature Representation (1280-D)
       │
       ▼
256-D Embedding Head
       │
       ▼
BatchNorm + ReLU + Dropout(0.2)
       │
       ▼
L2 Normalization
       │
┌──────┴───────────────┐
↓                      ↓
Classification Branch   Cosine Similarity Branch
(Linear: 1280 -> 4)    (256-D Unit Hypersphere)
       │                      │
       ▼                      ▼
Pattern Family Proxy    Retrieval & Verification
```

## 12. EfficientNet-B0 Backbone
Initialized with ImageNet-1K pretrained weights (`EfficientNet_B0_Weights.IMAGENET1K_V1`). The backbone extracts general texture, edge, and weave structural representations efficiently on CPU.

## 13. 256-D Embedding Head
Consists of `Linear(1280, 256)` -> `BatchNorm1d(256)` -> `ReLU(inplace=True)` -> `Dropout(p=0.2)`. Projects high-dimensional feature maps into a compact, discriminative latent space.

## 14. L2 Normalization
Embedding vectors are normalized to unit Euclidean length:
$$\mathbf{e} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2 + \epsilon}$$
mapping representations to the 255-sphere $\mathbb{S}^{255}$, where Euclidean distance monotonically maps to cosine similarity:
$$\|\mathbf{e}_a - \mathbf{e}_b\|_2^2 = 2 - 2 \cos(\mathbf{e}_a, \mathbf{e}_b)$$

## 15. Classification Training (Experiment A)
Baseline classifier trained with categorical Cross-Entropy loss on pooled backbone features using AdamW ($\text{lr} = 10^{-4}$, weight decay $10^{-4}$). Achieved 60.91% test accuracy and 0.5829 Macro F1.

## 16. Metric-Learning Training (Experiment B)
Trained to map same-family sarees close together and different-family sarees far apart on the unit sphere. Combines batch-hard triplet loss with multi-task cross-entropy regularization.

## 17. Batch-Hard Triplet Loss Methodology
For each anchor $\mathbf{a}_i$, the hardest positive $\mathbf{p}_i = \arg\max_{\mathbf{p} \in \mathcal{P}_i} D(\mathbf{a}_i, \mathbf{p})$ and hardest negative $\mathbf{n}_i = \arg\min_{\mathbf{n} \in \mathcal{N}_i} D(\mathbf{a}_i, \mathbf{n})$ in the batch are selected:
$$\mathcal{L}_{\text{BHT}} = \frac{1}{N} \sum_{i=1}^N \max\left(0, \|\mathbf{e}_i - \mathbf{e}_p\|_2 - \|\mathbf{e}_i - \mathbf{e}_n\|_2 + \alpha\right)$$
with margin $\alpha = 0.2$.

## 18. Sampling Strategy
`PatternFamilyBatchSampler` (`src/sampler.py`) guarantees that every mini-batch contains balanced samples across all 4 pattern families, preventing triplet collapse and degenerate zero-gradient batches.

## 19. Augmentation Strategy
Transforms in `src/augmentations.py`:
* Geometric: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=10)`.
* Photometric: `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)`, `RandomGrayscale(p=0.1)`.
*Encourages structural weave invariance while maintaining geometric consistency.*

## 20. Retrieval Methodology
* Reference Gallery: Precomputed 256-D L2-normalized embeddings of the 1,027 training images.
* Query: Test queries are embedded and compared via exhaustive matrix inner products (cosine similarity).
* Self-exclusion: If a query image is in the gallery, it is excluded from its own candidate list.

## 21. Verification Methodology
Pairwise similarity is computed as the dot product $s = \mathbf{e}_1^\top \mathbf{e}_2$. If $s \ge \tau$, the pair is classified as SAME (same pattern family); otherwise DIFFERENT.

## 22. Threshold Selection Protocol
Threshold selection is performed **exclusively on validation pairs** (1,600 pairs). Sweeping $\tau \in [-0.5, 1.0]$ determined the threshold maximizing validation F1:
$$\tau^* = 0.2895 \quad (\text{Val F1} = 0.7015)$$
This threshold is frozen and applied to test pairs without test-label optimization.

## 23. Synthetic Color Robustness
Test images ($N=220$) are subjected to 6 controlled transformations:
* Color Jitter (mean sim = 0.7619 | Recall@1 = 84.55%)
* Light Hue Shift (mean sim = 0.7366 | Recall@1 = 83.64%)
* Palette Inversion (mean sim = 0.7104 | Recall@1 = 72.27%)
* Channel Swap (mean sim = 0.6770 | Recall@1 = 78.64%)
* Grayscale (mean sim = 0.6305 | Recall@1 = 84.55%)
* Heavy Hue Shift (mean sim = 0.5782 | Recall@1 = 73.64%)

## 24. Handloom Handling
The 165 artisanal handloom images are unannotated. They are preserved in `outputs/results/handloom_unlabeled.csv` and indexed in `outputs/embeddings/handloom_embeddings.npy`. Handloom sarees can be queried against the gallery for visual inspection, but are strictly excluded from supervised evaluation.

## 25. Efficiency Benchmarking
Measured on local CPU (`src/benchmark.py`):
* Total parameters: 4,337,024
* Trainable parameters: 4,337,024
* Model disk size: 19.35 MB (16.64 MB baseline)
* Single-image CPU latency: 68.59 ms
* Batch-32 CPU latency: 2,234.19 ms

## 26. Actual Results
* **Pattern-Family Retrieval:** Recall@1 = **87.73%**, Recall@3 = **91.82%**, Recall@5 = **93.64%**, Recall@10 = **97.27%**, MRR = **0.9079**.
* **Pattern-Family Verification:** Validation threshold = **0.2895**, Test Accuracy = **57.63%**, Test Precision = **54.78%**, Test Recall = **87.38%**, Test F1 = **0.6734**, Test ROC-AUC = **0.6529**.
* **Synthetic Color Robustness:** Mean similarity = **0.6824**, Median similarity = **0.6883**, Classification consistency = **54.39%**, Mean Recall@1 retention = **90.67%**.

## 27. Limitations
1. No verified fine-grained design IDs exist in the dataset.
2. No ground-truth colorway labels or verified same-design/different-color pairs exist.
3. High pattern-family retrieval does not equal fine-grained design identification.
4. Synthetic color transformations do not substitute for authentic dyeings across different colorways.
5. Handloom images are unlabeled and cannot be evaluated for supervised accuracy.

## 28. Future Work
1. Curate authentic ground-truth multi-colorway saree swatches with verified design identifiers.
2. Ingest curated datasets into the pre-built `src/case_evaluation.py` framework.
3. Train using Supervised Contrastive (SupCon) loss targeting cross-colorway positive pairs and same-colorway distractor negative pairs.

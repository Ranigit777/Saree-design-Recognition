# System Architecture Specification

**Project:** DeepLure / AIE-CASE — Color-Invariant Saree Design Recognition

---

## 1. Conceptual End-to-End Pipeline

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
Feature Representation
       │
       ▼
256-D Embedding Head
       │
       ▼
BatchNorm + ReLU + Dropout
       │
       ▼
L2 Normalization
       │
┌──────┴───────────────┐
↓                      ↓
Classification      Cosine Similarity
Branch                 ↓
↓                  Retrieval /
Pattern Family     Verification
Proxy Prediction
```

---

## 2. Component Explanations

### 2.1 RGB Saree Image
High-resolution raw photograph of a saree surface. Images from the Indian Saree Patterns dataset ($640 \times 640$ Roboflow export) and Handloom dataset (variable resolution) are accepted in standard formats (JPEG, PNG, WEBP).

### 2.2 224×224 Preprocessing
- **Resizing & Cropping:** Evaluated images are resized to $256 \times 256$ and center-cropped to $224 \times 224$ pixels. During training, `RandomResizedCrop(224, scale=(0.8, 1.0))` is paired with horizontal flipping and small rotations.
- **Normalization:** Standard ImageNet channel normalization with mean $\mu = [0.485, 0.456, 0.406]$ and standard deviation $\sigma = [0.229, 0.224, 0.225]$.

### 2.3 EfficientNet-B0 Backbone
- A lightweight convolutional neural network utilizing compound scaling across depth, width, and resolution.
- Initialized with ImageNet-1K pretrained weights (`EfficientNet_B0_Weights.IMAGENET1K_V1`).
- Operates with frozen weights during training on CPU, accelerating convergence and preventing catastrophic forgetting while preserving rich visual edge, texture, and weave features.

### 2.4 Feature Representation
- Global Average Pooling (GAP) collapses the final convolutional spatial feature map into a dense 1,280-dimensional feature vector.
- This serves as the shared foundational representation for both downstream branches.

### 2.5 256-D Embedding Head
- Projects the 1,280-D backbone features into a compact 256-dimensional metric latent space via a fully connected linear layer (`Linear(1280, 256)`).

### 2.6 BatchNorm + ReLU + Dropout
- **BatchNorm1d(256):** Stabilizes internal feature distributions and accelerates metric learning optimization.
- **ReLU(inplace=True):** Introduces non-linearity for manifold modeling.
- **Dropout(p=0.2):** Regularizes the metric head against overfitting on training motifs.

### 2.7 L2 Hypersphere Normalization
- All 256-D output vectors are projected onto the surface of the unit hypersphere $\mathbb{S}^{255}$:
  $$\mathbf{e} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2 + \epsilon}, \quad \|\mathbf{e}\|_2 = 1.0$$
- Eliminates vector magnitude variance, ensuring that inner product operations correspond directly to cosine similarity:
  $$\text{sim}(\mathbf{e}_a, \mathbf{e}_b) = \mathbf{e}_a^\top \mathbf{e}_b$$

### 2.8 Downstream Decision Branches

#### A. Classification Branch (Proxy Baseline)
- Multi-class linear layer (`Linear(1280, 4)`) mapping backbone features to class logits for the 4 proxy pattern families (Banarasi, Bandhani, Ikat, Pichwai).
- Trained with Cross-Entropy loss as Experiment A baseline (Test Accuracy: 60.91%, Macro F1: 0.5829).

#### B. Cosine Similarity Branch (Retrieval / Verification)
- **Retrieval Engine:** Precomputes L2-normalized vectors for the 1,027-image reference gallery. Queries compute matrix inner products $\mathbf{S} = \mathbf{e}_q \mathbf{G}^\top$ and return the Top-$K$ ranked nearest neighbors.
- **Pairwise Verification:** Computes cosine similarity $s = \mathbf{e}_1^\top \mathbf{e}_2$ and compares against the validation-selected decision threshold $\tau^* = 0.2895$. Returns `SAME PATTERN FAMILY` if $s \ge \tau^*$, else `DIFFERENT PATTERN FAMILY`.

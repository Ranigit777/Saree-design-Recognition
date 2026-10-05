# Proxy Task Definition (Phase 3)

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

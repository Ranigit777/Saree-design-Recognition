# CASE Evaluation Protocol Specification

## Purpose
This document defines the exact evaluation protocols and data schema for **AIE-CASE (Color-Invariant Saree Design Recognition)** when verified fine-grained design and colorway ground truth becomes available.

---

## 1. Required Manifest Schema

Future verified datasets must be provided as a CSV file with the following columns:

| Column Name | Type | Description |
|---|---|---|
| `image_path` | string | Relative path to image within workspace |
| `design_id` | string/int | Verified unique identifier of the surface motif/design |
| `colorway_id` | string/int | Identifier of the specific color palette/dye combination |
| `split` | string | Optional: `train`, `val`, `test`, `gallery`, `query` |

---

## 2. Evaluation Protocols

### Protocol 1: Cross-Colorway Same-Design Retrieval
- **Query:** Image $(D_i, C_a)$
- **Gallery:** All gallery images, including $(D_i, C_b)$ ($b \ne a$) and distractors $(D_j, C_k)$ ($j \ne i$).
- **Success Criterion:** The model successfully retrieves the matching design $D_i$ rendered in a *different* colorway $C_b$.
- **Reported Metrics:** `Recall@1`, `Recall@3`, `Recall@5`, `MRR`.

### Protocol 2: Color-Distractor Discrimination (The Core CASE Test)
- **Query:** $(D_i, C_a)$
- **Candidates Evaluated:**
  - True Target: $(D_i, C_b)$ [Same design, different colorway]
  - Color Imposter: $(D_j, C_a)$ [Different design, identical colorway]
- **Success Criterion:** $\text{sim}(D_i C_b, D_i C_a) > \text{sim}(D_j C_a, D_i C_a)$.
- **Significance:** Measures whether the model's representations are genuinely driven by geometric surface motifs rather than dominant palette colors.
- **Reported Metrics:** `case_discrimination_rate` (percentage of wins), `mean_design_over_color_margin`.

### Protocol 3: Hard-Pair Verification
- **Positive Pairs:** Same design across different colorways: $(D_i, C_a)$ vs $(D_i, C_b)$.
- **Hard Negative Pairs:** Different designs sharing identical colorways: $(D_i, C_a)$ vs $(D_j, C_a)$.
- **Reported Metrics:** `ROC-AUC`, `F1 Score`, `Accuracy` on hard pairs.

---

## 3. Running Future CASE Evaluations

Once a verified CSV is available:

```bash
python src/case_evaluation.py --manifest path/to/verified_case_manifest.csv
```

To run the schema/framework self-test:

```bash
python src/case_evaluation.py --mock
```

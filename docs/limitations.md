# Scientific Limitations & Data Integrity Disclosure

**Project:** DeepLure / AIE-CASE — Color-Invariant Saree Design Recognition

---

## 1. Core Scientific Limitations

To maintain uncompromising scientific honesty and engineering integrity, this document explicitly records the fundamental data limitations of the current dataset:

1. **No Verified Fine-Grained Design IDs:**
   The dataset contains broad regional motif families (**Banarasi**, **Bandhani**, **Ikat**, **Pichwai**), but no fine-grained design identifiers distinguishing specific jaal, floral border, or motif arrangements.

2. **No Verified Colorway IDs:**
   No standardized or calibrated color palette annotations exist in the source dataset. Filename-derived keywords are incomplete, uncalibrated, and non-authoritative.

3. **No Genuine Same-Design / Different-Colorway Pairs:**
   The dataset does not contain authentic photographs of the identical weave design rendered across different colorways or yarn dye lots (e.g., Design #42 in Red/Gold and Design #42 in Blue/Silver).

4. **Pattern Family is Strictly a Proxy Task:**
   All supervised classification, metric learning, retrieval, and verification experiments operate on **Proxy Task A (coarse pattern-family discrimination)**. High proxy retrieval accuracy does NOT equal fine-grained design identification.

5. **Synthetic Color Transformations are Not Authentic Colorways:**
   Controlled digital perturbations (ColorJitter, hue shift, inversion, desaturation) test mathematical embedding stability under numerical pixel shifts. They do NOT simulate physical yarn dye chemistry, metallic zari reflectance, or artisanal colorway variations.

6. **Handloom Images are Completely Unlabeled:**
   The 165 artisanal handloom images under `data/deeplure/` contain product catalog numbers (`img_<number>.jpg`) with no design or pattern annotations. They cannot be used to compute supervised accuracy.

7. **True CASE Identification and Verification Cannot Currently Be Claimed:**
   Because neither verified design IDs nor verified multi-colorway pairs exist in the dataset, validated color-invariant fine-grained design recognition cannot be claimed.

---

## 2. Fundamental Equivalence Distinctions

$$\mathbf{pattern\_family \ne design\_id}$$
$$\mathbf{augmentation\_group \ne design\_id}$$
$$\mathbf{synthetic\ color\ transformation \ne authentic\ colorway}$$

* **Pattern Family $\ne$ Design ID:** A pattern family encompasses thousands of distinct designs. Matching two Banarasi sarees does not mean they share the same design.
* **Augmentation Group $\ne$ Design ID:** Roboflow augmentation clusters (`.rf.<hash>`) represent synthetic noise variants of single source photos used strictly to prevent split leakage, not different design variants or colorways.
* **Synthetic Color Transformation $\ne$ Authentic Colorway:** Inverting an RGB tensor or shifting hue digitally does not represent the complex spectral reflectance of genuine artisanal handloom textiles.

---

## 3. Required Data Specification for True CASE Evaluation

To complete authentic validation of the AIE-CASE objective, a newly curated dataset with the following structure is required:

### Required Metadata Schema (CSV / JSON)
| Column Name | Type | Description | Example |
|---|---|---|---|
| `image_path` | string | Relative file path to the high-resolution saree image | `data/verified/d101_c01.jpg` |
| `design_id` | string | Unique registered design identifier for the surface motif | `design_101` |
| `colorway_id` | string | Standardized colorway / dye lot identifier | `color_crimson_gold` |
| `palette_family` | string | Broad dominant color category (for distractor mining) | `red` |
| `craft_tradition` | string | Weaving tradition / craft form | `Banarasi` |

### Required Dataset Properties
1. **Multi-Colorway Same-Design Sets:** At least $K \ge 2$ distinct colorway images for every `design_id`.
2. **Color-Distractor Sets:** Pairs of *different* designs sharing identical or near-identical color palettes (to test that the model does not match purely on color).
3. **Hard Verification Pairs:** Curated evaluation pairs:
   - Positive pairs: Same `design_id`, different `colorway_id`.
   - Negative distractors: Different `design_id`, same `colorway_id`.

---

## 4. Architectural Readiness for Future Data

The codebase is engineered to immediately support this data:
* **CASE Evaluation Module:** `src/case_evaluation.py` implements all 3 CASE evaluation protocols (Cross-Colorway Retrieval, Color-Distractor Discrimination, and Hard-Pair Verification).
* **Graceful Fallback:** When run without verified annotations, `case_evaluation.json` explicitly outputs `CASE evaluation: NOT AVAILABLE` rather than fabricating synthetic or proxy metrics.

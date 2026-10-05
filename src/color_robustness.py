"""
Synthetic color-robustness experiment.

Controlled transforms (brightness, contrast, saturation, hue, grayscale) are
applied to test images. Embedding cosine similarity and classifier consistency
are measured against the original image.

This is NOT verified same-design/different-color ground truth and is not CASE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform
from config import (
    BEST_CLASSIFIER_PATH,
    BEST_METRIC_MODEL_PATH,
    COLOR_ROBUSTNESS_METRICS_PATH,
    COLOR_ROBUSTNESS_PLOT,
    COLOR_ROBUSTNESS_RESULTS_PATH,
    LABEL_TO_PATTERN_FAMILY,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TEST_MANIFEST_PATH,
    TRAIN_CONFIG,
)
from loaders import get_device
from model import SareeClassifier, SareeEmbeddingModel


def transform_identity(img: Image.Image) -> Image.Image:
    return img


def transform_brightness(img: Image.Image) -> Image.Image:
    return TF.adjust_brightness(img, 1.4)


def transform_contrast(img: Image.Image) -> Image.Image:
    return TF.adjust_contrast(img, 1.4)


def transform_saturation(img: Image.Image) -> Image.Image:
    return TF.adjust_saturation(img, 1.6)


def transform_hue(img: Image.Image) -> Image.Image:
    return TF.adjust_hue(img, 0.15)


def transform_grayscale(img: Image.Image) -> Image.Image:
    return TF.to_grayscale(img, num_output_channels=3)


TRANSFORMS: dict[str, tuple[str, Callable[[Image.Image], Image.Image]]] = {
    "brightness": ("Brightness increase (factor 1.4)", transform_brightness),
    "contrast": ("Contrast increase (factor 1.4)", transform_contrast),
    "saturation": ("Saturation increase (factor 1.6)", transform_saturation),
    "hue": ("Hue shift (+0.15)", transform_hue),
    "grayscale": ("Grayscale (luminance only)", transform_grayscale),
}


def _load_metric_model(device: torch.device) -> SareeEmbeddingModel:
    if not BEST_METRIC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Metric checkpoint missing: {BEST_METRIC_MODEL_PATH}")
    ckpt = torch.load(BEST_METRIC_MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = SareeEmbeddingModel(
        embedding_dim=int(cfg.get("embedding_dim", TRAIN_CONFIG.embedding_dim)),
        pretrained=False,
        dropout=float(cfg.get("dropout", TRAIN_CONFIG.dropout)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_classifier(device: torch.device) -> SareeClassifier:
    if not BEST_CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"Classifier checkpoint missing: {BEST_CLASSIFIER_PATH}")
    ckpt = torch.load(BEST_CLASSIFIER_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = SareeClassifier(
        pretrained=False,
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_and_classify(
    images: list[Image.Image],
    metric_model: SareeEmbeddingModel,
    classifier: SareeClassifier,
    eval_tf,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = torch.stack([eval_tf(im) for im in images]).to(device)
    emb = metric_model(tensors, return_logits=False)["embedding"].cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    logits = classifier(tensors)
    preds = logits.argmax(dim=1).cpu().numpy()
    return emb, preds


def evaluate_color_robustness(
    test_manifest_path: Path = TEST_MANIFEST_PATH,
    batch_size: int = 16,
) -> dict:
    """
    Stream test images from disk. For each required transform, compare original vs
    transformed embeddings and classifier predictions.
    """
    device = get_device()
    print(f"[Color Robustness] Device: {device}")
    metric_model = _load_metric_model(device)
    classifier = _load_classifier(device)
    eval_tf = build_eval_transform()
    test_df = pd.read_csv(test_manifest_path)

    orig_embs: list[np.ndarray] = []
    orig_preds: list[np.ndarray] = []
    tfm_store: dict[str, dict[str, list]] = {
        name: {"embs": [], "preds": []} for name in TRANSFORMS
    }

    paths = [PROJECT_ROOT / row["image_path"] for _, row in test_df.iterrows()]
    print(f"[Color Robustness] Streaming {len(paths)} test images (batch={batch_size})...")

    for start in tqdm(range(0, len(paths), batch_size), desc="color-robust"):
        batch_paths = paths[start : start + batch_size]
        originals: list[Image.Image] = []
        for p in batch_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Missing test image: {p}")
            with Image.open(p) as im:
                originals.append(im.convert("RGB"))

        o_emb, o_pred = embed_and_classify(originals, metric_model, classifier, eval_tf, device)
        orig_embs.append(o_emb)
        orig_preds.append(o_pred)

        for name, (_desc, fn) in TRANSFORMS.items():
            transformed = [fn(im.copy()) for im in originals]
            t_emb, t_pred = embed_and_classify(
                transformed, metric_model, classifier, eval_tf, device
            )
            tfm_store[name]["embs"].append(t_emb)
            tfm_store[name]["preds"].append(t_pred)

        del originals

    orig_embs_np = np.concatenate(orig_embs, axis=0)
    orig_preds_np = np.concatenate(orig_preds, axis=0)

    transform_results = {}
    all_sims: list[float] = []
    all_consistent: list[int] = []

    for name, (desc, _fn) in TRANSFORMS.items():
        t_embs = np.concatenate(tfm_store[name]["embs"], axis=0)
        t_preds = np.concatenate(tfm_store[name]["preds"], axis=0)
        sims = np.sum(orig_embs_np * t_embs, axis=1)
        consistent = (t_preds == orig_preds_np).astype(int)
        all_sims.extend(sims.tolist())
        all_consistent.extend(consistent.tolist())
        pred_families = [LABEL_TO_PATTERN_FAMILY[int(i)] for i in t_preds[:5]]
        transform_results[name] = {
            "description": desc,
            "mean_cosine_similarity": round(float(np.mean(sims)), 4),
            "median_cosine_similarity": round(float(np.median(sims)), 4),
            "std_cosine_similarity": round(float(np.std(sims)), 4),
            "classification_consistency_pct": round(float(100.0 * np.mean(consistent)), 2),
            "n_images": int(len(sims)),
            "sample_predicted_families": pred_families,
        }

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(transform_results.keys())
    mean_sims = [transform_results[n]["mean_cosine_similarity"] for n in names]
    cons = [transform_results[n]["classification_consistency_pct"] / 100.0 for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width / 2, mean_sims, width, label="Mean cosine similarity", color="tab:blue", alpha=0.85)
    ax1.set_ylabel("Mean cosine similarity")
    ax1.set_ylim(0, 1.1)
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, cons, width, label="Classifier consistency", color="tab:orange", alpha=0.85)
    ax2.set_ylabel("Classification consistency")
    ax2.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_title("Synthetic color-robustness experiment (test set)")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(COLOR_ROBUSTNESS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    overall_mean = float(np.mean(all_sims))
    overall_median = float(np.median(all_sims))
    overall_cons = float(100.0 * np.mean(all_consistent))

    report = {
        "task": "Synthetic Color Robustness Evaluation",
        "status": "completed",
        "total_test_images_evaluated": int(len(test_df)),
        "scientific_integrity_statement": (
            "This is a synthetic color-robustness experiment and is not a substitute "
            "for verified same-design/different-color ground truth."
        ),
        "overall_summary": {
            "mean_cosine_similarity": round(overall_mean, 4),
            "median_cosine_similarity": round(overall_median, 4),
            "mean_classification_consistency_pct": round(overall_cons, 2),
        },
        "transformations": transform_results,
    }
    with open(COLOR_ROBUSTNESS_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(COLOR_ROBUSTNESS_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def analyze_image_color_robustness(image_path: str | Path, device: torch.device | None = None) -> dict:
    """Per-image color robustness for inference CLI."""
    device = device or get_device()
    metric_model = _load_metric_model(device)
    classifier = _load_classifier(device)
    eval_tf = build_eval_transform()
    p = Path(image_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as im:
        original = im.convert("RGB")
    o_emb, o_pred = embed_and_classify([original], metric_model, classifier, eval_tf, device)
    rows = []
    for name, (desc, fn) in TRANSFORMS.items():
        t_emb, t_pred = embed_and_classify([fn(original.copy())], metric_model, classifier, eval_tf, device)
        sim = float(np.sum(o_emb * t_emb))
        rows.append(
            {
                "transform": name,
                "description": desc,
                "cosine_similarity": round(sim, 4),
                "original_predicted_family": LABEL_TO_PATTERN_FAMILY[int(o_pred[0])],
                "transformed_predicted_family": LABEL_TO_PATTERN_FAMILY[int(t_pred[0])],
                "classification_consistent": bool(int(t_pred[0]) == int(o_pred[0])),
            }
        )
    return {
        "image_path": str(p),
        "scientific_integrity_statement": (
            "This is a synthetic color-robustness experiment and is not a substitute "
            "for verified same-design/different-color ground truth."
        ),
        "transforms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic color robustness experiment")
    parser.parse_args()
    report = evaluate_color_robustness()
    print(json.dumps(report["overall_summary"], indent=2))
    print(report["scientific_integrity_statement"])


if __name__ == "__main__":
    main()

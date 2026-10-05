"""
Extract L2-normalized embeddings from the best metric-learning checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform
from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    EMBEDDING_STATISTICS_PATH,
    HANDLOOM_UNLABELED_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    TRAIN_CONFIG,
    VAL_MANIFEST_PATH,
)
from dataset import SareeDataset
from loaders import collate_saree_batch, get_device
from model import SareeEmbeddingModel


def load_metric_model(device: torch.device) -> SareeEmbeddingModel:
    if not BEST_METRIC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {BEST_METRIC_MODEL_PATH}")
    ckpt = torch.load(BEST_METRIC_MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", TRAIN_CONFIG.to_dict())
    model = SareeEmbeddingModel(
        embedding_dim=int(cfg.get("embedding_dim", TRAIN_CONFIG.embedding_dim)),
        pretrained=False,
        dropout=float(cfg.get("dropout", TRAIN_CONFIG.dropout)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_manifest(manifest_path: Path, model, device, batch_size: int = 32) -> tuple[np.ndarray, pd.DataFrame]:
    ds = SareeDataset(manifest_path, transform=build_eval_transform())
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_saree_batch, num_workers=0
    )
    embs = []
    meta_rows = []
    for batch in tqdm(loader, desc=f"embed {manifest_path.name}"):
        images = batch["image"].to(device)
        out = model(images, return_logits=False)
        embs.append(out["embedding"].detach().cpu().numpy())
        for i, path in enumerate(batch["image_path"]):
            md = {k: batch["metadata"][k][i] for k in batch["metadata"]}
            meta_rows.append(
                {
                    "image_path": path,
                    "pattern_family": md.get("pattern_family"),
                    "dataset_source": batch["metadata"].get("dataset_source", [None])[i]
                    if "dataset_source" in batch["metadata"]
                    else None,
                    "original_split": md.get("original_split"),
                    "augmentation_group": md.get("augmentation_group"),
                    "duplicate_group": md.get("duplicate_group"),
                    "label": int(batch["label"][i].item()),
                }
            )
    manifest_df = pd.read_csv(manifest_path)
    meta_df = pd.DataFrame(meta_rows)
    if "dataset_source" not in meta_df.columns or meta_df["dataset_source"].isna().all():
        meta_df = meta_df.drop(columns=["dataset_source"], errors="ignore")
        meta_df = meta_df.merge(
            manifest_df[
                [
                    "image_path",
                    "dataset_source",
                    "pattern_family",
                    "original_split",
                    "augmentation_group",
                    "duplicate_group",
                ]
            ],
            on="image_path",
            how="left",
            suffixes=("_drop", ""),
        )
    return np.concatenate(embs), meta_df


@torch.no_grad()
def extract_handloom(model, device, batch_size: int = 32) -> tuple[np.ndarray, pd.DataFrame] | None:
    if not HANDLOOM_UNLABELED_PATH.is_file():
        return None
    from PIL import Image, UnidentifiedImageError
    from config import PROJECT_ROOT, IMAGE_SIZE
    from torchvision import transforms

    df = pd.read_csv(HANDLOOM_UNLABELED_PATH)
    tfm = build_eval_transform()
    embs = []
    rows = []
    batch_imgs = []
    batch_meta = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        out = model(x, return_logits=False)
        embs.append(out["embedding"].detach().cpu().numpy())
        rows.extend(batch_meta)
        batch_imgs.clear()
        batch_meta.clear()

    for _, row in df.iterrows():
        path = PROJECT_ROOT / row["image_path"]
        try:
            with Image.open(path) as im:
                img = tfm(im.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to load handloom image: {path}") from exc
        batch_imgs.append(img)
        batch_meta.append(
            {
                "image_path": row["image_path"],
                "pattern_family": "unknown",
                "dataset_source": row.get("dataset_source", "handloom"),
                "original_split": row.get("original_split", ""),
                "augmentation_group": None,
                "duplicate_group": None,
            }
        )
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return np.concatenate(embs), pd.DataFrame(rows)


def save_embedding_stats(all_embs: list[np.ndarray]) -> dict:
    embs = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(embs, axis=1)
    stats = {
        "num_vectors": int(embs.shape[0]),
        "embedding_dimension": int(embs.shape[1]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "approximately_l2_normalized": bool(np.allclose(norms, 1.0, atol=1e-2)),
    }
    with open(EMBEDDING_STATISTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def run_extraction() -> dict:
    device = get_device()
    model = load_metric_model(device)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {}
    all_embs = []
    for name, path in (
        ("train", TRAIN_MANIFEST_PATH),
        ("val", VAL_MANIFEST_PATH),
        ("test", TEST_MANIFEST_PATH),
    ):
        embs, meta = extract_manifest(path, model, device)
        np.save(EMBEDDINGS_DIR / f"{name}_embeddings.npy", embs)
        meta.to_csv(EMBEDDINGS_DIR / f"{name}_metadata.csv", index=False)
        outputs[name] = embs.shape
        all_embs.append(embs)

    handloom = extract_handloom(model, device)
    if handloom is not None:
        h_emb, h_meta = handloom
        np.save(EMBEDDINGS_DIR / "handloom_embeddings.npy", h_emb)
        h_meta.to_csv(EMBEDDINGS_DIR / "handloom_metadata.csv", index=False)
        all_embs.append(h_emb)

    stats = save_embedding_stats(all_embs)
    return {"shapes": outputs, "norm_stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_extraction()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
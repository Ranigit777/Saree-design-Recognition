"""
Experiment B: pattern-family metric learning (256-D embeddings + batch-hard triplet loss).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_METRIC_MODEL_PATH,
    LABEL_TO_PATTERN_FAMILY,
    MODELS_DIR,
    PATTERN_FAMILY_TO_LABEL,
    TRAIN_CONFIG,
    TrainConfig,
)
from loaders import build_loaders, collate_saree_batch, get_device
from losses import TripletLoss
from metrics_utils import pattern_family_retrieval_metrics
from model import SareeEmbeddingModel
from utils import set_seed


@torch.no_grad()
def embed_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, return_logits=False)
        embs.append(out["embedding"].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(embs), np.concatenate(labels)


def val_retrieval_metrics(model, val_loader, device) -> dict:
    embs, labels = embed_loader(model, val_loader, device)
    sim = embs @ embs.T
    np.fill_diagonal(sim, -np.inf)
    return pattern_family_retrieval_metrics(labels, labels, sim, ks=(1, 3, 4, 5, 10))


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    n = 0
    for batch in tqdm(loader, leave=False, desc="metric-train" if train else "metric-val"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            out = model(images, return_logits=False)
            loss = criterion(out["embedding"], labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = images.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def smoke_test(device: torch.device, cfg: TrainConfig) -> None:
    from augmentations import build_train_transform
    from config import TRAIN_MANIFEST_PATH
    from dataset import SareeDataset
    from torch.utils.data import DataLoader
    from sampler import PatternFamilyBatchSampler

    ds = SareeDataset(TRAIN_MANIFEST_PATH, transform=build_train_transform())
    sampler = PatternFamilyBatchSampler(ds, batch_size=16, samples_per_family=4, families_per_batch=4)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_saree_batch)
    batch = next(iter(loader))
    model = SareeEmbeddingModel(
        pretrained=False,
        embedding_dim=cfg.embedding_dim,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    out = model(images, return_logits=True)
    loss = TripletLoss()(out["embedding"], labels)
    loss.backward()
    opt.step()
    print(
        "[smoke] input:", tuple(images.shape),
        "embedding:", tuple(out["embedding"].shape),
        "logits:", tuple(out["logits"].shape),
        "loss:", float(loss.detach()),
    )


def train_metric(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TRAIN_CONFIG
    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, _ = build_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_family_batch_sampler=True,
        samples_per_family=max(2, cfg.batch_size // 4),
        families_per_batch=4,
    )
    model = SareeEmbeddingModel(
        embedding_dim=cfg.embedding_dim,
        pretrained=cfg.pretrained_backbone,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
    ).to(device)
    criterion = TripletLoss(margin=cfg.triplet_margin, p=cfg.triplet_p)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = []
    best_recall1 = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)
        retrieval = val_retrieval_metrics(model, val_loader, device)
        r1 = retrieval["pattern_family_recall_at_1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **retrieval})
        print(
            f"Metric epoch {epoch}/{cfg.epochs} | train {train_loss:.4f} val {val_loss:.4f} "
            f"val_R@1 {r1:.4f} MRR {retrieval['mrr']:.4f}"
        )
        if r1 > best_recall1 or best_epoch == 0:
            best_recall1 = r1
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_recall1,
                    "config": cfg.to_dict(),
                    "class_to_idx": PATTERN_FAMILY_TO_LABEL,
                    "idx_to_class": LABEL_TO_PATTERN_FAMILY,
                    "task": "pattern_family_metric_learning",
                },
                BEST_METRIC_MODEL_PATH,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping metric model at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(MODELS_DIR / "metric_history.csv", index=False)
    return {"best_epoch": best_epoch, "best_val_recall_at_1": best_recall1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    device = get_device()
    smoke_test(device, cfg)
    if not args.smoke_only:
        train_metric(cfg)


if __name__ == "__main__":
    main()
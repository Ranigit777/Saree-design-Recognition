"""
Experiment A: pattern-family classification baseline (EfficientNet-B0 + CrossEntropy).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_CLASSIFIER_PATH,
    CLASSIFIER_HISTORY_PATH,
    LABEL_TO_PATTERN_FAMILY,
    MODELS_DIR,
    PATTERN_FAMILY_TO_LABEL,
    TRAIN_CONFIG,
    TrainConfig,
)
from loaders import build_loaders, collate_saree_batch, get_device
from metrics_utils import classification_metrics
from model import SareeClassifier
from utils import set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, dict]:
    model.train(train)
    total_loss = 0.0
    all_y, all_pred = [], []
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_y.extend(labels.cpu().numpy())
        all_pred.extend(preds)
    n = len(all_y)
    metrics = classification_metrics(
        __import__("numpy").array(all_y), __import__("numpy").array(all_pred)
    )
    return total_loss / max(n, 1), metrics


def smoke_test(device: torch.device, cfg: TrainConfig) -> None:
    from augmentations import build_train_transform
    from config import TRAIN_MANIFEST_PATH
    from dataset import SareeDataset
    from torch.utils.data import DataLoader

    ds = SareeDataset(TRAIN_MANIFEST_PATH, transform=build_train_transform())
    loader = DataLoader(ds, batch_size=min(8, cfg.batch_size), collate_fn=collate_saree_batch)
    batch = next(iter(loader))
    model = SareeClassifier(pretrained=False, freeze_backbone=cfg.freeze_backbone).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    opt.step()
    print("[smoke] input:", tuple(images.shape), "logits:", tuple(logits.shape), "loss:", float(loss.detach()))


def train_classifier(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TRAIN_CONFIG
    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, _ = build_loaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    model = SareeClassifier(
        pretrained=cfg.pretrained_backbone,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    history_rows = []
    best_val_f1 = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_m = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_m["accuracy"],
                "val_accuracy": val_m["accuracy"],
                "train_precision": train_m["macro_precision"],
                "val_precision": val_m["macro_precision"],
                "train_recall": train_m["macro_recall"],
                "val_recall": val_m["macro_recall"],
                "train_f1": train_m["macro_f1"],
                "val_f1": val_m["macro_f1"],
                "learning_rate": lr,
            }
        )
        print(
            f"Epoch {epoch}/{cfg.epochs} | train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
            f"val_acc {val_m['accuracy']:.4f} val_f1 {val_m['macro_f1']:.4f}"
        )
        if val_m["macro_f1"] > best_val_f1 or best_epoch == 0:
            best_val_f1 = val_m["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_val_f1,
                    "config": cfg.to_dict(),
                    "class_to_idx": PATTERN_FAMILY_TO_LABEL,
                    "idx_to_class": LABEL_TO_PATTERN_FAMILY,
                    "task": "pattern_family_classification",
                },
                BEST_CLASSIFIER_PATH,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history_rows).to_csv(CLASSIFIER_HISTORY_PATH, index=False)
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_val_f1}


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
        train_classifier(cfg)


if __name__ == "__main__":
    main()
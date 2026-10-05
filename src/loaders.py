"""
DataLoader factories for train/validation/test manifests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import build_eval_transform, build_train_transform
from config import (
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from dataset import SareeDataset
from sampler import PatternFamilyBatchSampler


def collate_saree_batch(batch: list[dict]) -> dict:
    """Collate batch; keep string metadata as lists (default_collate fails on mixed NaN/str)."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    image_paths = [b["image_path"] for b in batch]
    metadata = {key: [b["metadata"][key] for b in batch] for key in batch[0]["metadata"]}
    return {
        "image": images,
        "label": labels,
        "image_path": image_paths,
        "metadata": metadata,
    }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    use_family_batch_sampler: bool = False,
    samples_per_family: int = 2,
    families_per_batch: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.device]:
    device = get_device()
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    train_ds = SareeDataset(TRAIN_MANIFEST_PATH, transform=build_train_transform())
    val_ds = SareeDataset(VAL_MANIFEST_PATH, transform=build_eval_transform())
    test_ds = SareeDataset(TEST_MANIFEST_PATH, transform=build_eval_transform())

    if use_family_batch_sampler:
        batch_sampler = PatternFamilyBatchSampler(
            train_ds,
            batch_size=batch_size,
            samples_per_family=samples_per_family,
            families_per_batch=families_per_batch,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_saree_batch,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_saree_batch,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_saree_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_saree_batch,
    )

    print(f"Detected device: {device}")
    return train_loader, val_loader, test_loader, device


if __name__ == "__main__":
    train_loader, val_loader, test_loader, device = build_loaders(batch_size=16)
    batch = next(iter(train_loader))
    print("train batch image shape:", batch["image"].shape)
    print("train batches:", len(train_loader), "val:", len(val_loader), "test:", len(test_loader))

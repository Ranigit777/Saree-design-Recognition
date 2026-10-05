"""
PyTorch Dataset for leakage-safe split manifests (Indian pattern families only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from config import PATTERN_FAMILY_TO_LABEL, PROJECT_ROOT


class SareeDataset(Dataset):
    """
    Loads images from a split manifest CSV.

    Returns pattern-family label (proxy supervision), not fine-grained design ID.
    Handloom images are excluded; use handloom_unlabeled.csv separately.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_csv)
        self.transform = transform
        self.df = pd.read_csv(self.manifest_path)
        self._validate()

    def _validate(self) -> None:
        if self.df.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        unknown = self.df[~self.df["pattern_family"].isin(PATTERN_FAMILY_TO_LABEL.keys())]
        if not unknown.empty:
            raise ValueError(
                f"Manifest contains non-pattern-family labels (e.g. handloom): "
                f"{self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = PROJECT_ROOT / row["image_path"]
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor

            image = to_tensor(image)

        label = PATTERN_FAMILY_TO_LABEL[row["pattern_family"]]
        metadata = {
            "dataset_source": row.get("dataset_source"),
            "original_split": row.get("original_split"),
            "heuristic_color": row.get("heuristic_color"),
            "augmentation_group": row.get("augmentation_group"),
            "duplicate_group": row.get("duplicate_group"),
            "md5": row.get("md5"),
            "split": row.get("split"),
            "pattern_family": row["pattern_family"],
        }
        return {
            "image": image,
            "label": label,
            "image_path": str(row["image_path"]),
            "metadata": metadata,
        }

"""
Phase 3: Leakage-safe grouped train/validation/test splits for Indian patterns.

Groups are formed by union of identical MD5 hashes and heuristic augmentation_group IDs.
Handloom images are exported separately and excluded from supervised splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AUGMENTATION_GROUPS_PATH,
    HANDLOOM_UNLABELED_PATH,
    MASTER_DATASET_PATH,
    PATTERN_FAMILIES,
    RESULTS_DIR,
    SPLIT_RANDOM_SEED,
    SPLIT_TARGETS,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from utils import set_seed


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_merged_table() -> pd.DataFrame:
    master = pd.read_csv(MASTER_DATASET_PATH)
    aug = pd.read_csv(AUGMENTATION_GROUPS_PATH)
    aug = aug.rename(columns={"augmentation_group": "augmentation_group_id"})
    merged = master.merge(aug[["image_path", "augmentation_group_id"]], on="image_path", how="left")
    merged["md5_group"] = merged["md5"]
    return merged


def build_split_groups(indian: pd.DataFrame) -> pd.DataFrame:
    uf = UnionFind()
    md5_to_paths: dict[str, list[str]] = defaultdict(list)
    aug_to_paths: dict[str, list[str]] = defaultdict(list)

    for _, row in indian.iterrows():
        path = row["image_path"]
        uf.add(path)
        if pd.notna(row["md5"]) and row["md5"]:
            md5_to_paths[row["md5"]].append(path)
        if pd.notna(row.get("augmentation_group_id")) and row.get("augmentation_group_id"):
            aug_to_paths[row["augmentation_group_id"]].append(path)

    for paths in md5_to_paths.values():
        for p in paths[1:]:
            uf.union(paths[0], p)
    for paths in aug_to_paths.values():
        for p in paths[1:]:
            uf.union(paths[0], p)

    indian = indian.copy()
    indian["split_group_id"] = indian["image_path"].map(lambda p: uf.find(p))
    return indian


def assign_splits(indian: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test at split_group_id level (~70/15/15 by image count)."""
    groups: dict[str, pd.DataFrame] = {
        gid: gdf for gid, gdf in indian.groupby("split_group_id")
    }
    group_ids = list(groups.keys())
    group_sizes = {gid: len(groups[gid]) for gid in group_ids}
    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    rng.shuffle(group_ids)

    total_images = len(indian)
    target_counts = {k: SPLIT_TARGETS[k] * total_images for k in SPLIT_TARGETS}
    split_counts = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    split_order = ("train", "val", "test")

    for gid in group_ids:
        size = group_sizes[gid]
        chosen: str | None = None

        for split_name in split_order:
            if split_counts[split_name] + size <= target_counts[split_name]:
                chosen = split_name
                break

        if chosen is None:
            # All targets would be exceeded: assign to split with lowest fill ratio.
            def fill_ratio(s: str) -> float:
                return split_counts[s] / max(target_counts[s], 1)

            chosen = min(split_order, key=fill_ratio)

        assignments[gid] = chosen
        split_counts[chosen] += size

    indian = indian.copy()
    indian["split"] = indian["split_group_id"].map(assignments)
    return indian


def export_handloom(handloom: pd.DataFrame) -> None:
    cols = [
        "image_path",
        "design_id",
        "dataset_source",
        "original_split",
        "pattern_family",
        "width",
        "height",
        "md5",
        "heuristic_color",
    ]
    out = handloom.copy()
    out["design_id"] = None
    out["pattern_family"] = "unknown"
    out[cols].to_csv(HANDLOOM_UNLABELED_PATH, index=False)


def export_manifest(indian: pd.DataFrame, split_name: str, path: Path) -> None:
    subset = indian[indian["split"] == split_name].copy()
    subset["augmentation_group"] = subset["augmentation_group_id"]
    cols = [
        "image_path",
        "dataset_source",
        "original_split",
        "pattern_family",
        "heuristic_color",
        "augmentation_group",
        "duplicate_group",
        "md5",
        "split",
    ]
    subset[cols].to_csv(path, index=False)


def run_create_splits() -> dict:
    set_seed(SPLIT_RANDOM_SEED)
    merged = load_merged_table()

    handloom = merged[merged["dataset_source"] == "handloom"].copy()
    indian = merged[merged["dataset_source"] == "indian_patterns"].copy()

    if indian["augmentation_group_id"].isna().any():
        missing = int(indian["augmentation_group_id"].isna().sum())
        raise RuntimeError(f"Indian images missing augmentation_group: {missing}")

    indian = build_split_groups(indian)
    indian = assign_splits(indian)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    export_handloom(handloom)
    export_manifest(indian, "train", TRAIN_MANIFEST_PATH)
    export_manifest(indian, "val", VAL_MANIFEST_PATH)
    export_manifest(indian, "test", TEST_MANIFEST_PATH)

    split_stats = {
        "random_state": SPLIT_RANDOM_SEED,
        "split_targets": SPLIT_TARGETS,
        "indian_image_counts": indian["split"].value_counts().astype(int).to_dict(),
        "indian_group_counts": indian.groupby("split")["split_group_id"].nunique().astype(int).to_dict(),
        "actual_split_percentages": {
            k: round(v / len(indian), 4)
            for k, v in indian["split"].value_counts().items()
        },
        "handloom_count": int(len(handloom)),
    }
    return {"indian": indian, "handloom": handloom, "stats": split_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe dataset splits")
    parser.parse_args()
    result = run_create_splits()
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()

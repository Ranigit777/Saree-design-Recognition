"""
Verify train/val/test manifests for MD5, augmentation-group, and path leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RESULTS_DIR,
    SPLIT_VERIFICATION_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from create_splits import run_create_splits


def load_all_manifests() -> pd.DataFrame:
    frames = []
    for path, split in (
        (TRAIN_MANIFEST_PATH, "train"),
        (VAL_MANIFEST_PATH, "val"),
        (TEST_MANIFEST_PATH, "test"),
    ):
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_leakage(df: pd.DataFrame) -> dict:
    md5_to_splits: dict[str, set[str]] = defaultdict(set)
    aug_to_splits: dict[str, set[str]] = defaultdict(set)
    path_to_splits: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        split = row["split"]
        path = row["image_path"]
        path_to_splits[path].add(split)
        if pd.notna(row["md5"]) and row["md5"]:
            md5_to_splits[row["md5"]].add(split)
        if pd.notna(row["augmentation_group"]) and row["augmentation_group"]:
            aug_to_splits[row["augmentation_group"]].add(split)

    md5_leak = [k for k, s in md5_to_splits.items() if len(s) > 1]
    aug_leak = [k for k, s in aug_to_splits.items() if len(s) > 1]
    path_leak = [k for k, s in path_to_splits.items() if len(s) > 1]

    family_distribution: dict[str, dict[str, float]] = {}
    source_distribution: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        fam_counts = sub["pattern_family"].value_counts().astype(int).to_dict()
        total = len(sub)
        family_distribution[split] = {
            k: {"count": v, "pct": round(100.0 * v / total, 2) if total else 0.0}
            for k, v in fam_counts.items()
        }
        source_distribution[split] = sub["dataset_source"].value_counts().astype(int).to_dict()

    return {
        "md5_leakage": len(md5_leak) > 0,
        "augmentation_leakage": len(aug_leak) > 0,
        "path_leakage": len(path_leak) > 0,
        "md5_leakage_examples": md5_leak[:10],
        "augmentation_leakage_examples": aug_leak[:10],
        "path_leakage_examples": path_leak[:10],
        "train_count": int((df["split"] == "train").sum()),
        "val_count": int((df["split"] == "val").sum()),
        "test_count": int((df["split"] == "test").sum()),
        "family_distribution": family_distribution,
        "source_distribution": source_distribution,
    }


def print_report(report: dict) -> None:
    print("Split verification")
    print(f"  train: {report['train_count']}  val: {report['val_count']}  test: {report['test_count']}")
    print(f"  md5_leakage: {report['md5_leakage']}")
    print(f"  augmentation_leakage: {report['augmentation_leakage']}")
    print(f"  path_leakage: {report['path_leakage']}")
    for split in ("train", "val", "test"):
        print(f"  {split} pattern families:", report["family_distribution"].get(split, {}))


def run_verify(recreate_if_leakage: bool = True) -> dict:
    if not TRAIN_MANIFEST_PATH.is_file():
        run_create_splits()

    report = check_leakage(load_all_manifests())
    if recreate_if_leakage and (
        report["md5_leakage"] or report["augmentation_leakage"] or report["path_leakage"]
    ):
        run_create_splits()
        report = check_leakage(load_all_manifests())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_VERIFICATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset splits")
    parser.parse_args()
    report = run_verify()
    if report["md5_leakage"] or report["augmentation_leakage"] or report["path_leakage"]:
        raise SystemExit("Leakage detected after split creation; manual fix required.")
    print(f"Saved {SPLIT_VERIFICATION_PATH}")


if __name__ == "__main__":
    main()

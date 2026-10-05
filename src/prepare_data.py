"""
Phase 2: Extract datasets, build manifests, duplicates, quality and augmentation analysis.

Does not assign design IDs or train models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLEAN_MANIFEST_PATH,
    DATA_PREPARATION_REPORT_PATH,
    DATA_ROOT,
    DATASET_STATISTICS_PATH,
    DEEPLURE_DIR,
    DUPLICATE_GROUPS_PATH,
    AUGMENTATION_GROUPS_PATH,
    HANDLOOM_ZIP,
    IMAGE_EXTENSIONS,
    IMAGE_QUALITY_REPORT_PATH,
    INDIAN_SAREE_DIR,
    INDIAN_SAREE_ZIP,
    MASTER_DATASET_PATH,
    ORIGINAL_SPLITS,
    PATTERN_FAMILIES,
    PHASE2_SUMMARY_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
)

Image.MAX_IMAGE_PIXELS = None

ROBOFLOW_HASH_PATTERN = re.compile(r"\.rf\.[a-f0-9]{32}", re.IGNORECASE)
COLOR_KEYWORDS = (
    "cyan",
    "golden",
    "pink",
    "red",
    "blue",
    "green",
    "black",
    "white",
    "maroon",
    "orange",
    "yellow",
    "purple",
    "grey",
    "gray",
    "beige",
    "cream",
    "gold",
)


def rel_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def heuristic_color_from_filename(filename: str) -> str | None:
    lower = filename.lower().replace("_", "-")
    found = [c for c in COLOR_KEYWORDS if c in lower]
    return ",".join(sorted(set(found))) if found else None


def extract_archives(force: bool = False) -> dict[str, Any]:
    """Extract zip copies into data/deeplure and data/indian_saree; zips remain untouched."""
    extractions: list[dict[str, Any]] = []

    jobs = [
        {
            "zip_path": HANDLOOM_ZIP,
            "dest_dir": DEEPLURE_DIR,
            "dataset_name": "handloom_deeplure",
        },
        {
            "zip_path": INDIAN_SAREE_ZIP,
            "dest_dir": INDIAN_SAREE_DIR,
            "dataset_name": "indian_saree_patterns",
        },
    ]

    for job in jobs:
        zp: Path = job["zip_path"]
        dest: Path = job["dest_dir"]
        entry: dict[str, Any] = {
            "source_zip": rel_path(zp),
            "destination_dir": rel_path(dest),
            "zip_exists": zp.is_file(),
            "extracted": False,
            "skipped_reason": None,
            "files_in_zip": 0,
            "bytes_extracted": 0,
        }

        if not zp.is_file():
            entry["skipped_reason"] = "zip_not_found"
            extractions.append(entry)
            continue

        with zipfile.ZipFile(zp, "r") as zf:
            entry["files_in_zip"] = len(zf.infolist())
            entry["bytes_extracted"] = sum(i.file_size for i in zf.infolist())

        if dest.exists() and any(dest.iterdir()) and not force:
            entry["skipped_reason"] = "destination_already_populated"
            entry["extracted"] = False
        else:
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(dest)
            entry["extracted"] = True
            entry["skipped_reason"] = None

        extractions.append(entry)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Original ZIP files under data/ were not modified or deleted.",
        "extractions": extractions,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_PREPARATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def iter_image_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (DEEPLURE_DIR, INDIAN_SAREE_DIR):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return sorted(paths)


def parse_indian_metadata(path: Path) -> tuple[str, str, str | None]:
    """Return (dataset_source, original_split, pattern_family)."""
    parts = path.relative_to(INDIAN_SAREE_DIR).parts
    original_split = parts[0] if parts and parts[0] in ORIGINAL_SPLITS else ""
    pattern_family = parts[1] if len(parts) >= 2 and parts[1] in PATTERN_FAMILIES else ""
    if not pattern_family and len(parts) >= 2:
        pattern_family = parts[1]
    return "indian_patterns", original_split, pattern_family or None


def parse_handloom_metadata(path: Path) -> tuple[str, str, str]:
    return "handloom", "", "unknown"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def analyze_image(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "corrupted": False,
        "corruption_reason": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "color_mode": None,
        "extremely_small": False,
        "unreadable": False,
    }
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        row["corrupted"] = True
        row["unreadable"] = True
        row["corruption_reason"] = f"stat_error:{exc}"
        return row

    row["file_size"] = file_size

    try:
        with Image.open(path) as im:
            im.load()
            row["width"] = im.width
            row["height"] = im.height
            row["aspect_ratio"] = round(im.width / im.height, 4) if im.height else None
            row["color_mode"] = im.mode
            if im.width < 16 or im.height < 16:
                row["extremely_small"] = True
    except UnidentifiedImageError:
        row["corrupted"] = True
        row["unreadable"] = True
        row["corruption_reason"] = "unidentified_image_format"
    except OSError as exc:
        row["corrupted"] = True
        row["unreadable"] = True
        row["corruption_reason"] = f"os_error:{exc}"

    return row


def roboflow_source_stem(filename: str) -> tuple[str, str, str]:
    """
    Derive a source identifier from Roboflow export filenames.

    Returns (source_filename_or_identifier, relationship_confidence, note).
    """
    stem = Path(filename).stem
    if ".rf." in stem.lower():
        base = ROBOFLOW_HASH_PATTERN.sub("", stem)
        return base, "medium", "grouped_by_stem_before_.rf._hash_suffix"
    return stem, "low", "no_.rf._pattern_in_filename"


def build_master_records(image_paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in image_paths:
        if is_under(path, INDIAN_SAREE_DIR):
            dataset_source, original_split, pattern_family = parse_indian_metadata(path)
        elif is_under(path, DEEPLURE_DIR):
            dataset_source, original_split, pattern_family = parse_handloom_metadata(path)
        else:
            dataset_source, original_split, pattern_family = "unknown", "", "unknown"

        quality = analyze_image(path)
        digest = None
        if not quality.get("unreadable"):
            try:
                digest = md5_file(path)
            except OSError as exc:
                quality["corrupted"] = True
                quality["unreadable"] = True
                quality["corruption_reason"] = f"md5_read_error:{exc}"

        records.append(
            {
                "image_path": rel_path(path),
                "design_id": None,
                "dataset_source": dataset_source,
                "original_split": original_split,
                "pattern_family": pattern_family,
                "filename": path.name,
                "width": quality.get("width"),
                "height": quality.get("height"),
                "file_size": quality.get("file_size"),
                "md5": digest,
                "heuristic_color": heuristic_color_from_filename(path.name),
                "is_exact_duplicate": False,
                "duplicate_group": None,
                "_quality": quality,
            }
        )
    return records


def assign_duplicate_groups(records: list[dict[str, Any]]) -> None:
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if rec.get("md5"):
            by_md5[rec["md5"]].append(i)

    group_idx = 0
    for md5, indices in sorted(by_md5.items(), key=lambda x: -len(x[1])):
        if len(indices) < 2:
            continue
        group_id = f"dup_{group_idx:04d}"
        group_idx += 1
        for i in indices:
            records[i]["duplicate_group"] = group_id
            records[i]["is_exact_duplicate"] = True


def build_duplicate_groups_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("duplicate_group"):
            rows.append(
                {
                    "duplicate_group": rec["duplicate_group"],
                    "image_path": rec["image_path"],
                    "md5": rec["md5"],
                }
            )
    return pd.DataFrame(rows, columns=["duplicate_group", "image_path", "md5"])


def build_augmentation_groups(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Investigate Roboflow 3x augmentation grouping for Indian patterns only."""
    indian = [r for r in records if r["dataset_source"] == "indian_patterns"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in indian:
        source_id, conf, _note = roboflow_source_stem(rec["filename"])
        key = f"{rec['pattern_family']}|{rec['original_split']}|{source_id}"
        groups[key].append({**rec, "_source_id": source_id, "_conf": conf})

    rows: list[dict[str, Any]] = []
    size_counter: Counter[int] = Counter()
    for key, members in groups.items():
        size_counter[len(members)] += 1
        aug_group = f"aug_{hashlib.md5(key.encode()).hexdigest()[:12]}"
        source_id = members[0]["_source_id"]
        base_conf = members[0]["_conf"]
        if len(members) == 3:
            confidence = "high"
            note = "group_size_equals_3_as_per_roboflow_readme"
        elif len(members) == 2:
            confidence = "medium"
            note = "group_size_2_possible_partial_augment_set_or_duplicate"
        elif len(members) > 3:
            confidence = "low"
            note = f"unexpected_group_size_{len(members)}"
        else:
            confidence = "low"
            note = "singleton_after_stem_grouping"

        for m in members:
            rows.append(
                {
                    "augmentation_group": aug_group,
                    "image_path": m["image_path"],
                    "source_filename_or_identifier": source_id,
                    "relationship_confidence": confidence if base_conf != "low" else "low",
                    "group_size": len(members),
                    "confidence_note": note,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "reliable_augmentation_groups_recovered": False,
            "reason": "no_indian_pattern_images",
        }
    else:
        size_dist = {str(k): int(v) for k, v in sorted(size_counter.items())}
        pct_size_3 = float(size_counter[3]) / max(len(groups), 1)
        summary = {
            "reliable_augmentation_groups_recovered": pct_size_3 > 0.5,
            "grouping_method": "filename_stem_before_.rf._32hex_hash",
            "total_groups": len(groups),
            "group_size_distribution": size_dist,
            "fraction_groups_of_size_3": round(pct_size_3, 4),
            "interpretation": (
                "Augmentation groups are heuristic clusters of export variants sharing "
                "a Roboflow stem; they are NOT verified fine-grained design IDs. "
                "Exact-duplicate MD5 groups may overlap with augmentation groups."
            ),
            "relationship_not_verified_as_design_id": True,
        }
    return df, summary


def build_quality_report(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        q = rec.pop("_quality", {})
        rows.append(
            {
                "image_path": rec["image_path"],
                "dataset_source": rec["dataset_source"],
                "corrupted": q.get("corrupted", False),
                "corruption_reason": q.get("corruption_reason"),
                "width": q.get("width"),
                "height": q.get("height"),
                "aspect_ratio": q.get("aspect_ratio"),
                "color_mode": q.get("color_mode"),
                "extremely_small": q.get("extremely_small", False),
                "unreadable": q.get("unreadable", False),
            }
        )
    return pd.DataFrame(rows)


def usable_record(rec: dict[str, Any], quality_row: pd.Series) -> bool:
    if quality_row["unreadable"] or quality_row["corrupted"]:
        return False
    return True


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
    }


def run_preparation(force_extract: bool = False) -> dict[str, Any]:
    extract_report = extract_archives(force=force_extract)
    image_paths = iter_image_paths()
    if not image_paths:
        raise RuntimeError(
            "No extracted images found under data/deeplure/ or data/indian_saree/. "
            "Check zip extraction in data_preparation_report.json."
        )

    records = build_master_records(image_paths)
    assign_duplicate_groups(records)

    quality_df = build_quality_report(records)
    aug_df, aug_summary = build_augmentation_groups(records)

    master_cols = [
        "image_path",
        "design_id",
        "dataset_source",
        "original_split",
        "pattern_family",
        "filename",
        "width",
        "height",
        "file_size",
        "md5",
        "heuristic_color",
        "is_exact_duplicate",
        "duplicate_group",
    ]
    master_df = pd.DataFrame(records)[master_cols]
    master_df.to_csv(MASTER_DATASET_PATH, index=False)

    dup_df = build_duplicate_groups_csv(records)
    dup_df.to_csv(DUPLICATE_GROUPS_PATH, index=False)

    if not aug_df.empty:
        aug_df[
            [
                "augmentation_group",
                "image_path",
                "source_filename_or_identifier",
                "relationship_confidence",
            ]
        ].to_csv(AUGMENTATION_GROUPS_PATH, index=False)
    else:
        pd.DataFrame(
            columns=[
                "augmentation_group",
                "image_path",
                "source_filename_or_identifier",
                "relationship_confidence",
            ]
        ).to_csv(AUGMENTATION_GROUPS_PATH, index=False)

    quality_df.to_csv(IMAGE_QUALITY_REPORT_PATH, index=False)

    quality_index = quality_df.set_index("image_path")
    clean_rows = []
    for rec in records:
        q = quality_index.loc[rec["image_path"]]
        if not usable_record(rec, q):
            continue
        clean_rows.append(
            {
                "image_path": rec["image_path"],
                "dataset_source": rec["dataset_source"],
                "original_split": rec["original_split"],
                "pattern_family": rec["pattern_family"],
                "width": rec["width"],
                "height": rec["height"],
                "md5": rec["md5"],
                "duplicate_group": rec["duplicate_group"],
                "heuristic_color": rec["heuristic_color"],
            }
        )
    clean_df = pd.DataFrame(clean_rows)
    clean_df.to_csv(CLEAN_MANIFEST_PATH, index=False)

    dup_images = master_df[master_df["is_exact_duplicate"]]
    stats = {
        "total_images": int(len(master_df)),
        "usable_images": int(len(clean_df)),
        "images_by_dataset": master_df["dataset_source"].value_counts().astype(int).to_dict(),
        "images_by_pattern_family": master_df["pattern_family"].value_counts().astype(int).to_dict(),
        "images_by_original_split": (
            master_df[master_df["dataset_source"] == "indian_patterns"]["original_split"]
            .value_counts()
            .astype(int)
            .to_dict()
        ),
        "number_of_duplicate_groups": int(dup_df["duplicate_group"].nunique()) if not dup_df.empty else 0,
        "number_of_duplicate_images": int(len(dup_images)),
        "image_width_statistics": numeric_stats(master_df["width"]),
        "image_height_statistics": numeric_stats(master_df["height"]),
        "aspect_ratio_statistics": numeric_stats(
            master_df["width"] / master_df["height"].replace(0, np.nan)
        ),
        "quality_summary": {
            "corrupted_or_unreadable": int(quality_df["corrupted"].sum()),
            "extremely_small": int(quality_df["extremely_small"].sum()),
            "grayscale_mode_count": int((quality_df["color_mode"] == "L").sum()),
            "rgb_like_count": int(quality_df["color_mode"].isin(["RGB", "RGBA"]).sum()),
        },
        "augmentation_analysis": aug_summary,
        "verified_design_ids_exist": False,
        "ground_truth_color_labels_exist": False,
        "verified_color_invariance_pairs_exist": False,
    }
    with open(DATASET_STATISTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    phase2 = {
        "extracted_image_count": int(len(master_df)),
        "usable_image_count": int(len(clean_df)),
        "duplicate_image_count": int(len(dup_images)),
        "duplicate_group_count": stats["number_of_duplicate_groups"],
        "pattern_family_counts": stats["images_by_pattern_family"],
        "handloom_count": int((master_df["dataset_source"] == "handloom").sum()),
        "design_ids_exist": False,
        "color_labels_exist": False,
        "reliable_augmentation_source_groups": bool(
            aug_summary.get("reliable_augmentation_groups_recovered", False)
        ),
        "augmentation_groups_note": (
            "Stem-based groups align with README (many size-3 sets) for leakage control only; "
            "not verified design IDs or color-invariance pairs."
        ),
        "verified_color_invariance_pairs_exist": False,
        "limitations": [
            "The supplied datasets do not contain verified fine-grained design identifiers "
            "or ground-truth same-design/different-color pairs.",
            "Indian pattern folders (Banarasi, Bandhani, Ikat, Pichwai) are pattern-family "
            "labels only, not design IDs.",
            "Handloom img_* filenames are product/image identifiers, not verified design IDs.",
            "heuristic_color is filename-based and must not be used as ground truth.",
        ],
        "extraction": extract_report,
    }
    with open(PHASE2_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(phase2, f, indent=2)

    return {
        "master_df": master_df,
        "clean_df": clean_df,
        "stats": stats,
        "phase2": phase2,
        "aug_summary": aug_summary,
        "quality_df": quality_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 dataset preparation")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract zips even if destination folders are populated",
    )
    args = parser.parse_args()
    result = run_preparation(force_extract=args.force_extract)
    m = result["master_df"]
    s = result["stats"]
    print(f"Master dataset: {m.shape[0]} rows x {m.shape[1]} columns -> {MASTER_DATASET_PATH}")
    print("images_by_dataset:", s["images_by_dataset"])
    print("pattern_family:", s["images_by_pattern_family"])
    print("duplicate_groups:", s["number_of_duplicate_groups"], "duplicate_images:", s["number_of_duplicate_images"])


if __name__ == "__main__":
    main()

"""
Recursively inspect saree datasets under data/deeplure/ and data/indian_saree/.

If those directories are missing, reports archive contents under data/ without
extracting or modifying original files (read-only zip inspection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from PIL import Image, UnidentifiedImageError

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_ROOT,
    DATASET_SUMMARY_PATH,
    DEEPLURE_DIR,
    IMAGE_EXTENSIONS,
    IMAGE_INVENTORY_PATH,
    INDIAN_SAREE_DIR,
    METADATA_EXTENSIONS,
    PROJECT_ROOT,
    RESULTS_DIR,
)

Image.MAX_IMAGE_PIXELS = None


@dataclass
class ImageRecord:
    image_path: str
    filename: str
    extension: str
    width: int | None
    height: int | None
    folder: str
    possible_label: str | None
    possible_design_id: str | None
    possible_color: str | None
    source_root: str
    storage: str  # "filesystem" | "zip"
    corrupted: bool
    corruption_reason: str | None
    file_size_bytes: int | None
    content_hash: str | None


@dataclass
class InspectionState:
    image_records: list[ImageRecord] = field(default_factory=list)
    metadata_files: list[dict[str, Any]] = field(default_factory=list)
    folder_paths: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _infer_label_from_path(parts: tuple[str, ...]) -> str | None:
    """Infer folder-based labels only when path structure clearly indicates class folders."""
    # Roboflow layout: {split}/{class_name}/file.jpg
    if len(parts) >= 2 and parts[0] in {"train", "valid", "test", "validation"}:
        return parts[1]
    # Single class folder under dataset root
    if len(parts) >= 2 and parts[0] not in {"images", "img", "data"}:
        parent = parts[-2]
        if parent.lower() not in {"deeplure", "indian_saree", "handloom_sarees"}:
            return parent
    return None


def _infer_color_from_filename(filename: str) -> str | None:
    """
    Do not treat arbitrary filename tokens as ground-truth color labels.
    Only record explicit color-like phrases when clearly present in Kaggle-style names.
    """
    lower = filename.lower().replace("_", "-")
    color_keywords = (
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
    found = [c for c in color_keywords if c in lower]
    if found:
        return ",".join(sorted(set(found)))
    return None


def _open_image_from_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as im:
        im.load()
        return im.size


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _inspect_image_bytes(
    *,
    display_path: str,
    filename: str,
    folder: str,
    source_root: str,
    storage: str,
    data: bytes | None,
    path_parts: tuple[str, ...],
) -> ImageRecord:
    ext = Path(filename).suffix.lower()
    possible_label = _infer_label_from_path(path_parts)
    possible_color = _infer_color_from_filename(filename)
    possible_design_id = None  # no verified design ID field in current layouts

    if data is None:
        return ImageRecord(
            image_path=display_path,
            filename=filename,
            extension=ext,
            width=None,
            height=None,
            folder=folder,
            possible_label=possible_label,
            possible_design_id=possible_design_id,
            possible_color=possible_color,
            source_root=source_root,
            storage=storage,
            corrupted=True,
            corruption_reason="missing_or_unreadable_bytes",
            file_size_bytes=None,
            content_hash=None,
        )

    file_size = len(data)
    content_hash = _hash_bytes(data)
    width, height = None, None
    corrupted = False
    reason = None
    try:
        width, height = _open_image_from_bytes(data)
        if width < 16 or height < 16:
            corrupted = True
            reason = f"extremely_small_image_{width}x{height}"
    except UnidentifiedImageError:
        corrupted = True
        reason = "unidentified_image_format"
    except OSError as exc:
        corrupted = True
        reason = f"os_error:{exc}"

    return ImageRecord(
        image_path=display_path,
        filename=filename,
        extension=ext,
        width=width,
        height=height,
        folder=folder,
        possible_label=possible_label,
        possible_design_id=possible_design_id,
        possible_color=possible_color,
        source_root=source_root,
        storage=storage,
        corrupted=corrupted,
        corruption_reason=reason,
        file_size_bytes=file_size,
        content_hash=content_hash,
    )


def iter_filesystem_images(root: Path) -> Iterator[tuple[Path, Path]]:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield root, path


def iter_zip_images(zip_path: Path) -> Iterator[tuple[str, str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            try:
                data = zf.read(info)
            except (OSError, zipfile.BadZipFile) as exc:
                yield name, name, None  # type: ignore[misc]
                continue
            yield name, name, data


def collect_metadata_files(root: Path, state: InspectionState, label: str) -> None:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in METADATA_EXTENSIONS:
            continue
        rel = _rel(path)
        state.metadata_files.append(
            {
                "path": rel,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "dataset_root": label,
            }
        )
        if path.suffix.lower() == ".csv":
            state.notes.append(f"CSV metadata found: {rel}")
        if path.suffix.lower() == ".json":
            state.notes.append(f"JSON metadata found: {rel}")


def collect_zip_metadata(zip_path: Path, state: InspectionState, label: str) -> None:
    rel_zip = _rel(zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in METADATA_EXTENSIONS:
                    continue
                info = zf.getinfo(name)
                state.metadata_files.append(
                    {
                        "path": f"{rel_zip}::{name}",
                        "extension": suffix,
                        "size_bytes": info.file_size,
                        "dataset_root": label,
                        "storage": "zip",
                    }
                )
    except zipfile.BadZipFile:
        state.problems.append(f"Corrupted or invalid zip archive: {rel_zip}")


def inspect_directory(root: Path, dataset_name: str, state: InspectionState) -> None:
    if not root.is_dir():
        state.notes.append(f"Expected directory missing: {_rel(root)}")
        return

    collect_metadata_files(root, state, dataset_name)

    for base, img_path in iter_filesystem_images(root):
        rel_img = _rel(img_path)
        parts = img_path.relative_to(base).parts
        folder = str(Path(*parts[:-1])) if len(parts) > 1 else ""
        state.folder_paths.add(str(base.relative_to(root)) if base != root else ".")
        if parts[:-1]:
            state.folder_paths.add("/".join(parts[:-1]))

        try:
            data = img_path.read_bytes()
        except OSError as exc:
            state.problems.append(f"Could not read image: {rel_img} ({exc})")
            data = None

        record = _inspect_image_bytes(
            display_path=rel_img,
            filename=img_path.name,
            folder=folder,
            source_root=_rel(root),
            storage="filesystem",
            data=data,
            path_parts=parts,
        )
        state.image_records.append(record)


def inspect_zip_as_dataset(zip_path: Path, dataset_name: str, state: InspectionState) -> None:
    if not zip_path.is_file():
        return

    rel_zip = _rel(zip_path)
    state.notes.append(f"Inspecting archive (read-only, not extracted): {rel_zip}")
    collect_zip_metadata(zip_path, state, dataset_name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    parts = Path(info.filename.replace("\\", "/")).parts
                    if parts:
                        state.folder_paths.add("/".join(parts[:-1]) if len(parts) > 1 else ".")
    except zipfile.BadZipFile:
        state.problems.append(f"Corrupted or invalid zip archive: {rel_zip}")
        return

    for internal_name, _, data in iter_zip_images(zip_path):
        filename = Path(internal_name).name
        parts = Path(internal_name).parts
        folder = "/".join(parts[:-1]) if len(parts) > 1 else ""
        display_path = f"{rel_zip}::{internal_name}"

        if data is None:
            state.problems.append(f"Could not read zip entry: {display_path}")
            record = _inspect_image_bytes(
                display_path=display_path,
                filename=filename,
                folder=folder,
                source_root=rel_zip,
                storage="zip",
                data=None,
                path_parts=parts,
            )
            state.image_records.append(record)
            continue

        record = _inspect_image_bytes(
            display_path=display_path,
            filename=filename,
            folder=folder,
            source_root=rel_zip,
            storage="zip",
            data=data,
            path_parts=parts,
        )
        state.image_records.append(record)


def discover_archives(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    return sorted(data_root.glob("*.zip"))


def map_archives_to_datasets(archives: list[Path]) -> dict[str, list[Path]]:
    """Heuristic mapping only for inspection; documented in summary."""
    mapping: dict[str, list[Path]] = {"deeplure": [], "indian_saree": [], "unassigned": []}
    for zp in archives:
        name = zp.name.lower()
        if "handloom" in name or "deeplure" in name:
            mapping["deeplure"].append(zp)
        elif "indian" in name or "saree" in name or "pattern" in name or "fabric" in name:
            mapping["indian_saree"].append(zp)
        else:
            mapping["unassigned"].append(zp)
    return mapping


def build_summary(state: InspectionState) -> dict[str, Any]:
    df = pd.DataFrame([r.__dict__ for r in state.image_records])

    ext_counts: dict[str, int] = {}
    dim_counter: Counter[tuple[int | None, int | None]] = Counter()
    corrupted: list[str] = []
    by_source: dict[str, int] = defaultdict(int)

    if not df.empty:
        ext_counts = df["extension"].value_counts().astype(int).to_dict()
        dim_counter = Counter(zip(df["width"], df["height"]))
        corrupted = df.loc[df["corrupted"], "image_path"].tolist()
        for src, count in df["source_root"].value_counts().items():
            by_source[str(src)] = int(count)

    duplicate_groups: list[dict[str, Any]] = []
    if not df.empty and df["content_hash"].notna().any():
        dup = df[df["content_hash"].notna()].groupby("content_hash")
        for h, group in dup:
            if len(group) > 1:
                duplicate_groups.append(
                    {
                        "content_hash": h,
                        "count": int(len(group)),
                        "paths": group["image_path"].tolist()[:20],
                    }
                )

    labels_present = sorted(
        {r.possible_label for r in state.image_records if r.possible_label}
    )
    colors_inferred = int(sum(1 for r in state.image_records if r.possible_color))
    design_ids = sorted(
        {r.possible_design_id for r in state.image_records if r.possible_design_id}
    )

    csv_files = [m for m in state.metadata_files if m["extension"] == ".csv"]
    json_files = [m for m in state.metadata_files if m["extension"] == ".json"]
    txt_files = [m for m in state.metadata_files if m["extension"] == ".txt"]

    archives = discover_archives(DATA_ROOT)
    archive_map = map_archives_to_datasets(archives)

    return {
        "project_root": _rel(PROJECT_ROOT),
        "inspected_paths": {
            "deeplure_dir": _rel(DEEPLURE_DIR),
            "deeplure_dir_exists": DEEPLURE_DIR.is_dir(),
            "indian_saree_dir": _rel(INDIAN_SAREE_DIR),
            "indian_saree_dir_exists": INDIAN_SAREE_DIR.is_dir(),
        },
        "archives_in_data": [_rel(p) for p in archives],
        "archive_to_dataset_heuristic": {
            k: [_rel(p) for p in v] for k, v in archive_map.items()
        },
        "counts": {
            "total_images": len(state.image_records),
            "total_folders_observed": len(state.folder_paths),
            "corrupted_images": len(corrupted),
            "duplicate_content_hash_groups": len(duplicate_groups),
            "metadata_files": len(state.metadata_files),
            "csv_files": len(csv_files),
            "json_files": len(json_files),
            "txt_files": len(txt_files),
            "images_by_source_root": dict(by_source),
        },
        "image_extensions": ext_counts,
        "dimension_summary": {
            "unique_dimension_pairs": len(dim_counter),
            "most_common_dimensions": [
                {"width": w, "height": h, "count": c}
                for (w, h), c in dim_counter.most_common(15)
            ],
        },
        "labels_and_ids": {
            "verified_design_ids_present": bool(design_ids),
            "design_id_values": design_ids,
            "folder_derived_possible_labels": labels_present,
            "filename_inferred_possible_color_count": colors_inferred,
            "note": (
                "possible_label is inferred from folder names where structure is explicit "
                "(e.g. Roboflow train/valid/test class folders). "
                "possible_design_id is left empty unless a verified ID column/file exists. "
                "possible_color from filenames is heuristic only, not ground truth."
            ),
        },
        "metadata_files": state.metadata_files,
        "sample_filenames": [
            r.filename for r in state.image_records[:15]
        ],
        "problems": state.problems,
        "notes": state.notes,
        "duplicate_image_groups_sample": duplicate_groups[:50],
        "corrupted_image_paths": corrupted,
    }


def run_inspection() -> None:
    state = InspectionState()

    inspect_directory(DEEPLURE_DIR, "deeplure", state)
    inspect_directory(INDIAN_SAREE_DIR, "indian_saree", state)

    archives = discover_archives(DATA_ROOT)
    archive_map = map_archives_to_datasets(archives)

    if not DEEPLURE_DIR.is_dir() and archive_map["deeplure"]:
        for zp in archive_map["deeplure"]:
            inspect_zip_as_dataset(zp, "deeplure (archive)", state)
    if not INDIAN_SAREE_DIR.is_dir() and archive_map["indian_saree"]:
        for zp in archive_map["indian_saree"]:
            inspect_zip_as_dataset(zp, "indian_saree (archive)", state)
    for zp in archive_map["unassigned"]:
        inspect_zip_as_dataset(zp, "unassigned (archive)", state)

    if not state.image_records:
        state.problems.append(
            "No images found under data/deeplure/, data/indian_saree/, or data/*.zip archives."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    inventory_cols = [
        "image_path",
        "filename",
        "extension",
        "width",
        "height",
        "folder",
        "possible_label",
        "possible_design_id",
        "possible_color",
        "source_root",
        "storage",
        "corrupted",
        "corruption_reason",
        "file_size_bytes",
        "content_hash",
    ]
    inv_df = pd.DataFrame([r.__dict__ for r in state.image_records], columns=inventory_cols)
    inv_df.to_csv(IMAGE_INVENTORY_PATH, index=False)

    summary = build_summary(state)
    with open(DATASET_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {DATASET_SUMMARY_PATH}")
    print(f"Wrote {IMAGE_INVENTORY_PATH}")
    print(json.dumps(summary["counts"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saree datasets under data/")
    parser.parse_args()
    run_inspection()


if __name__ == "__main__":
    main()

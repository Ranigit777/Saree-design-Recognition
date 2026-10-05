"""Run Phase 3 pipeline: splits, verification, plots, summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_splits import run_create_splits
from loaders import build_loaders
from phase3_finalize import write_phase3_summary
from verify_splits import run_verify
from visualize_splits import run as visualize_run


def print_final_report(summary: dict, verification: dict) -> None:
    sc = summary["class_counts"]
    print("\nPHASE 3 COMPLETE\n")
    print("Dataset:")
    print(f"* Total: {summary['total_images_project']}")
    print(f"* Train: {summary['train_images']}")
    print(f"* Validation: {summary['validation_images']}")
    print(f"* Test: {summary['test_images']}")
    print(f"* Handloom: {summary['handloom_images']}\n")
    print("Pattern families (supervised splits combined):")
    for fam in ("Banarasi", "Bandhani", "Ikat", "Pichwai"):
        print(f"* {fam}: {sc.get(fam, 0)}")
    print()
    print("Leakage:")
    print(f"* MD5 leakage: {verification['md5_leakage']}")
    print(f"* Augmentation leakage: {verification['augmentation_leakage']}")
    print(f"* Path leakage: {verification['path_leakage']}\n")
    print("Proxy task:")
    print("* Pattern-family metric learning\n")
    print("Not available:")
    print("* Fine-grained design IDs")
    print("* Ground-truth color labels")
    print("* Verified same-design/different-color pairs\n")
    print("Next recommended phase:")
    print("* Model architecture + training baseline")


def main() -> None:
    run_create_splits()
    verification = run_verify(recreate_if_leakage=True)
    if verification["md5_leakage"] or verification["augmentation_leakage"] or verification["path_leakage"]:
        raise SystemExit("Split leakage could not be resolved.")
    visualize_run()
    summary = write_phase3_summary()
    build_loaders(batch_size=8, num_workers=0)
    print_final_report(summary, verification)


if __name__ == "__main__":
    main()

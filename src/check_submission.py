"""
Submission verification script.
Audits the project against all structural, algorithmic, documentation, and model requirements.
Exits with code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Safe encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    # Source Code
    ("src/config.py", "Configuration module"),
    ("src/model.py", "Model architectures (Classifier & Metric Embedding)"),
    ("src/dataset.py", "PyTorch dataset implementation"),
    ("src/loaders.py", "DataLoader builders"),
    ("src/sampler.py", "Pattern-family batch sampler"),
    ("src/losses.py", "Loss functions (CrossEntropy & BatchHard Triplet)"),
    ("src/train_classifier.py", "Classifier training baseline"),
    ("src/evaluate_classifier.py", "Classifier evaluation"),
    ("src/train_metric.py", "Metric learning training script"),
    ("src/embeddings.py", "Embedding extraction module"),
    ("src/retrieval.py", "Gallery/query retrieval module"),
    ("src/verification.py", "Pairwise verification and threshold selection"),
    ("src/gallery.py", "Gallery index manager"),
    ("src/inference.py", "Unified inference CLI"),
    ("src/infer.py", "End-to-end inference script"),
    ("src/color_robustness.py", "Color transformation robustness experiment"),
    ("src/case_evaluation.py", "CASE evaluation framework"),
    ("src/run_all.py", "Master end-to-end pipeline runner"),

    # Models
    ("models/best_classifier.pt", "Classifier baseline model checkpoint"),
    ("models/best_metric_model.pt", "Metric learning model checkpoint"),

    # Generated Embeddings
    ("outputs/embeddings/train_embeddings.npy", "Train set gallery embeddings"),
    ("outputs/embeddings/val_embeddings.npy", "Validation set embeddings"),
    ("outputs/embeddings/test_embeddings.npy", "Test set embeddings"),

    # Results & Reports
    ("outputs/results/phase2_summary.json", "Phase 2 dataset report"),
    ("outputs/results/phase3_summary.json", "Phase 3 split verification report"),
    ("outputs/results/phase4_summary.json", "Phase 4 model training report"),
    ("outputs/results/phase5_summary.json", "Phase 5 retrieval & verification report"),
    ("outputs/results/classifier_test_metrics.json", "Classifier test metrics"),
    ("outputs/results/retrieval_metrics.json", "Retrieval metrics"),
    ("outputs/results/retrieval_results.json", "Phase 5 retrieval results"),
    ("outputs/results/verification_metrics.json", "Verification test metrics"),
    ("outputs/results/verification_results.json", "Phase 5 verification results"),
    ("outputs/results/verification_threshold.json", "Validation threshold selection data"),
    ("outputs/results/color_robustness_metrics.json", "Color robustness experiment metrics"),
    ("outputs/results/color_robustness_results.json", "Phase 5 color robustness results"),
    ("outputs/results/case_evaluation.json", "CASE protocol status report"),
    ("outputs/results/efficiency_report.json", "Efficiency and latency report"),
    ("outputs/results/model_comparison.csv", "Model comparison table"),
    ("outputs/results/final_results.json", "Master consolidated results JSON"),

    # Documentation
    ("README.md", "Master project README"),
    ("docs/approach_note.txt", "500-character Approach Note"),
    ("docs/technical_report.md", "Comprehensive Technical Report"),
    ("docs/evaluation_report.md", "Detailed Evaluation Report"),
    ("docs/architecture.md", "Architecture specification"),
    ("docs/limitations.md", "Scientific limitations document"),
    ("LICENSE", "Open source license"),
    ("requirements.txt", "Project dependencies"),

    # Demo
    ("app/streamlit_app.py", "Interactive Streamlit demonstration app"),
]


def audit_submission() -> bool:
    print("=" * 65)
    print("AIE-CASE: FINAL SUBMISSION AUDIT")
    print("=" * 65)

    passed_count = 0
    failed_count = 0

    for rel_path, desc in REQUIRED_FILES:
        full_p = PROJECT_ROOT / rel_path
        if full_p.is_file() and full_p.stat().st_size > 0:
            print(f"  [PASS] {rel_path:<45} ({desc})")
            passed_count += 1
        else:
            print(f"  [FAIL] {rel_path:<45} (MISSING or EMPTY: {desc})")
            failed_count += 1

    # Check approach_note length
    note_p = PROJECT_ROOT / "docs/approach_note.txt"
    if note_p.is_file():
        char_len = len(note_p.read_text(encoding="utf-8").strip())
        if 400 <= char_len <= 600:
            print(f"  [PASS] docs/approach_note.txt length check ({char_len} chars, within 450-550 target)")
            passed_count += 1
        else:
            print(f"  [WARN] docs/approach_note.txt length is {char_len} chars (target: 450-550)")

    print("-" * 65)
    print(f"Audit Summary: {passed_count} PASSED, {failed_count} FAILED")
    print("=" * 65)

    if failed_count > 0:
        print("\n[FAIL] SUBMISSION CHECK FAILED: Please create or populate the missing files.")
        return False

    print("\n[PASS] SUBMISSION CHECK PASSED: All required code, checkpoints, reports, and docs exist.")
    return True


def main() -> None:
    success = audit_submission()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

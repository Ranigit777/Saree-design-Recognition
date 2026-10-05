"""Project paths and training configuration."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DEEPLURE_DIR = DATA_ROOT / "deeplure"
INDIAN_SAREE_DIR = DATA_ROOT / "indian_saree"

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_ROOT / "results"
EMBEDDINGS_DIR = OUTPUTS_ROOT / "embeddings"
PLOTS_DIR = OUTPUTS_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.json"
IMAGE_INVENTORY_PATH = RESULTS_DIR / "image_inventory.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

METADATA_EXTENSIONS = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml", ".tsv"}

HANDLOOM_ZIP = DATA_ROOT / "handloom_sarees-20261005T100532Z-1-001.zip"
INDIAN_SAREE_ZIP = DATA_ROOT / "Indian saree Patterns.zip"

MASTER_DATASET_PATH = RESULTS_DIR / "master_dataset.csv"
DUPLICATE_GROUPS_PATH = RESULTS_DIR / "duplicate_groups.csv"
AUGMENTATION_GROUPS_PATH = RESULTS_DIR / "augmentation_groups.csv"
IMAGE_QUALITY_REPORT_PATH = RESULTS_DIR / "image_quality_report.csv"
CLEAN_MANIFEST_PATH = RESULTS_DIR / "clean_manifest.csv"
DATASET_STATISTICS_PATH = RESULTS_DIR / "dataset_statistics.json"
DATA_PREPARATION_REPORT_PATH = RESULTS_DIR / "data_preparation_report.json"
PHASE2_SUMMARY_PATH = RESULTS_DIR / "phase2_summary.json"
DATASET_OVERVIEW_PLOT = PLOTS_DIR / "dataset_overview.png"
DIMENSION_PLOT = PLOTS_DIR / "image_dimensions.png"

PATTERN_FAMILIES = ("Banarasi", "Bandhani", "Ikat", "Pichwai")
ORIGINAL_SPLITS = ("train", "valid", "test")

SPLIT_RANDOM_SEED = 42
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}

TRAIN_MANIFEST_PATH = RESULTS_DIR / "train_manifest.csv"
VAL_MANIFEST_PATH = RESULTS_DIR / "val_manifest.csv"
TEST_MANIFEST_PATH = RESULTS_DIR / "test_manifest.csv"
HANDLOOM_UNLABELED_PATH = RESULTS_DIR / "handloom_unlabeled.csv"
SPLIT_VERIFICATION_PATH = RESULTS_DIR / "split_verification.json"
PHASE3_SUMMARY_PATH = RESULTS_DIR / "phase3_summary.json"
PROXY_TASK_DEFINITION_PATH = RESULTS_DIR / "proxy_task_definition.md"
SPLIT_DISTRIBUTION_PLOT = PLOTS_DIR / "split_distribution.png"
SAMPLE_GRID_PLOT = PLOTS_DIR / "sample_grid.png"

IMAGE_SIZE = 224
NUM_CLASSES = 4

PATTERN_FAMILY_TO_LABEL = {
    "Banarasi": 0,
    "Bandhani": 1,
    "Ikat": 2,
    "Pichwai": 3,
}
LABEL_TO_PATTERN_FAMILY = {v: k for k, v in PATTERN_FAMILY_TO_LABEL.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Phase 4 outputs
BEST_CLASSIFIER_PATH = MODELS_DIR / "best_classifier.pt"
BEST_METRIC_MODEL_PATH = MODELS_DIR / "best_metric_model.pt"
CLASSIFIER_HISTORY_PATH = RESULTS_DIR / "classifier_history.csv"
CLASSIFIER_TEST_METRICS_PATH = RESULTS_DIR / "classifier_test_metrics.json"
CLASSIFIER_REPORT_PATH = RESULTS_DIR / "classifier_classification_report.csv"
CLASSIFIER_CONFUSION_PLOT = PLOTS_DIR / "classifier_confusion_matrix.png"
CLASSIFIER_LOSS_PLOT = PLOTS_DIR / "classifier_loss_curve.png"
CLASSIFIER_ACCURACY_PLOT = PLOTS_DIR / "classifier_accuracy_curve.png"
RETRIEVAL_METRICS_PATH = RESULTS_DIR / "retrieval_metrics.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
EMBEDDING_STATISTICS_PATH = RESULTS_DIR / "embedding_statistics.json"
EFFICIENCY_REPORT_PATH = RESULTS_DIR / "efficiency_report.json"
PHASE4_SUMMARY_PATH = RESULTS_DIR / "phase4_summary.json"
TEST_EMBEDDINGS_PCA_PLOT = PLOTS_DIR / "test_embeddings_pca.png"
TEST_EMBEDDINGS_TSNE_PLOT = PLOTS_DIR / "test_embeddings_tsne.png"

# Phase 5 outputs
RETRIEVAL_RESULTS_PATH = RESULTS_DIR / "retrieval_results.json"
VERIFICATION_THRESHOLD_PATH = RESULTS_DIR / "verification_threshold.json"
VERIFICATION_METRICS_PATH = RESULTS_DIR / "verification_metrics.json"
VERIFICATION_RESULTS_PATH = RESULTS_DIR / "verification_results.json"
VERIFICATION_ROC_PLOT = PLOTS_DIR / "verification_roc_curve.png"
VERIFICATION_THRESHOLD_PLOT = PLOTS_DIR / "verification_threshold_selection.png"
COLOR_ROBUSTNESS_METRICS_PATH = RESULTS_DIR / "color_robustness_metrics.json"
COLOR_ROBUSTNESS_RESULTS_PATH = RESULTS_DIR / "color_robustness_results.json"
COLOR_ROBUSTNESS_PLOT = PLOTS_DIR / "color_robustness_analysis.png"
COLOR_ROBUSTNESS_GRID_PLOT = PLOTS_DIR / "color_robustness_transforms_grid.png"
CASE_EVALUATION_PATH = RESULTS_DIR / "case_evaluation.json"
CASE_EVALUATION_PROTOCOL_PATH = RESULTS_DIR / "case_evaluation_protocol.md"
CASE_MOCK_METRICS_PATH = RESULTS_DIR / "case_mock_metrics.json"
PHASE5_SUMMARY_PATH = RESULTS_DIR / "phase5_summary.json"


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 224
    embedding_dim: int = 256
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    triplet_margin: float = 0.2
    triplet_p: int = 2
    num_workers: int = 0
    patience: int = 5
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.2
    run_color_robustness_experiment: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


TRAIN_CONFIG = TrainConfig()
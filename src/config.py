from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DRONEFACE_ROOT = DATA_DIR / "DroneFace" / "open_data_set"

FIGS_DIR = PROJECT_ROOT / "figs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INDEX_DIR = OUTPUTS_DIR / "indexes"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
RESULTS_DIR = OUTPUTS_DIR / "results"
TABLES_DIR = OUTPUTS_DIR / "tables"
CROPS_DIR = OUTPUTS_DIR / "crops"

PROJECT_DIRS = [
    FIGS_DIR / "dataset",
    FIGS_DIR / "detection",
    FIGS_DIR / "recognition",
    FIGS_DIR / "metrics",
    INDEX_DIR,
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    CROPS_DIR,
    CROPS_DIR / "full_images",
]


def ensure_project_dirs() -> None:
    """Create output folders used by the DroneFace baseline preparation."""
    for directory in PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

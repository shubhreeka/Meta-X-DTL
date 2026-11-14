"""Shared paths and settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folder that contains original_data/ and augmented_data/ (see README > Dataset).
DATA_ROOT = Path(os.environ.get("ORAL_DATA_ROOT", ROOT / "data" / "Oral Images Dataset"))

ARTIFACTS = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for p in (ARTIFACTS, MODELS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
SEED = 42
CLASSES = ["benign", "malignant"]  # label 0, 1

SOURCES = {
    "benign": ["original_data/benign_lesions", "augmented_data/augmented_benign"],
    "malignant": ["original_data/malignant_lesions", "augmented_data/augmented_malignant"],
}

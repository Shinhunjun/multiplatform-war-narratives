from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

MODULE_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = MODULE_ROOT / "preprocessing"
EDA_DIR = MODULE_ROOT / "eda"
DATA_COLLECTION_DIR = MODULE_ROOT / "data_collection"
WEEKLY_UPDATE_DIR = MODULE_ROOT / "weekly_update"

for path in (MODULE_ROOT, PREPROCESSING_DIR, EDA_DIR, DATA_COLLECTION_DIR, WEEKLY_UPDATE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

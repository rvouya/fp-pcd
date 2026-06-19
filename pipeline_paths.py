"""Shared dataset paths and batch I/O helpers for the run_local.py drivers.

filtering_spatial/run_local.py and filtering_frequency/run_local.py both need
the same "resolve CSV image list" and "resize+normalize into output/01" steps.
This module is the single source of truth for those so the two drivers (and
enhancement_roi/run_local.py) stay in sync.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
from tqdm import tqdm

from preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CORRUPTED_DIR = DATA_DIR / "corrupted"
GROUNDTRUTH_DIR = DATA_DIR / "groundtruth"
LABELS_CSV = DATA_DIR / "balanced_2500.csv"

OUTPUT_ROOT = PROJECT_ROOT / "output"
NORMALIZED_DIR = OUTPUT_ROOT / "01_preprocessing" / "normalized"
GROUNDTRUTH_NORMALIZED_DIR = OUTPUT_ROOT / "01_preprocessing" / "groundtruth_normalized"


def list_image_names(limit: Optional[int] = None) -> List[str]:
    """Return image filenames from the labels CSV, in CSV order."""
    df = pd.read_csv(LABELS_CSV)
    names = df["Image Index"].tolist()
    return names[:limit] if limit else names


def list_image_paths(image_dir: Path = CORRUPTED_DIR, limit: Optional[int] = None) -> List[Path]:
    """Resolve image paths under image_dir for names present in the labels CSV."""
    names = list_image_names()
    paths = [image_dir / n for n in names if (image_dir / n).exists()]
    missing = len(names) - len(paths)
    if missing:
        logger.warning("%d images listed in CSV not found in %s", missing, image_dir)
    return paths[:limit] if limit else paths


def ensure_normalized(
    paths: List[Path],
    dest_dir: Path,
    size: int = 224,
    norm: str = "minmax",
    force: bool = False,
    desc: str = "preprocess",
) -> None:
    """Resize+normalize images into dest_dir, skipping ones already processed."""
    pre = Preprocessor()
    dest_dir.mkdir(parents=True, exist_ok=True)

    todo = paths if force else [p for p in paths if not (dest_dir / p.name).exists()]
    if not todo:
        print(f"[{desc}] {len(paths)} images already processed -> skip")
        return

    failed = 0
    for src in tqdm(todo, desc=f"{desc} {size}px/{norm}", unit="img"):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed += 1
            continue
        out = pre.process(img, target_size=(size, size), norm_method=norm)
        pre.save(out, dest_dir / src.name)
    print(f"[{desc}] {len(todo) - failed}/{len(todo)} saved -> {dest_dir}, failed={failed}")

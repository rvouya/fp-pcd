"""Dataset loader for chest X-Ray images (original and corrupted splits)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

ORIGINAL_IMG_DIR = DATASET_ROOT / "original" / "fp" / "balanced"
ORIGINAL_LABELS_CSV = DATASET_ROOT / "original" / "balanced_prompts_fixed.csv"

CORRUPTED_BALANCED_IMG_DIR = (
    DATASET_ROOT / "corrupted" / "combined" / "combined" / "balanced" / "balanced"
)
CORRUPTED_BALANCED_LABELS_CSV = DATASET_ROOT / "corrupted" / "balanced_2500.csv"

CORRUPTED_IMBALANCED_IMG_DIR = (
    DATASET_ROOT / "corrupted" / "combined" / "combined" / "Imbalanced" / "imbalanced"
)
CORRUPTED_IMBALANCED_LABELS_CSV = DATASET_ROOT / "corrupted" / "imbalanced_2500.csv"


class DatasetLoader:
    """Load chest X-Ray dataset images and their labels.

    Supports original and corrupted (balanced/imbalanced) splits.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path = base_path or DATASET_ROOT
        logger.info("DatasetLoader initialized with base: %s", self.base_path)

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------

    def _load_labels_original(self) -> pd.DataFrame:
        """Return DataFrame with columns [image_index, label]."""
        df = pd.read_csv(ORIGINAL_LABELS_CSV)
        df = df.rename(
            columns={"Image Index": "image_index", "Finding Labels": "label"}
        )
        return df[["image_index", "label"]].copy()

    def _load_labels_corrupted(self, split: str = "balanced") -> pd.DataFrame:
        """Return DataFrame with columns [image_index, label] for corrupted split."""
        if split == "balanced":
            csv_path = CORRUPTED_BALANCED_LABELS_CSV
        else:
            csv_path = CORRUPTED_IMBALANCED_LABELS_CSV

        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={"Image Index": "image_index", "Finding Labels": "label"}
        )
        return df[["image_index", "label"]].copy()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """Load a single grayscale X-Ray image as uint8 numpy array.

        Args:
            image_path: Absolute path to image file.

        Returns:
            Grayscale image array of shape (H, W).

        Raises:
            FileNotFoundError: If image does not exist.
            IOError: If OpenCV cannot decode the image.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"OpenCV failed to read: {image_path}")
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_original(self) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Load all available original images with labels.

        Returns:
            Tuple of (images, labels, image_paths_str).
        """
        labels_df = self._load_labels_original()
        images, labels, paths = [], [], []

        for _, row in labels_df.iterrows():
            img_path = ORIGINAL_IMG_DIR / row["image_index"]
            if not img_path.exists():
                logger.warning("Missing original image: %s", img_path)
                continue
            try:
                img = self.load_image(img_path)
                images.append(img)
                labels.append(row["label"])
                paths.append(str(img_path))
            except (IOError, FileNotFoundError) as exc:
                logger.error("Skipping %s: %s", img_path, exc)

        logger.info("Loaded %d original images", len(images))
        return images, labels, paths

    def load_corrupted(
        self, split: str = "balanced"
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Load corrupted images for the given split.

        Args:
            split: "balanced" or "imbalanced".

        Returns:
            Tuple of (images, labels, image_paths_str).
        """
        labels_df = self._load_labels_corrupted(split)
        img_dir = (
            CORRUPTED_BALANCED_IMG_DIR
            if split == "balanced"
            else CORRUPTED_IMBALANCED_IMG_DIR
        )
        images, labels, paths = [], [], []

        for _, row in labels_df.iterrows():
            img_path = img_dir / row["image_index"]
            if not img_path.exists():
                logger.warning("Missing corrupted image: %s", img_path)
                continue
            try:
                img = self.load_image(img_path)
                images.append(img)
                labels.append(row["label"])
                paths.append(str(img_path))
            except (IOError, FileNotFoundError) as exc:
                logger.error("Skipping %s: %s", img_path, exc)

        logger.info("Loaded %d corrupted/%s images", len(images), split)
        return images, labels, paths

    def get_label_dataframe(
        self, dataset: str = "original", split: str = "balanced"
    ) -> pd.DataFrame:
        """Return full labels DataFrame for a dataset version.

        Args:
            dataset: "original" or "corrupted".
            split: "balanced" or "imbalanced" (only for corrupted).
        """
        if dataset == "original":
            return self._load_labels_original()
        return self._load_labels_corrupted(split)

    def get_unique_labels(
        self, dataset: str = "original", split: str = "balanced"
    ) -> List[str]:
        """Return sorted unique class labels."""
        df = self.get_label_dataframe(dataset, split)
        return sorted(df["label"].unique().tolist())

    def get_image_paths(
        self, dataset: str = "original", split: str = "balanced"
    ) -> Dict[str, Path]:
        """Return {image_index: absolute_path} mapping."""
        df = self.get_label_dataframe(dataset, split)
        if dataset == "original":
            img_dir = ORIGINAL_IMG_DIR
        elif split == "balanced":
            img_dir = CORRUPTED_BALANCED_IMG_DIR
        else:
            img_dir = CORRUPTED_IMBALANCED_IMG_DIR

        return {
            row["image_index"]: img_dir / row["image_index"]
            for _, row in df.iterrows()
        }

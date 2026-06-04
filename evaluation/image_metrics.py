"""Image quality metrics: PSNR and SSIM."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

logger = logging.getLogger(__name__)


def compute_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Reference image (uint8 or float).
        processed: Degraded/processed image.
        data_range: Value range. If None, inferred from dtype.

    Returns:
        PSNR in dB. Returns inf if images are identical.
    """
    if data_range is None:
        data_range = 255.0 if original.dtype == np.uint8 else 1.0
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, (original.shape[1], original.shape[0])
        )
    return float(skimage_psnr(original, processed, data_range=data_range))


def compute_ssim(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute Structural Similarity Index between two images.

    Args:
        original: Reference image.
        processed: Comparison image.
        data_range: Value range. If None, inferred from dtype.

    Returns:
        SSIM in [-1, 1]. Higher is better.
    """
    if data_range is None:
        data_range = 255.0 if original.dtype == np.uint8 else 1.0
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, (original.shape[1], original.shape[0])
        )
    return float(skimage_ssim(original, processed, data_range=data_range))


class ImageMetricsEvaluator:
    """Batch image quality evaluation across multiple processing stages."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir

    def compare_pair(
        self,
        original_path: Path,
        processed_path: Path,
        label: str = "",
    ) -> Dict[str, object]:
        """Compute PSNR and SSIM for a single image pair.

        Returns:
            Dict with keys: image_name, stage, psnr, ssim.
        """
        orig = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        proc = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)

        if orig is None or proc is None:
            logger.error("Cannot read pair: %s / %s", original_path, processed_path)
            return {
                "image_name": original_path.name,
                "stage": label,
                "psnr": float("nan"),
                "ssim": float("nan"),
            }

        return {
            "image_name": original_path.name,
            "stage": label,
            "psnr": compute_psnr(orig, proc),
            "ssim": compute_ssim(orig, proc),
        }

    def compare_batch(
        self,
        original_paths: List[Path],
        processed_paths: List[Path],
        stage_label: str = "processed",
    ) -> pd.DataFrame:
        """Compute metrics for aligned lists of original and processed images.

        Args:
            original_paths: Reference image paths (same order as processed).
            processed_paths: Processed image paths.
            stage_label: Column value identifying the processing stage.

        Returns:
            DataFrame with columns [image_name, stage, psnr, ssim].
        """
        assert len(original_paths) == len(processed_paths), \
            "original and processed lists must have equal length"

        rows = [
            self.compare_pair(o, p, stage_label)
            for o, p in zip(original_paths, processed_paths)
        ]
        df = pd.DataFrame(rows)
        logger.info(
            "Image metrics [%s]: mean PSNR=%.2f, mean SSIM=%.4f",
            stage_label,
            df["psnr"].mean(),
            df["ssim"].mean(),
        )
        return df

    def compare_stages(
        self,
        original_dir: Path,
        stage_dirs: Dict[str, Path],
    ) -> pd.DataFrame:
        """Compare original images against multiple processing stages.

        Args:
            original_dir: Directory of reference images.
            stage_dirs: Mapping of stage name → processed image directory.

        Returns:
            Combined DataFrame with all stages.
        """
        orig_files = sorted(original_dir.glob("*.png"))
        all_rows: List[pd.DataFrame] = []

        for stage_name, stage_dir in stage_dirs.items():
            stage_rows = []
            for orig_path in orig_files:
                proc_path = stage_dir / orig_path.name
                if proc_path.exists():
                    stage_rows.append(self.compare_pair(orig_path, proc_path, stage_name))
                else:
                    logger.warning("No processed image for %s in %s", orig_path.name, stage_dir)

            if stage_rows:
                all_rows.append(pd.DataFrame(stage_rows))

        if not all_rows:
            return pd.DataFrame(columns=["image_name", "stage", "psnr", "ssim"])
        return pd.concat(all_rows, ignore_index=True)

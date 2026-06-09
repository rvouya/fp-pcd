"""High-level spatial filtering pipeline: Gaussian LPF + Unsharp Masking."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .gaussian_lpf import GaussianLPF
from .unsharp_masking import UnsharpMasking

logger = logging.getLogger(__name__)


class SpatialFilter:
    """Batch spatial filtering pipeline.

    Wraps GaussianLPF and UnsharpMasking for processing image collections.
    """

    def __init__(
        self,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
        unsharp_radius: float = 1.0,
        unsharp_amount: float = 1.0,
        unsharp_threshold: int = 0,
    ) -> None:
        self.gaussian = GaussianLPF(gaussian_kernel_size, gaussian_sigma)
        self.unsharp = UnsharpMasking(unsharp_radius, unsharp_amount, unsharp_threshold)

    # ------------------------------------------------------------------
    # Single-image methods
    # ------------------------------------------------------------------

    def apply_gaussian(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Gaussian LPF with optional parameter override."""
        k = kernel_size or self.gaussian.kernel_size
        s = sigma or self.gaussian.sigma
        return self.gaussian.apply_with_params(image, k, s)

    def apply_unsharp(
        self,
        image: np.ndarray,
        radius: Optional[float] = None,
        amount: Optional[float] = None,
        threshold: Optional[int] = None,
    ) -> np.ndarray:
        """Apply unsharp masking with optional parameter override."""
        r = radius or self.unsharp.radius
        a = amount or self.unsharp.amount
        t = threshold if threshold is not None else self.unsharp.threshold
        return self.unsharp.apply_with_params(image, r, a, t)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    @staticmethod
    def _save(image: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if image.dtype in (np.float32, np.float64):
            save_img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            save_img = image
        cv2.imwrite(str(path), save_img)

    def batch_gaussian(
        self,
        image_paths: List[Path],
        output_dir: Path,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> List[Path]:
        """Apply Gaussian LPF to all images and save results."""
        out_dir = output_dir / "gaussian"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for src in image_paths:
            try:
                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError(f"Cannot read {src}")
                filtered = self.apply_gaussian(img, kernel_size, sigma)
                out_path = out_dir / src.name
                self._save(filtered, out_path)
                saved.append(out_path)
            except Exception as exc:
                logger.error("Gaussian batch error on %s: %s", src, exc)

        logger.info("Gaussian batch: %d images saved to %s", len(saved), out_dir)
        return saved

    def batch_unsharp(
        self,
        image_paths: List[Path],
        output_dir: Path,
        radius: float = 1.0,
        amount: float = 1.0,
        threshold: int = 0,
    ) -> List[Path]:
        """Apply unsharp masking to all images and save results."""
        out_dir = output_dir / "unsharp"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for src in image_paths:
            try:
                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError(f"Cannot read {src}")
                sharpened = self.apply_unsharp(img, radius, amount, threshold)
                out_path = out_dir / src.name
                self._save(sharpened, out_path)
                saved.append(out_path)
            except Exception as exc:
                logger.error("Unsharp batch error on %s: %s", src, exc)

        logger.info("Unsharp batch: %d images saved to %s", len(saved), out_dir)
        return saved

    def compare_both(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return original, gaussian, and unsharp versions for comparison."""
        return {
            "original": image,
            "gaussian_lpf": self.apply_gaussian(image),
            "unsharp_masking": self.apply_unsharp(image),
        }

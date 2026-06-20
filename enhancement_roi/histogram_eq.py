"""Global Histogram Equalization enhancement."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HistogramEqualizer:
    """Apply global histogram equalization to grayscale X-Ray images.

    Unlike CLAHE (which equalizes locally per tile), this spreads the global
    intensity histogram across the full dynamic range. It is a classic,
    parameter-free contrast enhancement baseline for X-ray imaging.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply global histogram equalization to a single grayscale image.

        Args:
            image: uint8 grayscale image.

        Returns:
            Equalized uint8 image.
        """
        return self.apply_with_params(image)

    @staticmethod
    def apply_with_params(image: np.ndarray) -> np.ndarray:
        """Apply global histogram equalization with explicit (no) parameters.

        Args:
            image: uint8 grayscale image.

        Returns:
            Equalized uint8 image.
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.equalizeHist(image)

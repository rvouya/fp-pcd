"""Morphological operations: Top-Hat, Opening, Closing."""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MorphologicalOps:
    """Morphological image processing operations.

    All methods are static/class methods — no instance state required.
    """

    @staticmethod
    def _build_kernel(kernel_size: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
        """Build a structuring element.

        Args:
            kernel_size: Side length of the kernel.
            shape: cv2.MORPH_ELLIPSE, MORPH_RECT, or MORPH_CROSS.
        """
        ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.getStructuringElement(shape, (ksize, ksize))

    @classmethod
    def top_hat(
        cls,
        image: np.ndarray,
        kernel_size: int = 15,
    ) -> np.ndarray:
        """Apply white top-hat transform.

        Extracts small bright features from the background.

        Args:
            image: uint8 grayscale image.
            kernel_size: Structuring element size.

        Returns:
            Top-hat result (uint8).
        """
        kernel = cls._build_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    @classmethod
    def opening(
        cls,
        image: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Apply morphological opening (erosion then dilation).

        Removes small bright objects (noise removal).

        Args:
            image: uint8 grayscale image.
            kernel_size: Structuring element size.

        Returns:
            Opened image (uint8).
        """
        kernel = cls._build_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @classmethod
    def closing(
        cls,
        image: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Apply morphological closing (dilation then erosion).

        Fills small dark holes within bright regions.

        Args:
            image: uint8 grayscale image.
            kernel_size: Structuring element size.

        Returns:
            Closed image (uint8).
        """
        kernel = cls._build_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @classmethod
    def black_hat(
        cls,
        image: np.ndarray,
        kernel_size: int = 15,
    ) -> np.ndarray:
        """Apply black top-hat transform.

        Extracts small dark features from the background.
        """
        kernel = cls._build_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    @classmethod
    def experiment_top_hat(
        cls,
        image: np.ndarray,
        kernel_sizes: list = (5, 10, 15, 25, 35),
    ) -> dict:
        """Run top-hat with different kernel sizes."""
        return {f"tophat_k{k}": cls.top_hat(image, k) for k in kernel_sizes}

    @classmethod
    def experiment_opening(
        cls,
        image: np.ndarray,
        kernel_sizes: list = (3, 5, 7, 11),
    ) -> dict:
        """Run opening with different kernel sizes."""
        return {f"opening_k{k}": cls.opening(image, k) for k in kernel_sizes}

    @classmethod
    def experiment_closing(
        cls,
        image: np.ndarray,
        kernel_sizes: list = (3, 5, 7, 11),
    ) -> dict:
        """Run closing with different kernel sizes."""
        return {f"closing_k{k}": cls.closing(image, k) for k in kernel_sizes}

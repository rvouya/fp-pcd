"""Gaussian Low Pass Filter implementation."""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GaussianLPF:
    """Apply Gaussian low-pass filter to grayscale images.

    Experiments with kernel size and sigma are supported.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        """Initialize with default kernel parameters.

        Args:
            kernel_size: Must be odd and positive.
            sigma: Standard deviation of the Gaussian kernel.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to a single image.

        Args:
            image: Grayscale uint8 or float image.

        Returns:
            Blurred image with same dtype as input.
        """
        ksize = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        return cv2.GaussianBlur(image, (ksize, ksize), self.sigma)

    def apply_with_params(
        self, image: np.ndarray, kernel_size: int, sigma: float
    ) -> np.ndarray:
        """Apply Gaussian blur with explicit parameters."""
        ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def experiment(
        self,
        image: np.ndarray,
        kernel_sizes: List[int] = (3, 5, 7, 11, 15),
        sigmas: List[float] = (0.5, 1.0, 2.0, 3.0),
    ) -> Dict[str, np.ndarray]:
        """Run parameter sweep returning all filtered variants.

        Args:
            image: Input grayscale image.
            kernel_sizes: List of kernel sizes to test.
            sigmas: List of sigma values to test.

        Returns:
            Dict keyed by "k{kernel_size}_s{sigma}" with filtered arrays.
        """
        results: Dict[str, np.ndarray] = {}
        for k in kernel_sizes:
            for s in sigmas:
                key = f"k{k}_s{s}"
                results[key] = self.apply_with_params(image, k, s)
        logger.debug(
            "Gaussian experiment: %d variants generated", len(results)
        )
        return results

    def get_kernel(self) -> np.ndarray:
        """Return the Gaussian kernel matrix for visualization."""
        ksize = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        kernel = cv2.getGaussianKernel(ksize, self.sigma)
        return kernel @ kernel.T

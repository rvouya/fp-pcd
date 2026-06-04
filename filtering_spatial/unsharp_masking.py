"""Unsharp masking sharpening filter."""

import logging
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class UnsharpMasking:
    """Unsharp masking for edge enhancement.

    Formula: output = image + amount * (image - blurred) if diff > threshold
    """

    def __init__(
        self,
        radius: float = 1.0,
        amount: float = 1.0,
        threshold: int = 0,
    ) -> None:
        """Initialize unsharp mask parameters.

        Args:
            radius: Blur radius (controls kernel size: ksize = 2*ceil(radius)+1).
            amount: Sharpening strength multiplier.
            threshold: Minimum pixel difference to apply sharpening.
        """
        self.radius = radius
        self.amount = amount
        self.threshold = threshold

    def _radius_to_ksize(self, radius: float) -> int:
        import math
        ksize = 2 * math.ceil(radius) + 1
        return max(3, ksize)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to image.

        Args:
            image: Grayscale uint8 image.

        Returns:
            Sharpened uint8 image.
        """
        return self.apply_with_params(image, self.radius, self.amount, self.threshold)

    def apply_with_params(
        self,
        image: np.ndarray,
        radius: float,
        amount: float,
        threshold: int,
    ) -> np.ndarray:
        """Apply with explicit parameters.

        Args:
            image: Grayscale uint8 image.
            radius: Blur radius.
            amount: Sharpening strength.
            threshold: Min absolute difference threshold.

        Returns:
            Sharpened uint8 image.
        """
        ksize = self._radius_to_ksize(radius)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)

        img_float = image.astype(np.float64)
        blur_float = blurred.astype(np.float64)
        diff = img_float - blur_float

        if threshold > 0:
            mask = np.abs(diff) >= threshold
            sharpened = np.where(mask, img_float + amount * diff, img_float)
        else:
            sharpened = img_float + amount * diff

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def experiment(
        self,
        image: np.ndarray,
        radii: List[float] = (0.5, 1.0, 2.0, 3.0),
        amounts: List[float] = (0.5, 1.0, 1.5, 2.0),
        thresholds: List[int] = (0, 5, 10),
    ) -> Dict[str, np.ndarray]:
        """Run parameter sweep.

        Returns:
            Dict keyed by "r{radius}_a{amount}_t{threshold}".
        """
        results: Dict[str, np.ndarray] = {}
        for r in radii:
            for a in amounts:
                for t in thresholds:
                    key = f"r{r}_a{a}_t{t}"
                    results[key] = self.apply_with_params(image, r, a, t)
        logger.debug("Unsharp experiment: %d variants", len(results))
        return results

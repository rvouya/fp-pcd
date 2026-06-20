"""Adaptive Gamma Correction enhancement."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GammaCorrector:
    """Apply gamma correction to grayscale X-Ray images.

    Output pixel = 255 * (input / 255) ** gamma.
        - gamma < 1 brightens (lifts dark soft-tissue/lung regions),
        - gamma > 1 darkens (suppresses bright bone),
        - gamma == 1 is identity.

    The default is an *adaptive* gamma chosen from the image's mean brightness so
    dark images are brightened and bright images are darkened toward a mid-gray
    target, without manual tuning. A fixed gamma can override the heuristic.
    """

    def __init__(self, gamma: float = 0.0) -> None:
        """Initialize the corrector.

        Args:
            gamma: Fixed gamma to use. If <= 0 (default), gamma is chosen
                adaptively per image from its mean brightness.
        """
        self.gamma = gamma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to a single grayscale image.

        Uses the configured fixed gamma, or the adaptive heuristic when the
        configured gamma is <= 0.

        Args:
            image: uint8 grayscale image.

        Returns:
            Gamma-corrected uint8 image.
        """
        gamma = self.gamma if self.gamma > 0 else self.auto_gamma(image)
        return self.apply_with_params(image, gamma)

    @staticmethod
    def auto_gamma(image: np.ndarray, target: float = 0.5) -> float:
        """Pick a gamma that nudges mean brightness toward a target.

        Solves ``mean_norm ** gamma == target`` for gamma, i.e.
        ``gamma = log(target) / log(mean_norm)``, where ``mean_norm`` is the
        image mean scaled to [0, 1]. Dark images (mean_norm < target) get
        gamma < 1 (brighten); bright images get gamma > 1 (darken). The result
        is clamped to a safe [0.4, 2.5] range to avoid extreme corrections.

        Args:
            image: uint8 grayscale image.
            target: Desired normalized mean brightness in (0, 1).

        Returns:
            A gamma value in [0.4, 2.5].
        """
        mean_norm = float(np.mean(image)) / 255.0
        mean_norm = min(max(mean_norm, 1e-3), 1.0 - 1e-3)
        gamma = np.log(target) / np.log(mean_norm)
        return float(np.clip(gamma, 0.4, 2.5))

    @staticmethod
    def apply_with_params(image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction with an explicit gamma via a LUT.

        Args:
            image: uint8 grayscale image.
            gamma: Gamma exponent (> 0).

        Returns:
            Gamma-corrected uint8 image.
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        gamma = max(gamma, 1e-6)
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.float32
        )
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[image]

"""Butterworth Low-Pass and High-Pass filter masks in the frequency domain."""

import logging
from typing import Literal

import numpy as np

from .fft_transform import FFTTransform

logger = logging.getLogger(__name__)

FilterType = Literal["lpf", "hpf"]


class ButterworthFilter:
    """Create and apply Butterworth frequency-domain filter masks.

    The filter is applied as element-wise multiplication in the frequency domain.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        cutoff: float = 30.0,
        order: int = 2,
        filter_type: FilterType = "lpf",
    ) -> None:
        """Initialize Butterworth filter.

        Args:
            rows: Image height.
            cols: Image width.
            cutoff: Cutoff frequency D0 (in pixels from center).
            order: Filter order n. Higher = steeper roll-off.
            filter_type: "lpf" (low-pass) or "hpf" (high-pass).
        """
        self.rows = rows
        self.cols = cols
        self.cutoff = cutoff
        self.order = order
        self.filter_type = filter_type
        self._mask: np.ndarray = self._build_mask()

    def _build_mask(self) -> np.ndarray:
        D = FFTTransform.get_distance_grid(self.rows, self.cols)
        # Avoid division by zero at center
        D = np.where(D == 0, 1e-10, D)

        if self.filter_type == "lpf":
            H = 1.0 / (1.0 + (D / self.cutoff) ** (2 * self.order))
        else:
            H = 1.0 / (1.0 + (self.cutoff / D) ** (2 * self.order))
        return H

    def get_mask(self) -> np.ndarray:
        """Return the filter mask as a float64 array in [0, 1]."""
        return self._mask.copy()

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to a grayscale image.

        Args:
            image: Grayscale uint8 or float image of shape (H, W).
                   Must match rows×cols used at construction.

        Returns:
            Filtered uint8 image.
        """
        if image.shape[:2] != (self.rows, self.cols):
            raise ValueError(
                f"Image shape {image.shape[:2]} != filter shape ({self.rows},{self.cols})"
            )
        fft_shifted = FFTTransform.compute_fft(image)
        filtered_fft = fft_shifted * self._mask
        return FFTTransform.compute_ifft(filtered_fft)

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        cutoff: float = 30.0,
        order: int = 2,
        filter_type: FilterType = "lpf",
    ) -> "ButterworthFilter":
        """Convenience factory that reads shape from image."""
        h, w = image.shape[:2]
        return cls(h, w, cutoff, order, filter_type)

    @staticmethod
    def apply_with_params(
        image: np.ndarray,
        cutoff: float,
        order: int,
        filter_type: FilterType = "lpf",
    ) -> np.ndarray:
        """Build a filter from the image shape and apply it in one call.

        Mirrors GaussianLPF.apply_with_params for parameter sweeps.
        """
        return ButterworthFilter.from_image(image, cutoff, order, filter_type).apply(image)

    def experiment(
        self,
        image: np.ndarray,
        cutoffs: list = (10, 30, 50, 80),
        orders: list = (1, 2, 4),
    ) -> dict:
        """Run parameter sweep for cutoff and order.

        Returns:
            Dict keyed by "d{cutoff}_n{order}" with filtered arrays.
        """
        results = {}
        for d0 in cutoffs:
            for n in orders:
                key = f"d{d0}_n{n}"
                filt = ButterworthFilter(
                    self.rows, self.cols, d0, n, self.filter_type
                )
                results[key] = filt.apply(image)
        return results

"""2D FFT and inverse FFT utilities for frequency-domain image processing."""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FFTTransform:
    """Compute 2D FFT, spectrum visualization, and inverse FFT."""

    @staticmethod
    def compute_fft(image: np.ndarray) -> np.ndarray:
        """Compute shifted 2D FFT of a grayscale image.

        Args:
            image: Grayscale uint8 or float image of shape (H, W).

        Returns:
            Complex array of shape (H, W) — zero-frequency component centered.
        """
        img_float = image.astype(np.float64)
        fft = np.fft.fft2(img_float)
        fft_shifted = np.fft.fftshift(fft)
        return fft_shifted

    @staticmethod
    def compute_ifft(fft_shifted: np.ndarray) -> np.ndarray:
        """Reconstruct image from shifted FFT.

        Args:
            fft_shifted: Complex FFT array (zero-centered).

        Returns:
            Real-valued reconstructed image, clipped to [0, 255], uint8.
        """
        fft_unshifted = np.fft.ifftshift(fft_shifted)
        reconstructed = np.fft.ifft2(fft_unshifted)
        magnitude = np.abs(reconstructed)
        clipped = np.clip(magnitude, 0, 255)
        return clipped.astype(np.uint8)

    @staticmethod
    def get_magnitude_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
        """Compute log-scaled magnitude spectrum for display.

        Returns:
            uint8 array suitable for imshow.
        """
        magnitude = np.abs(fft_shifted)
        log_spectrum = np.log1p(magnitude)
        normalized = (log_spectrum / log_spectrum.max() * 255).astype(np.uint8)
        return normalized

    @staticmethod
    def get_phase_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
        """Compute phase spectrum normalized to [0, 255]."""
        phase = np.angle(fft_shifted)
        normalized = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        return normalized

    @staticmethod
    def get_frequency_grid(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (u, v) frequency coordinate grids centered at (rows//2, cols//2)."""
        u = np.fft.fftfreq(rows) * rows
        v = np.fft.fftfreq(cols) * cols
        u_shifted = np.fft.fftshift(u)
        v_shifted = np.fft.fftshift(v)
        V, U = np.meshgrid(v_shifted, u_shifted)
        return U, V

    @staticmethod
    def get_distance_grid(rows: int, cols: int) -> np.ndarray:
        """Return distance-from-center grid D(u,v) = sqrt(u²+v²)."""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        V, U = np.meshgrid(v, u)
        return np.sqrt(U**2 + V**2)

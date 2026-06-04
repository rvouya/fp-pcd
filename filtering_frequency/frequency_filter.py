"""High-level frequency filtering pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .butterworth import ButterworthFilter
from .fft_transform import FFTTransform

logger = logging.getLogger(__name__)


class FrequencyFilter:
    """End-to-end frequency domain filtering pipeline.

    Supports Butterworth LPF and HPF with configurable parameters.
    Also exposes raw FFT spectrum and inverse FFT outputs.
    """

    def __init__(
        self,
        cutoff: float = 30.0,
        order: int = 2,
    ) -> None:
        self.cutoff = cutoff
        self.order = order

    # ------------------------------------------------------------------
    # Single-image operations
    # ------------------------------------------------------------------

    def apply_lpf(
        self,
        image: np.ndarray,
        cutoff: Optional[float] = None,
        order: Optional[int] = None,
    ) -> np.ndarray:
        """Apply Butterworth LPF."""
        d0 = cutoff or self.cutoff
        n = order or self.order
        filt = ButterworthFilter.from_image(image, d0, n, "lpf")
        return filt.apply(image)

    def apply_hpf(
        self,
        image: np.ndarray,
        cutoff: Optional[float] = None,
        order: Optional[int] = None,
    ) -> np.ndarray:
        """Apply Butterworth HPF."""
        d0 = cutoff or self.cutoff
        n = order or self.order
        filt = ButterworthFilter.from_image(image, d0, n, "hpf")
        return filt.apply(image)

    def full_pipeline(
        self,
        image: np.ndarray,
        cutoff: Optional[float] = None,
        order: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run full frequency pipeline and return all intermediate outputs.

        Returns dict with keys:
            - fft_spectrum: log magnitude spectrum (uint8)
            - lpf: Butterworth LPF result
            - hpf: Butterworth HPF result
            - inverse_fft_lpf: IFFT of LPF
        """
        d0 = cutoff or self.cutoff
        n = order or self.order

        fft_shifted = FFTTransform.compute_fft(image)
        spectrum = FFTTransform.get_magnitude_spectrum(fft_shifted)

        lpf_result = self.apply_lpf(image, d0, n)
        hpf_result = self.apply_hpf(image, d0, n)

        # IFFT on LPF (already done inside apply_lpf, re-expose here for clarity)
        filt_lpf = ButterworthFilter.from_image(image, d0, n, "lpf")
        filtered_fft = fft_shifted * filt_lpf.get_mask()
        inverse_fft_result = FFTTransform.compute_ifft(filtered_fft)

        return {
            "fft_spectrum": spectrum,
            "lpf": lpf_result,
            "hpf": hpf_result,
            "inverse_fft_lpf": inverse_fft_result,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    @staticmethod
    def _save(image: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    def batch_process(
        self,
        image_paths: List[Path],
        output_dir: Path,
        cutoff: float = 30.0,
        order: int = 2,
    ) -> Dict[str, List[Path]]:
        """Process all images through full frequency pipeline.

        Saves FFT spectra, LPF results, HPF results, and inverse FFT results.

        Returns:
            Dict mapping stage name to list of saved paths.
        """
        stages = {
            "fft": output_dir / "fft",
            "butterworth_lpf": output_dir / "butterworth_lpf",
            "butterworth_hpf": output_dir / "butterworth_hpf",
            "inverse_fft": output_dir / "inverse_fft",
        }
        for d in stages.values():
            d.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, List[Path]] = {k: [] for k in stages}

        for src in image_paths:
            try:
                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError(f"Cannot read {src}")
                results = self.full_pipeline(img, cutoff, order)

                self._save(results["fft_spectrum"], stages["fft"] / src.name)
                self._save(results["lpf"], stages["butterworth_lpf"] / src.name)
                self._save(results["hpf"], stages["butterworth_hpf"] / src.name)
                self._save(results["inverse_fft_lpf"], stages["inverse_fft"] / src.name)

                saved["fft"].append(stages["fft"] / src.name)
                saved["butterworth_lpf"].append(stages["butterworth_lpf"] / src.name)
                saved["butterworth_hpf"].append(stages["butterworth_hpf"] / src.name)
                saved["inverse_fft"].append(stages["inverse_fft"] / src.name)

            except Exception as exc:
                logger.error("Frequency batch error on %s: %s", src, exc)

        logger.info("Frequency batch done. LPF: %d, HPF: %d images",
                    len(saved["butterworth_lpf"]), len(saved["butterworth_hpf"]))
        return saved

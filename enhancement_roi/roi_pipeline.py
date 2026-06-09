"""ROI Enhancement Pipeline combining CLAHE with morphological operations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .clahe import CLAHEEnhancer
from .morphological import MorphologicalOps

logger = logging.getLogger(__name__)

PipelineMode = str  # "clahe" | "clahe_tophat" | "clahe_opening" | "clahe_closing"


class ROIEnhancementPipeline:
    """Combine CLAHE and morphological ops into configurable ROI enhancement pipelines.

    Supported pipeline modes:
        - "clahe"         : CLAHE only
        - "clahe_tophat"  : CLAHE → Top-Hat
        - "clahe_opening" : CLAHE → Opening
        - "clahe_closing" : CLAHE → Closing
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        morph_kernel_size: int = 15,
    ) -> None:
        self.clahe = CLAHEEnhancer(clip_limit, tile_grid_size)
        self.morph = MorphologicalOps()
        self.morph_kernel_size = morph_kernel_size

    # ------------------------------------------------------------------
    # Pipeline variants
    # ------------------------------------------------------------------

    def clahe_only(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE only."""
        return self.clahe.apply(image)

    def clahe_tophat(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE then Top-Hat transform."""
        enhanced = self.clahe.apply(image)
        return self.morph.top_hat(enhanced, self.morph_kernel_size)

    def clahe_opening(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE then morphological Opening."""
        enhanced = self.clahe.apply(image)
        return self.morph.opening(enhanced, self.morph_kernel_size)

    def clahe_closing(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE then morphological Closing."""
        enhanced = self.clahe.apply(image)
        return self.morph.closing(enhanced, self.morph_kernel_size)

    def apply(self, image: np.ndarray, mode: PipelineMode = "clahe") -> np.ndarray:
        """Apply a named pipeline mode.

        Args:
            image: uint8 grayscale image.
            mode: One of "clahe", "clahe_tophat", "clahe_opening", "clahe_closing".

        Returns:
            Enhanced image.
        """
        dispatch = {
            "clahe": self.clahe_only,
            "clahe_tophat": self.clahe_tophat,
            "clahe_opening": self.clahe_opening,
            "clahe_closing": self.clahe_closing,
        }
        if mode not in dispatch:
            raise ValueError(f"Unknown mode {mode!r}. Choose from {list(dispatch)}")
        return dispatch[mode](image)

    def apply_all(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Return results for all pipeline modes.

        Returns:
            Dict with keys "original", "clahe", "clahe_tophat",
            "clahe_opening", "clahe_closing".
        """
        return {
            "original": image,
            "clahe": self.clahe_only(image),
            "clahe_tophat": self.clahe_tophat(image),
            "clahe_opening": self.clahe_opening(image),
            "clahe_closing": self.clahe_closing(image),
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
        modes: Optional[List[PipelineMode]] = None,
    ) -> Dict[str, List[Path]]:
        """Apply all (or selected) pipeline modes to a list of images.

        Args:
            image_paths: Source image paths.
            output_dir: Root output directory (subdirs created per mode).
            modes: Subset of modes to run. Default: all four modes.

        Returns:
            Dict mapping mode name to list of saved output paths.
        """
        if modes is None:
            modes = ["clahe", "clahe_tophat", "clahe_opening", "clahe_closing"]

        out_dirs = {
            "clahe": output_dir / "clahe",
            "clahe_tophat": output_dir / "top_hat",
            "clahe_opening": output_dir / "opening",
            "clahe_closing": output_dir / "closing",
        }
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, List[Path]] = {m: [] for m in modes}

        for src in image_paths:
            try:
                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise IOError(f"Cannot read {src}")

                for mode in modes:
                    result = self.apply(img, mode)
                    key = mode
                    out_path = out_dirs[key] / src.name
                    self._save(result, out_path)
                    saved[key].append(out_path)

            except Exception as exc:
                logger.error("ROI batch error on %s: %s", src, exc)

        for mode in modes:
            logger.info("ROI [%s]: %d images saved", mode, len(saved[mode]))

        return saved

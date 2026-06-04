"""CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement."""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CLAHEEnhancer:
    """Apply CLAHE to grayscale X-Ray images.

    Supports parameter experiments over clip limit and tile grid size.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """Initialize CLAHE with default parameters.

        Args:
            clip_limit: Threshold for contrast limiting. Higher = more contrast.
            tile_grid_size: Size of grid for histogram equalization.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a single grayscale image.

        Args:
            image: uint8 grayscale image.

        Returns:
            CLAHE-enhanced uint8 image.
        """
        return self.apply_with_params(image, self.clip_limit, self.tile_grid_size)

    @staticmethod
    def apply_with_params(
        image: np.ndarray,
        clip_limit: float,
        tile_grid_size: Tuple[int, int],
    ) -> np.ndarray:
        """Apply CLAHE with explicit parameters.

        Args:
            image: uint8 grayscale image.
            clip_limit: CLAHE clip limit.
            tile_grid_size: (rows, cols) grid size.

        Returns:
            Enhanced uint8 image.
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )
        return clahe.apply(image)

    def experiment(
        self,
        image: np.ndarray,
        clip_limits: List[float] = (1.0, 2.0, 3.0, 5.0, 8.0),
        tile_grid_sizes: List[Tuple[int, int]] = ((4, 4), (8, 8), (16, 16)),
    ) -> Dict[str, np.ndarray]:
        """Run parameter sweep over clip limits and tile grid sizes.

        Returns:
            Dict keyed by "cl{clip_limit}_tg{rows}x{cols}".
        """
        results: Dict[str, np.ndarray] = {}
        for cl in clip_limits:
            for tg in tile_grid_sizes:
                key = f"cl{cl}_tg{tg[0]}x{tg[1]}"
                results[key] = self.apply_with_params(image, cl, tg)
        logger.debug("CLAHE experiment: %d variants", len(results))
        return results

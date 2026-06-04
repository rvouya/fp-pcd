"""Image preprocessing utilities: resize and normalization."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_SIZES: Dict[str, Tuple[int, int]] = {
    "224": (224, 224),
    "512": (512, 512),
    "1024": (1024, 1024),
}

NORMALIZATION_METHODS = ("minmax", "zscore")


class Preprocessor:
    """Resize and normalize chest X-Ray images.

    All methods are pure (return new arrays, never mutate inputs).
    """

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    @staticmethod
    def resize(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """Resize image to target_size (width, height).

        Args:
            image: Input grayscale or BGR image.
            target_size: (width, height) tuple.
            interpolation: OpenCV interpolation flag.

        Returns:
            Resized image as uint8.
        """
        resized = cv2.resize(image, target_size, interpolation=interpolation)
        return resized

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def minmax_normalize(image: np.ndarray) -> np.ndarray:
        """Scale pixel values to [0, 1] using min-max normalization.

        Args:
            image: Input image array (any numeric dtype).

        Returns:
            Float64 array in [0.0, 1.0].
        """
        img_float = image.astype(np.float64)
        min_val = img_float.min()
        max_val = img_float.max()
        if max_val == min_val:
            return np.zeros_like(img_float)
        return (img_float - min_val) / (max_val - min_val)

    @staticmethod
    def zscore_normalize(image: np.ndarray) -> np.ndarray:
        """Z-score normalize pixel values (zero mean, unit variance).

        Args:
            image: Input image array (any numeric dtype).

        Returns:
            Float64 array with mean≈0, std≈1.
        """
        img_float = image.astype(np.float64)
        mean = img_float.mean()
        std = img_float.std()
        if std == 0.0:
            return np.zeros_like(img_float)
        return (img_float - mean) / std

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Clip and convert float image to uint8 [0, 255].

        Assumes float image is in [0, 1]; clamps before scaling.
        """
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        norm_method: str = "minmax",
    ) -> np.ndarray:
        """Apply resize then normalization.

        Args:
            image: Input grayscale image.
            target_size: (width, height). Pass None to skip resize.
            norm_method: "minmax" or "zscore".

        Returns:
            Processed float64 array.
        """
        if norm_method not in NORMALIZATION_METHODS:
            raise ValueError(
                f"norm_method must be one of {NORMALIZATION_METHODS}, got {norm_method!r}"
            )
        result = image.copy()
        if target_size is not None:
            result = self.resize(result, target_size)
        if norm_method == "minmax":
            result = self.minmax_normalize(result)
        else:
            result = self.zscore_normalize(result)
        return result

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save(image: np.ndarray, path: Path) -> None:
        """Save image to disk.

        Float images are converted to uint8 before saving.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if image.dtype in (np.float32, np.float64):
            save_img = Preprocessor.to_uint8(image)
        else:
            save_img = image
        cv2.imwrite(str(path), save_img)

    def batch_process(
        self,
        image_paths: List[Path],
        output_dir: Path,
        target_size: Tuple[int, int] = (224, 224),
        norm_method: str = "minmax",
        subfolder: str = "",
    ) -> List[Path]:
        """Process a list of image files and save results.

        Args:
            image_paths: List of source image paths.
            output_dir: Root output directory.
            target_size: Resize target.
            norm_method: Normalization method.
            subfolder: Optional subfolder under output_dir.

        Returns:
            List of saved output paths.
        """
        from .dataset_loader import DatasetLoader

        loader = DatasetLoader()
        dest_dir = output_dir / subfolder if subfolder else output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for src_path in image_paths:
            try:
                img = loader.load_image(src_path)
                processed = self.process(img, target_size, norm_method)
                out_path = dest_dir / src_path.name
                self.save(processed, out_path)
                saved.append(out_path)
            except Exception as exc:
                logger.error("Failed to process %s: %s", src_path, exc)

        logger.info("Batch processed %d/%d images -> %s", len(saved), len(image_paths), dest_dir)
        return saved

    def compare_sizes(
        self,
        image: np.ndarray,
        sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return dict of resized images for comparison.

        Args:
            image: Input image.
            sizes: List of (w, h) tuples. Defaults to 224, 512, 1024.

        Returns:
            {"224x224": arr, "512x512": arr, ...}
        """
        if sizes is None:
            sizes = [(224, 224), (512, 512), (1024, 1024)]
        return {f"{w}x{h}": self.resize(image, (w, h)) for w, h in sizes}

    def compare_normalizations(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return minmax and zscore normalized versions for comparison."""
        return {
            "original": image.astype(np.float64),
            "minmax": self.minmax_normalize(image),
            "zscore": self.zscore_normalize(image),
        }

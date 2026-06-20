"""PyTorch Dataset and DataModule for chest X-Ray classification."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalSplit:
    """Label mapping and train/val membership shared by every scenario.

    Built once from the full labels CSV so that all scenarios use the same
    integer label indices and the same canonical train/val image membership.
    This makes cross-scenario accuracy comparisons apples-to-apples; each
    scenario only drops the images that are physically missing from its own
    ``image_dir``.
    """

    class_names: List[str]
    label_to_idx: Dict[str, int]
    train_image_indices: List[str]
    val_image_indices: List[str]


def build_canonical_split(
    labels_csv: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> CanonicalSplit:
    """Build the canonical label map + train/val split from the full CSV.

    The class list and the train/val partition are derived from every row in
    the labels CSV (not from what happens to exist on disk for any one
    scenario), so they are identical across scenarios.

    Args:
        labels_csv: Path to the labels CSV (``Image Index``/``Finding Labels``).
        val_split: Fraction of rows held out for validation.
        seed: RNG seed for the deterministic permutation.

    Returns:
        A ``CanonicalSplit`` with sorted class names, label→index mapping, and
        the train/val image_index (filename) membership.
    """
    df = pd.read_csv(labels_csv)
    df = df.rename(columns={"Image Index": "image_index", "Finding Labels": "label"})

    class_names = sorted(df["label"].unique().tolist())
    label_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    image_indices = df["image_index"].tolist()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(image_indices), generator=generator).tolist()
    n_val = int(len(image_indices) * val_split)
    n_train = len(image_indices) - n_val

    train_image_indices = [image_indices[i] for i in perm[:n_train]]
    val_image_indices = [image_indices[i] for i in perm[n_train:]]

    logger.info(
        "Canonical split: %d classes, %d train / %d val images (seed=%d)",
        len(class_names), len(train_image_indices), len(val_image_indices), seed,
    )
    return CanonicalSplit(
        class_names=class_names,
        label_to_idx=label_to_idx,
        train_image_indices=train_image_indices,
        val_image_indices=val_image_indices,
    )


def default_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """Return standard augmentation + normalization pipeline for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def default_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> Callable:
    """Return deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class XRayDataset(Dataset):
    """PyTorch Dataset for chest X-Ray images.

    Loads images from disk lazily and applies transforms on access.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_paths: List of absolute paths to image files.
            labels: Integer class indices corresponding to image_paths.
            class_names: List of class name strings (index → name).
            transform: torchvision transform pipeline.
        """
        assert len(image_paths) == len(labels), "Paths and labels must have equal length."
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Cannot read %s, using blank image", img_path)
            img = np.zeros((224, 224), dtype=np.uint8)

        # Convert grayscale to 3-channel for pretrained RGB models
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = transforms.ToTensor()(img_rgb)

        return img_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        counts = torch.zeros(len(self.class_names))
        for lbl in self.labels:
            counts[lbl] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum() * len(self.class_names)


class XRayDataModule:
    """Manages train/val splits and DataLoader creation for one scenario.

    Reads a CSV label file and matches images from the given directory. The
    label mapping and train/val membership come from an injected
    ``CanonicalSplit`` (built once from the full CSV) so every scenario shares
    the same labels and the same canonical split; this scenario only drops the
    images missing from its own ``image_dir``.
    """

    def __init__(
        self,
        image_dir: Path,
        labels_csv: Path,
        canonical_split: Optional[CanonicalSplit] = None,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        self.image_dir = image_dir
        self.labels_csv = labels_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

        # Fall back to building the canonical split locally if not injected, so
        # the DataModule still works standalone (tests, ad-hoc use).
        self.canonical_split = canonical_split or build_canonical_split(
            labels_csv, val_split=val_split, seed=seed
        )

        self._label_map: Dict[str, str] = self._load_label_map()

    def _load_label_map(self) -> Dict[str, str]:
        """Map image_index (filename) -> disease label from the labels CSV."""
        df = pd.read_csv(self.labels_csv)
        df = df.rename(columns={"Image Index": "image_index", "Finding Labels": "label"})
        return dict(zip(df["image_index"], df["label"]))

    @property
    def class_names(self) -> List[str]:
        return self.canonical_split.class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _build_dataset(
        self, image_indices: List[str], transform: Callable
    ) -> XRayDataset:
        """Build a dataset from canonical image_index names present on disk."""
        label_to_idx = self.canonical_split.label_to_idx
        paths: List[Path] = []
        labels: List[int] = []
        dropped = 0
        for name in image_indices:
            path = self.image_dir / name
            if not path.exists():
                dropped += 1
                continue
            paths.append(path)
            labels.append(label_to_idx[self._label_map[name]])
        if dropped:
            logger.info(
                "Scenario %s: dropped %d/%d images missing from %s",
                self.image_dir.name, dropped, len(image_indices), self.image_dir,
            )
        return XRayDataset(paths, labels, self.class_names, transform)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader)."""
        train_ds = self._build_dataset(
            self.canonical_split.train_image_indices,
            default_train_transforms(self.image_size),
        )
        val_ds = self._build_dataset(
            self.canonical_split.val_image_indices,
            default_val_transforms(self.image_size),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

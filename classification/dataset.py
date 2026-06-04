"""PyTorch Dataset and DataModule for chest X-Ray classification."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)


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
    """Manages train/val/test splits and DataLoader creation.

    Reads a CSV label file and matches images from the given directory.
    """

    def __init__(
        self,
        image_dir: Path,
        labels_csv: Path,
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

        self._df: Optional[pd.DataFrame] = None
        self._class_names: Optional[List[str]] = None
        self._label_to_idx: Optional[Dict[str, int]] = None

    def _load_metadata(self) -> None:
        df = pd.read_csv(self.labels_csv)
        df = df.rename(columns={"Image Index": "image_index", "Finding Labels": "label"})

        # Only keep rows where image file exists
        df["path"] = df["image_index"].apply(lambda x: self.image_dir / x)
        df = df[df["path"].apply(lambda p: p.exists())].copy()

        self._class_names = sorted(df["label"].unique().tolist())
        self._label_to_idx = {cls: idx for idx, cls in enumerate(self._class_names)}
        df["label_idx"] = df["label"].map(self._label_to_idx)
        self._df = df.reset_index(drop=True)
        logger.info(
            "DataModule loaded %d samples, %d classes from %s",
            len(df), len(self._class_names), self.image_dir,
        )

    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            self._load_metadata()
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _build_dataset(
        self, df_subset: pd.DataFrame, transform: Callable
    ) -> XRayDataset:
        paths = [Path(p) for p in df_subset["path"].tolist()]
        labels = df_subset["label_idx"].tolist()
        return XRayDataset(paths, labels, self.class_names, transform)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader)."""
        if self._df is None:
            self._load_metadata()

        generator = torch.Generator().manual_seed(self.seed)
        n_val = int(len(self._df) * self.val_split)
        n_train = len(self._df) - n_val

        indices = torch.randperm(len(self._df), generator=generator).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_df = self._df.iloc[train_indices].reset_index(drop=True)
        val_df = self._df.iloc[val_indices].reset_index(drop=True)

        train_ds = self._build_dataset(
            train_df, default_train_transforms(self.image_size)
        )
        val_ds = self._build_dataset(
            val_df, default_val_transforms(self.image_size)
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

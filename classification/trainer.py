"""Training loop, checkpointing, and inference for chest X-Ray classifiers."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import ClassificationConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full train/validate/save cycle.

    Args:
        model: PyTorch module.
        config: ClassificationConfig instance.
        device: "cuda", "mps", or "cpu".
    """

    def __init__(
        self,
        model: nn.Module,
        config: ClassificationConfig,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )

        self._best_val_loss = float("inf")
        self._no_improve_count = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Run full training with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            output_dir: Where to save checkpoints and logs.

        Returns:
            Training history dict.
        """
        ckpt_dir = output_dir or self.config.checkpoint_dir
        log_dir = self.config.log_dir
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training on %s for %d epochs", self.device, self.config.num_epochs
        )

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | %.1fs",
                epoch, self.config.num_epochs,
                train_loss, train_acc, val_loss, val_acc, elapsed,
            )

            # Save best checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._no_improve_count = 0
                if ckpt_dir:
                    self.save_checkpoint(ckpt_dir / "best.pth", epoch, val_loss)
            else:
                self._no_improve_count += 1

            # Early stopping
            if self._no_improve_count >= self.config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Save final history
        if log_dir:
            with open(log_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, object]:
        """Run inference on a DataLoader and collect predictions.

        Returns:
            Dict with keys: y_true (list[int]), y_pred (list[int]),
            y_prob (list[list[float]]), accuracy (float).
        """
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        for images, labels in loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

        accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / max(len(y_true), 1)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": accuracy,
        }

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config.__dict__,
            },
            path,
        )
        logger.debug("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """Load model weights from checkpoint.

        Returns:
            The epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt["epoch"])
        return ckpt["epoch"]

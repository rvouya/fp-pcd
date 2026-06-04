"""Classification evaluation metrics: accuracy, precision, recall, F1, ROC AUC."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_precision(
    y_true: List[int],
    y_pred: List[int],
    average: str = "weighted",
) -> float:
    """Compute precision with the given averaging strategy."""
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def compute_recall(
    y_true: List[int],
    y_pred: List[int],
    average: str = "weighted",
) -> float:
    """Compute recall with the given averaging strategy."""
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def compute_f1(
    y_true: List[int],
    y_pred: List[int],
    average: str = "weighted",
) -> float:
    """Compute F1 score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_roc_auc(
    y_true: List[int],
    y_prob: List[List[float]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class ROC AUC scores using one-vs-rest.

    Args:
        y_true: Ground-truth class indices.
        y_prob: Predicted probabilities shape (N, C).
        class_names: Optional class name list for keys.

    Returns:
        Dict mapping class name (or index) → AUC value, plus "macro_avg".
    """
    y_prob_arr = np.array(y_prob)
    n_classes = y_prob_arr.shape[1]
    classes = list(range(n_classes))
    y_bin = label_binarize(y_true, classes=classes)

    aucs: Dict[str, float] = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        auc_val = float(auc(fpr, tpr))
        key = class_names[i] if class_names else str(i)
        aucs[key] = auc_val

    aucs["macro_avg"] = float(np.mean(list(aucs.values())))
    return aucs


class ClassificationMetricsEvaluator:
    """Compute, visualize, and export classification metrics."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.class_names = class_names
        self.output_dir = output_dir

    def compute_all(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[List[float]]] = None,
    ) -> Dict[str, object]:
        """Compute all classification metrics.

        Returns:
            Dict with accuracy, precision, recall, f1, roc_auc (if probs provided),
            and full classification_report string.
        """
        results: Dict[str, object] = {
            "accuracy": compute_accuracy(y_true, y_pred),
            "precision_weighted": compute_precision(y_true, y_pred, "weighted"),
            "recall_weighted": compute_recall(y_true, y_pred, "weighted"),
            "f1_weighted": compute_f1(y_true, y_pred, "weighted"),
            "precision_macro": compute_precision(y_true, y_pred, "macro"),
            "recall_macro": compute_recall(y_true, y_pred, "macro"),
            "f1_macro": compute_f1(y_true, y_pred, "macro"),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                zero_division=0,
            ),
        }
        if y_prob is not None:
            try:
                results["roc_auc"] = compute_roc_auc(y_true, y_prob, self.class_names)
            except Exception as exc:
                logger.warning("ROC AUC computation failed: %s", exc)
                results["roc_auc"] = {}

        logger.info(
            "Metrics — accuracy=%.4f, f1_weighted=%.4f",
            results["accuracy"], results["f1_weighted"],
        )
        return results

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        save_path: Optional[Path] = None,
        title: str = "Confusion Matrix",
    ) -> None:
        """Plot and optionally save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Confusion matrix saved: %s", save_path)
        plt.close(fig)

    def plot_roc_curves(
        self,
        y_true: List[int],
        y_prob: List[List[float]],
        save_path: Optional[Path] = None,
        title: str = "ROC Curves",
    ) -> None:
        """Plot per-class ROC curves."""
        y_prob_arr = np.array(y_prob)
        n_classes = y_prob_arr.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
            auc_val = auc(fpr, tpr)
            label = self.class_names[i] if self.class_names else str(i)
            ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={auc_val:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("ROC curves saved: %s", save_path)
        plt.close(fig)

    def metrics_to_dataframe(
        self, results: Dict[str, object], scenario: str = ""
    ) -> pd.DataFrame:
        """Convert scalar metrics dict to a single-row DataFrame."""
        scalar_keys = [
            "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro",
        ]
        row = {"scenario": scenario}
        for k in scalar_keys:
            row[k] = results.get(k, float("nan"))
        return pd.DataFrame([row])

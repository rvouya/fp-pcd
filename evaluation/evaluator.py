"""Master experiment evaluator: aggregates image and classification metrics."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .classification_metrics import ClassificationMetricsEvaluator
from .image_metrics import ImageMetricsEvaluator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Canonical pipeline-step order for the image-quality report, so "step N image
# quality" lines up with "step N classification accuracy" (spatial, then
# frequency, then ROI enhancement spatial-source then frequency-source).
STAGE_ORDER: List[str] = [
    "spatial_gaussian",
    "spatial_unsharp",
    "freq_lpf",
    "freq_hpf",
    "roi_spatial_clahe",
    "roi_spatial_tophat",
    "roi_spatial_opening",
    "roi_spatial_closing",
    "roi_spatial_histeq",
    "roi_spatial_gamma",
    "roi_frequency_clahe",
    "roi_frequency_tophat",
    "roi_frequency_opening",
    "roi_frequency_closing",
    "roi_frequency_histeq",
    "roi_frequency_gamma",
]


class ExperimentEvaluator:
    """Aggregate all evaluation results across pipeline stages and scenarios.

    Usage:
        evaluator = ExperimentEvaluator()
        evaluator.evaluate_image_quality(...)
        evaluator.evaluate_classification(...)
        evaluator.export_all()
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or (OUTPUT_ROOT / "06_evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._image_results: List[pd.DataFrame] = []
        self._classification_results: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Image quality evaluation
    # ------------------------------------------------------------------

    def evaluate_image_quality(
        self,
        original_dir: Optional[Path] = None,
        stage_dirs: Optional[Dict[str, Path]] = None,
    ) -> pd.DataFrame:
        """Compute PSNR/SSIM for all filtering stages vs. original images.

        If original_dir or stage_dirs are None, uses default output layout.

        Returns:
            DataFrame with columns [image_name, stage, psnr, ssim].
        """
        if original_dir is None:
            # Clean reference (groundtruth, resized+normalized the same way as the
            # corrupted set) so PSNR/SSIM measure restoration quality, not just
            # "different from the corrupted input".
            original_dir = OUTPUT_ROOT / "01_preprocessing" / "groundtruth_normalized"

        if stage_dirs is None:
            stage_dirs = {
                "spatial_gaussian": OUTPUT_ROOT / "02_spatial_filtering" / "gaussian",
                "spatial_unsharp": OUTPUT_ROOT / "02_spatial_filtering" / "unsharp",
                "freq_lpf": OUTPUT_ROOT / "03_frequency_filtering" / "butterworth_lpf",
                "freq_hpf": OUTPUT_ROOT / "03_frequency_filtering" / "butterworth_hpf",
                "roi_spatial_clahe": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "clahe",
                "roi_spatial_tophat": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "top_hat",
                "roi_spatial_opening": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "opening",
                "roi_spatial_closing": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "closing",
                "roi_spatial_histeq": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "histeq",
                "roi_spatial_gamma": OUTPUT_ROOT / "04_roi_enhancement" / "spatial" / "gamma",
                "roi_frequency_clahe": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "clahe",
                "roi_frequency_tophat": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "top_hat",
                "roi_frequency_opening": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "opening",
                "roi_frequency_closing": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "closing",
                "roi_frequency_histeq": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "histeq",
                "roi_frequency_gamma": OUTPUT_ROOT / "04_roi_enhancement" / "frequency" / "gamma",
            }

        evaluator = ImageMetricsEvaluator(self.output_dir)
        df = evaluator.compare_stages(original_dir, stage_dirs)
        self._image_results.append(df)
        logger.info("Image quality evaluation: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Classification evaluation
    # ------------------------------------------------------------------

    def evaluate_classification(
        self,
        scenario: str,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Compute classification metrics for one scenario.

        Args:
            scenario: Name of the training scenario.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_prob: Class probabilities (optional, for ROC AUC).
            class_names: Class name list.

        Returns:
            Metrics dict.
        """
        eval_cls = ClassificationMetricsEvaluator(class_names, self.output_dir)
        results = eval_cls.compute_all(y_true, y_pred, y_prob)

        # Save confusion matrix
        cm_path = self.output_dir / "classification_metrics" / f"cm_{scenario}.png"
        eval_cls.plot_confusion_matrix(y_true, y_pred, cm_path, title=f"CM: {scenario}")

        # Save ROC if probabilities available
        if y_prob:
            roc_path = self.output_dir / "classification_metrics" / f"roc_{scenario}.png"
            try:
                eval_cls.plot_roc_curves(y_true, y_prob, roc_path, title=f"ROC: {scenario}")
            except Exception as exc:
                logger.warning("ROC plot failed for %s: %s", scenario, exc)

        row_df = eval_cls.metrics_to_dataframe(results, scenario)
        self._classification_results.append(row_df)
        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV under output_dir."""
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("CSV saved: %s", path)
        return path

    def export_json(self, data: Dict, filename: str) -> Path:
        """Save dict as JSON under output_dir."""
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON saved: %s", path)
        return path

    def load_classification_results(self, csv_path: Optional[Path] = None) -> None:
        """Load a previously exported classification metrics CSV into this evaluator.

        Lets a later, separate evaluation run (e.g. after classification training
        finished in its own process) merge into one combined summary report.
        """
        path = csv_path or (self.output_dir / "classification_metrics" / "all_scenarios.csv")
        if path.exists():
            self._classification_results.append(pd.read_csv(path))
            logger.info("Loaded existing classification results from %s", path)
        else:
            logger.info("No existing classification results at %s, skipping", path)

    def export_all(self) -> None:
        """Write consolidated CSVs and JSON summaries for all collected results."""
        # Image quality
        if self._image_results:
            df_img = pd.concat(self._image_results, ignore_index=True)
            self.export_csv(df_img, "psnr/image_quality_metrics.csv")

            summary_img = (
                df_img.groupby("stage")[["psnr", "ssim"]]
                .agg(["mean", "std"])
                .reset_index()
            )
            self.export_csv(summary_img, "psnr/image_quality_summary.csv")

        # Classification
        if self._classification_results:
            df_cls = pd.concat(self._classification_results, ignore_index=True)
            self.export_csv(df_cls, "classification_metrics/all_scenarios.csv")

            best_idx = df_cls["f1_weighted"].idxmax()
            best_row = df_cls.loc[best_idx].to_dict()
            self.export_json(best_row, "classification_metrics/best_scenario.json")

        logger.info("All evaluation exports complete -> %s", self.output_dir)

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary of all experiments."""
        lines = ["=" * 60, "EXPERIMENT SUMMARY REPORT", "=" * 60, ""]

        if self._image_results:
            df_img = pd.concat(self._image_results, ignore_index=True)
            lines.append("IMAGE QUALITY METRICS (mean +/- std per stage):")
            grouped = df_img.groupby("stage")[["psnr", "ssim"]].agg(["mean", "std"])
            # Order stages by pipeline step; unknown stages fall to the end.
            order = {name: i for i, name in enumerate(STAGE_ORDER)}
            ordered_stages = sorted(grouped.index, key=lambda s: order.get(s, len(order)))
            lines.append(f"{'stage':<24}{'PSNR (mean +/- std)':<26}{'SSIM (mean +/- std)':<26}")
            for stage in ordered_stages:
                psnr_m = grouped.loc[stage, ("psnr", "mean")]
                psnr_s = grouped.loc[stage, ("psnr", "std")]
                ssim_m = grouped.loc[stage, ("ssim", "mean")]
                ssim_s = grouped.loc[stage, ("ssim", "std")]
                psnr_str = f"{psnr_m:.2f} +/- {0.0 if pd.isna(psnr_s) else psnr_s:.2f}"
                ssim_str = f"{ssim_m:.4f} +/- {0.0 if pd.isna(ssim_s) else ssim_s:.4f}"
                lines.append(f"{stage:<24}{psnr_str:<26}{ssim_str:<26}")
            lines.append("")

        if self._classification_results:
            df_cls = pd.concat(self._classification_results, ignore_index=True)
            lines.append("CLASSIFICATION RESULTS:")
            cols = ["scenario", "accuracy", "f1_weighted", "f1_macro"]
            lines.append(df_cls[cols].to_string(index=False))
            lines.append("")

        report = "\n".join(lines)
        report_path = self.output_dir / "summary_report.txt"
        report_path.write_text(report)
        logger.info("Summary report saved: %s", report_path)
        return report

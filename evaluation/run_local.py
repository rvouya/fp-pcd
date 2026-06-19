"""Final evaluation driver: PSNR/SSIM across all filtering stages + classification
metrics, merged into one summary report.

Run this last, after stages 1-5 have produced their outputs:
    output/06_evaluation/psnr/image_quality_metrics.csv
    output/06_evaluation/psnr/image_quality_summary.csv
    output/06_evaluation/classification_metrics/all_scenarios.csv
    output/06_evaluation/classification_metrics/best_scenario.json
    output/06_evaluation/summary_report.txt

If classification hasn't been run yet, the classification section is simply
omitted (image quality metrics are still produced and saved).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import ExperimentEvaluator  # noqa: E402


def main() -> None:
    evaluator = ExperimentEvaluator()

    df_img = evaluator.evaluate_image_quality()
    if df_img.empty:
        print("[evaluation] no image-quality rows -- run stages 1-4 first")
    else:
        print(f"[evaluation] image quality: {len(df_img)} rows across {df_img['stage'].nunique()} stages")

    evaluator.load_classification_results()

    evaluator.export_all()
    report = evaluator.generate_summary_report()
    print("\n" + report)


if __name__ == "__main__":
    main()

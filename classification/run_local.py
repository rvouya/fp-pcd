"""Batch classification driver: train + evaluate ResNet18 across all 5 scenarios.

Reads images already produced by the upstream stages:
    original           <- output/01_preprocessing/normalized
    spatial             <- output/02_spatial_filtering/gaussian
    frequency           <- output/03_frequency_filtering/butterworth_lpf
    spatial_enhanced    <- output/04_roi_enhancement/spatial/clahe
    frequency_enhanced  <- output/04_roi_enhancement/frequency/clahe

For each scenario: trains ResNet18 with early stopping, saves the best
checkpoint + training history under output/05_classification/<scenario>/,
then runs the held-out validation set through evaluation.evaluator so
output/06_evaluation/classification_metrics/ gets a confusion matrix, ROC
curves, and per-scenario metrics row.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_paths import LABELS_CSV  # noqa: E402
from classification.config import ClassificationConfig, TRAINING_SCENARIOS  # noqa: E402
from classification.dataset import XRayDataModule  # noqa: E402
from classification.model import get_model  # noqa: E402
from classification.trainer import Trainer  # noqa: E402
from evaluation.evaluator import ExperimentEvaluator  # noqa: E402


def run_scenario(scenario: str, cfg: ClassificationConfig, evaluator: ExperimentEvaluator) -> None:
    image_dir = cfg.resolve_image_dir()
    if not any(image_dir.glob("*.png")):
        print(f"[{scenario}] no images in {image_dir} -- run the upstream stage first, skip")
        return

    print(f"[{scenario}] images <- {image_dir}")
    dm = XRayDataModule(
        image_dir=image_dir,
        labels_csv=LABELS_CSV,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    train_loader, val_loader = dm.get_loaders()
    cfg.num_classes = dm.num_classes
    print(f"[{scenario}] classes={dm.class_names} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = get_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    trainer = Trainer(model, cfg)

    t0 = time.perf_counter()
    trainer.train(train_loader, val_loader)
    dt = time.perf_counter() - t0
    print(f"[{scenario}] training done in {dt:.1f}s on {trainer.device}")

    results = trainer.evaluate(val_loader)
    print(f"[{scenario}] val accuracy={results['accuracy']:.4f}")

    evaluator.evaluate_classification(
        scenario=scenario,
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        y_prob=results["y_prob"],
        class_names=dm.class_names,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=TRAINING_SCENARIOS, choices=TRAINING_SCENARIOS)
    ap.add_argument("--epochs", type=int, default=20, help="max epochs per scenario, default 20")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience, default 5")
    args = ap.parse_args()

    evaluator = ExperimentEvaluator()

    for scenario in args.scenarios:
        cfg = ClassificationConfig(
            scenario=scenario,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        run_scenario(scenario, cfg, evaluator)

    evaluator.export_all()
    report = evaluator.generate_summary_report()
    print("\n" + report)


if __name__ == "__main__":
    main()

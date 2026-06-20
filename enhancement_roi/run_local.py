"""Batch ROI enhancement driver (CLAHE + morphological ops + histeq + gamma).

Runs on top of stage-2 (spatial/gaussian) and stage-3 (frequency/butterworth_lpf)
outputs to produce the enhancement-based classification scenarios
(04_spatial_enhanced / 05_frequency_enhanced for CLAHE, plus 06-09 for histeq and
gamma). Produces:
    output/04_roi_enhancement/spatial/{clahe,top_hat,opening,closing,histeq,gamma}/
    output/04_roi_enhancement/frequency/{clahe,top_hat,opening,closing,histeq,gamma}/

histeq (global histogram equalization) and gamma (adaptive gamma correction) are
independent alternative enhancement methods, not chained through CLAHE.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enhancement_roi.roi_pipeline import ROIEnhancementPipeline  # noqa: E402

SPATIAL_SRC = PROJECT_ROOT / "output" / "02_spatial_filtering" / "gaussian"
FREQUENCY_SRC = PROJECT_ROOT / "output" / "03_frequency_filtering" / "butterworth_lpf"
ROI_DIR = PROJECT_ROOT / "output" / "04_roi_enhancement"


def run_source(
    label: str,
    src_dir: Path,
    out_dir: Path,
    pipeline: ROIEnhancementPipeline,
    limit: Optional[int],
) -> None:
    """Apply all ROI enhancement modes to every image in src_dir."""
    paths = sorted(src_dir.glob("*.png"))
    if limit:
        paths = paths[:limit]
    if not paths:
        print(f"[roi:{label}] no images in {src_dir} -- run that stage's run_local.py first")
        return

    t0 = time.perf_counter()
    saved = pipeline.batch_process(paths, out_dir)
    dt = time.perf_counter() - t0
    n = len(paths)
    print(
        f"[roi:{label}] {n} images x{len(saved)} modes in {dt:.1f}s "
        f"({dt / max(n, 1) * 1000:.1f} ms/img) -> "
        f"{out_dir}/{{clahe,top_hat,opening,closing,histeq,gamma}}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    ap.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit, default 2.0")
    ap.add_argument(
        "--tile-grid", type=int, default=8, help="square CLAHE tile grid size, default 8"
    )
    ap.add_argument(
        "--morph-kernel", type=int, default=15, help="morphological kernel size, default 15"
    )
    ap.add_argument(
        "--gamma", type=float, default=0.0,
        help="fixed gamma for the gamma mode; <=0 (default) picks adaptive per-image gamma",
    )
    ap.add_argument(
        "--source",
        choices=("spatial", "frequency", "both"),
        default="both",
        help="which filtered stage to enhance, default both",
    )
    args = ap.parse_args()

    pipeline = ROIEnhancementPipeline(
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_grid, args.tile_grid),
        morph_kernel_size=args.morph_kernel,
        gamma=args.gamma,
    )

    if args.source in ("spatial", "both"):
        run_source("spatial", SPATIAL_SRC, ROI_DIR / "spatial", pipeline, args.limit)
    if args.source in ("frequency", "both"):
        run_source("frequency", FREQUENCY_SRC, ROI_DIR / "frequency", pipeline, args.limit)


if __name__ == "__main__":
    main()

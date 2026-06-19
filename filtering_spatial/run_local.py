import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from filtering_spatial.spatial_filter import SpatialFilter  # noqa: E402
from pipeline_paths import (  # noqa: E402
    CORRUPTED_DIR,
    NORMALIZED_DIR,
    ensure_normalized,
    list_image_paths,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ----------------------------------------------------------------------
# Paths (kept in sync with classification/config.py resolve_image_dir())
# ----------------------------------------------------------------------
SPATIAL_DIR = PROJECT_ROOT / "output" / "02_spatial_filtering"
GAUSSIAN_DIR = SPATIAL_DIR / "gaussian"
UNSHARP_DIR = SPATIAL_DIR / "unsharp"


def step_spatial(
    kernel_size: int,
    sigma: float,
    radius: float,
    amount: float,
    threshold: int,
) -> None:
    norm_paths = sorted(NORMALIZED_DIR.glob("*.png"))
    if not norm_paths:
        raise SystemExit(
            f"No normalized images in {NORMALIZED_DIR}. Run without --skip-preprocess."
        )
    sf = SpatialFilter()
    print(
        f"[spatial] gaussian(k={kernel_size}, sigma={sigma}) | "
        f"unsharp(radius={radius}, amount={amount}, threshold={threshold})"
    )

    t0 = time.perf_counter()
    GAUSSIAN_DIR.mkdir(parents=True, exist_ok=True)
    UNSHARP_DIR.mkdir(parents=True, exist_ok=True)
    for src in tqdm(norm_paths, desc="gaussian+unsharp", unit="img"):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        g = sf.apply_gaussian(img, kernel_size=kernel_size, sigma=sigma)
        SpatialFilter._save(g, GAUSSIAN_DIR / src.name)
        u = sf.apply_unsharp(img, radius=radius, amount=amount, threshold=threshold)
        SpatialFilter._save(u, UNSHARP_DIR / src.name)
    dt = time.perf_counter() - t0
    print(
        f"[spatial] {len(norm_paths)} images x2 filters in {dt:.1f}s "
        f"({dt / max(len(norm_paths), 1) * 1000:.1f} ms/img)"
    )
    print(f"[spatial] gaussian -> {GAUSSIAN_DIR}")
    print(f"[spatial] unsharp  -> {UNSHARP_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    ap.add_argument("--size", type=int, default=224, help="resize target (square), default 224")
    ap.add_argument("--norm", choices=("minmax", "zscore"), default="minmax")
    ap.add_argument("--force", action="store_true", help="re-run preprocessing even if outputs exist")
    ap.add_argument("--skip-preprocess", action="store_true", help="use existing normalized images")

    g = ap.add_argument_group("gaussian LPF")
    g.add_argument("--kernel-size", type=int, default=5, help="Gaussian kernel size (odd), default 5")
    g.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma, default 1.0")

    u = ap.add_argument_group("unsharp masking")
    u.add_argument("--radius", type=float, default=1.0, help="blur radius, default 1.0")
    u.add_argument("--amount", type=float, default=1.0, help="sharpening strength, default 1.0")
    u.add_argument("--threshold", type=int, default=0, help="min pixel diff to sharpen, default 0")

    args = ap.parse_args()

    print(f"corrupted source : {CORRUPTED_DIR}")
    paths = list_image_paths(CORRUPTED_DIR, args.limit)
    print(f"images to process: {len(paths)}")

    if not args.skip_preprocess:
        ensure_normalized(paths, NORMALIZED_DIR, args.size, args.norm, args.force, desc="preprocess")
    step_spatial(args.kernel_size, args.sigma, args.radius, args.amount, args.threshold)


if __name__ == "__main__":
    main()

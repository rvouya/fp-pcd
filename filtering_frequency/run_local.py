"""Batch frequency-domain filtering driver (preprocess -> FFT/Butterworth/IFFT).

Mirrors filtering_spatial/run_local.py. Produces:
    output/03_frequency_filtering/butterworth_lpf/   (consumed by classification)
    output/03_frequency_filtering/butterworth_hpf/
    output/03_frequency_filtering/fft/               (log magnitude spectra)
    output/03_frequency_filtering/inverse_fft/        (IFFT of LPF result)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocessor import Preprocessor  # noqa: E402
from filtering_frequency.frequency_filter import FrequencyFilter  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ----------------------------------------------------------------------
# Paths (kept in sync with classification/config.py resolve_image_dir())
# ----------------------------------------------------------------------
CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted"
LABELS_CSV = PROJECT_ROOT / "data" / "balanced_2500.csv"

NORMALIZED_DIR = PROJECT_ROOT / "output" / "01_preprocessing" / "normalized"
FREQ_DIR = PROJECT_ROOT / "output" / "03_frequency_filtering"
LPF_DIR = FREQ_DIR / "butterworth_lpf"
HPF_DIR = FREQ_DIR / "butterworth_hpf"
FFT_DIR = FREQ_DIR / "fft"
IFFT_DIR = FREQ_DIR / "inverse_fft"


def list_image_paths(limit: int | None) -> list[Path]:
    """Resolve corrupted image paths in the CSV's order."""
    df = pd.read_csv(LABELS_CSV)
    names = df["Image Index"].tolist()
    paths = [CORRUPTED_DIR / n for n in names if (CORRUPTED_DIR / n).exists()]
    missing = len(names) - len(paths)
    if missing:
        logging.warning("%d images listed in CSV not found on disk", missing)
    return paths[:limit] if limit else paths


def step_preprocess(paths: list[Path], size: int, norm: str, force: bool) -> None:
    pre = Preprocessor()
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    todo = paths if force else [p for p in paths if not (NORMALIZED_DIR / p.name).exists()]
    if not todo:
        print(f"[preprocess] {len(paths)} images already normalized -> skip")
        return

    t0 = time.perf_counter()
    failed = 0
    for src in tqdm(todo, desc=f"preprocess {size}px/{norm}", unit="img"):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed += 1
            continue
        out = pre.process(img, target_size=(size, size), norm_method=norm)
        pre.save(out, NORMALIZED_DIR / src.name)
    dt = time.perf_counter() - t0
    print(
        f"[preprocess] {len(todo) - failed}/{len(todo)} saved in {dt:.1f}s "
        f"({dt / max(len(todo), 1) * 1000:.1f} ms/img), failed={failed}"
    )


def step_frequency(cutoff: float, order: int) -> None:
    norm_paths = sorted(NORMALIZED_DIR.glob("*.png"))
    if not norm_paths:
        raise SystemExit(
            f"No normalized images in {NORMALIZED_DIR}. Run without --skip-preprocess."
        )
    ff = FrequencyFilter(cutoff=cutoff, order=order)
    print(f"[frequency] butterworth(cutoff={cutoff}, order={order}) | LPF + HPF + FFT + IFFT")

    for d in (LPF_DIR, HPF_DIR, FFT_DIR, IFFT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    failed = 0
    for src in tqdm(norm_paths, desc="frequency filtering", unit="img"):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed += 1
            continue
        res = ff.full_pipeline(img, cutoff=cutoff, order=order)
        cv2.imwrite(str(LPF_DIR / src.name), res["lpf"])
        cv2.imwrite(str(HPF_DIR / src.name), res["hpf"])
        cv2.imwrite(str(FFT_DIR / src.name), res["fft_spectrum"])
        cv2.imwrite(str(IFFT_DIR / src.name), res["inverse_fft_lpf"])
    dt = time.perf_counter() - t0
    print(
        f"[frequency] {len(norm_paths) - failed}/{len(norm_paths)} images x4 outputs "
        f"in {dt:.1f}s ({dt / max(len(norm_paths), 1) * 1000:.1f} ms/img), failed={failed}"
    )
    print(f"[frequency] lpf -> {LPF_DIR}")
    print(f"[frequency] hpf -> {HPF_DIR}")
    print(f"[frequency] fft -> {FFT_DIR}")
    print(f"[frequency] ifft-> {IFFT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    ap.add_argument("--size", type=int, default=224, help="resize target (square), default 224")
    ap.add_argument("--norm", choices=("minmax", "zscore"), default="minmax")
    ap.add_argument("--force", action="store_true", help="re-run preprocessing even if outputs exist")
    ap.add_argument("--skip-preprocess", action="store_true", help="use existing normalized images")

    b = ap.add_argument_group("butterworth")
    b.add_argument("--cutoff", type=float, default=30.0, help="cutoff frequency D0 (pixels), default 30")
    b.add_argument("--order", type=int, default=2, help="filter order n, default 2")

    args = ap.parse_args()

    print(f"corrupted source : {CORRUPTED_DIR}")
    paths = list_image_paths(args.limit)
    print(f"images to process: {len(paths)}")

    if not args.skip_preprocess:
        step_preprocess(paths, args.size, args.norm, args.force)
    step_frequency(args.cutoff, args.order)


if __name__ == "__main__":
    main()

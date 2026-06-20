"""Butterworth parameter optimization scored by PSNR/SSIM vs groundtruth.

Mirrors filtering_spatial/optimize_spatial.py. Sweeps Butterworth LPF
(cutoff x order), picks the best by average PSNR/SSIM rank, applies it to the
full set, and writes:
    output/03_frequency_filtering/butterworth_lpf/        (best LPF, full set)
    output/03_frequency_filtering/butterworth_lpf_sweep.csv
    output/03_frequency_filtering/butterworth_lpf_best_metrics.csv
    output/03_frequency_filtering/best_params.json

HPF is not scored against groundtruth (high-pass keeps edges, so PSNR/SSIM
vs the clean image is meaningless). It is batch-produced at the best LPF
params for the report only.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocessor import Preprocessor  # noqa: E402
from filtering_frequency.butterworth import ButterworthFilter  # noqa: E402
from evaluation.image_metrics import compute_psnr, compute_ssim  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ----------------------------------------------------------------------
# Paths (output folders match classification/config.py resolve_image_dir())
# ----------------------------------------------------------------------
CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted"
GROUNDTRUTH_DIR = PROJECT_ROOT / "data" / "groundtruth"
LABELS_CSV = PROJECT_ROOT / "data" / "balanced_2500.csv"

NORMALIZED_DIR = PROJECT_ROOT / "output" / "01_preprocessing" / "normalized"
GT_NORM_DIR = PROJECT_ROOT / "output" / "01_preprocessing" / "groundtruth_normalized"
FREQ_DIR = PROJECT_ROOT / "output" / "03_frequency_filtering"
LPF_DIR = FREQ_DIR / "butterworth_lpf"
HPF_DIR = FREQ_DIR / "butterworth_hpf"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_list(text: str, cast) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def list_names(limit: int | None) -> List[str]:
    """Image filenames present in BOTH corrupted and groundtruth, CSV order."""
    df = pd.read_csv(LABELS_CSV)
    names = [
        n for n in df["Image Index"].tolist()
        if (CORRUPTED_DIR / n).exists() and (GROUNDTRUTH_DIR / n).exists()
    ]
    return names[:limit] if limit else names


def preprocess_set(names, src_dir, dst_dir, size, norm, force, desc):
    """Resize + normalize every name from src_dir into dst_dir (uint8 PNG)."""
    pre = Preprocessor()
    dst_dir.mkdir(parents=True, exist_ok=True)
    todo = names if force else [n for n in names if not (dst_dir / n).exists()]
    if not todo:
        print(f"[{desc}] {len(names)} already done -> skip")
        return
    for n in tqdm(todo, desc=desc, unit="img"):
        img = cv2.imread(str(src_dir / n), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out = pre.process(img, target_size=(size, size), norm_method=norm)
        pre.save(out, dst_dir / n)


def load_gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def detail_ratio(output: np.ndarray, reference: np.ndarray) -> float:
    """Laplacian-variance ratio of output vs reference (edge/detail preserved).

    ~1.0 means fine texture/edges are preserved; ->0 means over-smoothed. Used
    to stop the PSNR/SSIM sweep from chasing maximum blur, which destroys the
    fine detail a CNN classifier needs.
    """
    ref_var = float(cv2.Laplacian(reference, cv2.CV_64F).var())
    out_var = float(cv2.Laplacian(output, cv2.CV_64F).var())
    if ref_var <= 0:
        return 0.0
    return out_var / ref_var


def add_combined_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by PSNR, SSIM (higher=better) and detail preservation.

    Detail is ranked by ``abs(detail_ratio - 1.0)`` ascending (closest to 1.0 =
    best). Best overall = lowest average of the three ranks.
    """
    df = df.copy()
    df["rank_psnr"] = df["psnr"].rank(ascending=False, method="min")
    df["rank_ssim"] = df["ssim"].rank(ascending=False, method="min")
    df["detail_dev"] = (df["detail_ratio"] - 1.0).abs()
    df["rank_detail"] = df["detail_dev"].rank(ascending=True, method="min")
    df["rank_avg"] = (df["rank_psnr"] + df["rank_ssim"] + df["rank_detail"]) / 3.0
    return df.sort_values("rank_avg").reset_index(drop=True)


# ----------------------------------------------------------------------
# Sweep (scored on a subset held in memory)
# ----------------------------------------------------------------------
def sweep_butterworth(sweep_names, cutoffs, orders, filter_type="lpf") -> pd.DataFrame:
    inputs = [load_gray(NORMALIZED_DIR / n) for n in sweep_names]
    refs = [load_gray(GT_NORM_DIR / n) for n in sweep_names]
    rows = []
    combos = [(d0, n) for d0 in cutoffs for n in orders]
    for d0, n in tqdm(combos, desc=f"sweep butterworth {filter_type}", unit="combo"):
        ps, ss, dr = [], [], []
        for inp, ref in zip(inputs, refs):
            out = ButterworthFilter.apply_with_params(inp, d0, n, filter_type)
            ps.append(compute_psnr(ref, out))
            ss.append(compute_ssim(ref, out))
            dr.append(detail_ratio(out, ref))
        rows.append({"cutoff": d0, "order": n,
                     "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                     "detail_ratio": float(np.mean(dr))})
    return add_combined_rank(pd.DataFrame(rows))


# ----------------------------------------------------------------------
# Apply best params to the full set + collect per-image metrics
# ----------------------------------------------------------------------
def apply_best(names, cutoff, order, filter_type, out_dir, score) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    desc = f"apply {filter_type} best"
    for n in tqdm(names, desc=desc, unit="img"):
        inp = load_gray(NORMALIZED_DIR / n)
        if inp is None:
            continue
        out = ButterworthFilter.apply_with_params(inp, cutoff, order, filter_type)
        cv2.imwrite(str(out_dir / n), out)
        if score:
            ref = load_gray(GT_NORM_DIR / n)
            if ref is None:
                continue
            rows.append({"image_name": n, "psnr": compute_psnr(ref, out),
                         "ssim": compute_ssim(ref, out)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total images processed (quick end-to-end test)")
    ap.add_argument("--sweep-size", type=int, default=200,
                    help="images used to score each param combo, default 200")
    ap.add_argument("--size", type=int, default=224, help="preprocess resize, default 224")
    ap.add_argument("--norm", choices=("minmax", "zscore"), default="minmax")
    ap.add_argument("--force", action="store_true", help="redo preprocessing")
    ap.add_argument("--seed", type=int, default=42, help="sweep subset sampling seed")

    ap.add_argument("--cutoffs", default="30,50,80,110", help="Butterworth cutoff D0 (csv)")
    ap.add_argument("--orders", default="1,2,4", help="Butterworth orders (csv)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    names = list_names(args.limit)
    print(f"images available : {len(names)}")
    if not names:
        raise SystemExit("No images present in BOTH corrupted and groundtruth.")

    # 1-2. preprocess inputs + references identically
    preprocess_set(names, CORRUPTED_DIR, NORMALIZED_DIR, args.size, args.norm,
                   args.force, "preprocess corrupted")
    preprocess_set(names, GROUNDTRUTH_DIR, GT_NORM_DIR, args.size, args.norm,
                   args.force, "preprocess groundtruth")

    # subset for the sweep
    rng = np.random.default_rng(args.seed)
    n_sweep = min(args.sweep_size, len(names))
    sweep_names = list(rng.choice(names, size=n_sweep, replace=False))
    print(f"sweep subset     : {n_sweep} images (seed={args.seed})")

    FREQ_DIR.mkdir(parents=True, exist_ok=True)

    # 3-4. Butterworth LPF sweep + pick
    lpf_df = sweep_butterworth(sweep_names, parse_list(args.cutoffs, float),
                               parse_list(args.orders, int), "lpf")
    lpf_df.to_csv(FREQ_DIR / "butterworth_lpf_sweep.csv", index=False)
    lpf_best = lpf_df.iloc[0]
    print(f"\nBEST Butterworth LPF: cutoff={lpf_best.cutoff} order={int(lpf_best.order)} "
          f"| PSNR={lpf_best.psnr:.2f} SSIM={lpf_best.ssim:.4f} (subset)")

    # 5. apply best LPF to ALL images + full-set metrics
    lpf_metrics = apply_best(names, float(lpf_best.cutoff), int(lpf_best.order),
                             "lpf", LPF_DIR, score=True)
    lpf_metrics.to_csv(FREQ_DIR / "butterworth_lpf_best_metrics.csv", index=False)

    # HPF batch (report only, unscored) at the same params
    apply_best(names, float(lpf_best.cutoff), int(lpf_best.order),
               "hpf", HPF_DIR, score=False)

    # 6. best_params.json
    best = {
        "sweep_size": n_sweep,
        "preprocess": {"size": args.size, "norm": args.norm},
        "selection": "average rank of PSNR, SSIM and detail-preservation (lower is better)",
        "butterworth_lpf": {
            "params": {"cutoff": float(lpf_best.cutoff), "order": int(lpf_best.order)},
            "sweep_psnr": float(lpf_best.psnr), "sweep_ssim": float(lpf_best.ssim),
            "sweep_detail_ratio": float(lpf_best.detail_ratio),
            "fullset_psnr": float(lpf_metrics.psnr.mean()) if not lpf_metrics.empty else None,
            "fullset_ssim": float(lpf_metrics.ssim.mean()) if not lpf_metrics.empty else None,
            "output_dir": str(LPF_DIR.relative_to(PROJECT_ROOT)),
        },
        "butterworth_hpf": {
            "params": {"cutoff": float(lpf_best.cutoff), "order": int(lpf_best.order)},
            "scored": False,
            "note": "produced for report only; not scored vs groundtruth",
            "output_dir": str(HPF_DIR.relative_to(PROJECT_ROOT)),
        },
    }
    (FREQ_DIR / "best_params.json").write_text(json.dumps(best, indent=2))

    dt = time.perf_counter() - t_start
    print("\n================ SUMMARY ================")
    print(f"Butterworth LPF best : {best['butterworth_lpf']['params']} | "
          f"full-set PSNR={best['butterworth_lpf']['fullset_psnr']:.2f} "
          f"SSIM={best['butterworth_lpf']['fullset_ssim']:.4f}")
    print(f"Saved (classified)   : {LPF_DIR}")
    print(f"HPF (report only)    : {HPF_DIR}")
    print(f"Reports              : {FREQ_DIR}/*.csv , best_params.json")
    print(f"Total time           : {dt:.1f}s")


if __name__ == "__main__":
    main()

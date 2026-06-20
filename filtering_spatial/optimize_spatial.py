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
from filtering_spatial.gaussian_lpf import GaussianLPF  # noqa: E402
from filtering_spatial.unsharp_masking import UnsharpMasking  # noqa: E402
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
SPATIAL_DIR = PROJECT_ROOT / "output" / "02_spatial_filtering"
GAUSSIAN_DIR = SPATIAL_DIR / "gaussian"
UNSHARP_DIR = SPATIAL_DIR / "unsharp"


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
# Sweeps (scored on a subset held in memory)
# ----------------------------------------------------------------------
def sweep_gaussian(sweep_names, kernels, sigmas) -> pd.DataFrame:
    inputs = [load_gray(NORMALIZED_DIR / n) for n in sweep_names]
    refs = [load_gray(GT_NORM_DIR / n) for n in sweep_names]
    glpf = GaussianLPF()
    rows = []
    combos = [(k, s) for k in kernels for s in sigmas]
    for k, s in tqdm(combos, desc="sweep gaussian", unit="combo"):
        ps, ss, dr = [], [], []
        for inp, ref in zip(inputs, refs):
            out = glpf.apply_with_params(inp, k, s)
            ps.append(compute_psnr(ref, out))
            ss.append(compute_ssim(ref, out))
            dr.append(detail_ratio(out, ref))
        rows.append({"kernel_size": k, "sigma": s,
                     "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                     "detail_ratio": float(np.mean(dr))})
    return add_combined_rank(pd.DataFrame(rows))


def sweep_unsharp(sweep_names, radii, amounts, thresholds) -> pd.DataFrame:
    inputs = [load_gray(NORMALIZED_DIR / n) for n in sweep_names]
    refs = [load_gray(GT_NORM_DIR / n) for n in sweep_names]
    um = UnsharpMasking()
    rows = []
    combos = [(r, a, t) for r in radii for a in amounts for t in thresholds]
    for r, a, t in tqdm(combos, desc="sweep unsharp", unit="combo"):
        ps, ss, dr = [], [], []
        for inp, ref in zip(inputs, refs):
            out = um.apply_with_params(inp, r, a, t)
            ps.append(compute_psnr(ref, out))
            ss.append(compute_ssim(ref, out))
            dr.append(detail_ratio(out, ref))
        rows.append({"radius": r, "amount": a, "threshold": t,
                     "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                     "detail_ratio": float(np.mean(dr))})
    return add_combined_rank(pd.DataFrame(rows))


# ----------------------------------------------------------------------
# Apply best params to the full set + collect per-image metrics
# ----------------------------------------------------------------------
def apply_gaussian_best(names, k, s, out_dir) -> pd.DataFrame:
    glpf = GaussianLPF()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in tqdm(names, desc="apply gaussian best", unit="img"):
        inp = load_gray(NORMALIZED_DIR / n)
        ref = load_gray(GT_NORM_DIR / n)
        if inp is None or ref is None:
            continue
        out = glpf.apply_with_params(inp, k, s)
        cv2.imwrite(str(out_dir / n), out)
        rows.append({"image_name": n, "psnr": compute_psnr(ref, out),
                     "ssim": compute_ssim(ref, out)})
    return pd.DataFrame(rows)


def apply_unsharp_best(names, r, a, t, out_dir) -> pd.DataFrame:
    um = UnsharpMasking()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in tqdm(names, desc="apply unsharp best", unit="img"):
        inp = load_gray(NORMALIZED_DIR / n)
        ref = load_gray(GT_NORM_DIR / n)
        if inp is None or ref is None:
            continue
        out = um.apply_with_params(inp, r, a, t)
        cv2.imwrite(str(out_dir / n), out)
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

    ap.add_argument("--kernels", default="3,5,7,9", help="Gaussian kernel sizes (csv)")
    ap.add_argument("--sigmas", default="0.5,1.0,1.5,2.0", help="Gaussian sigmas (csv)")
    ap.add_argument("--radii", default="0.5,1.0,2.0", help="Unsharp radii (csv)")
    ap.add_argument("--amounts", default="0.5,1.0,1.5,2.0", help="Unsharp amounts (csv)")
    ap.add_argument("--thresholds", default="0,5,10", help="Unsharp thresholds (csv)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    names = list_names(args.limit)
    print(f"images available : {len(names)}")

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

    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

    # 3-4. Gaussian sweep + pick
    g_df = sweep_gaussian(sweep_names, parse_list(args.kernels, int),
                          parse_list(args.sigmas, float))
    g_df.to_csv(SPATIAL_DIR / "gaussian_sweep.csv", index=False)
    g_best = g_df.iloc[0]
    print(f"\nBEST Gaussian: k={int(g_best.kernel_size)} sigma={g_best.sigma} "
          f"| PSNR={g_best.psnr:.2f} SSIM={g_best.ssim:.4f} (subset)")

    # Unsharp sweep + pick
    u_df = sweep_unsharp(sweep_names, parse_list(args.radii, float),
                         parse_list(args.amounts, float),
                         parse_list(args.thresholds, int))
    u_df.to_csv(SPATIAL_DIR / "unsharp_sweep.csv", index=False)
    u_best = u_df.iloc[0]
    print(f"BEST Unsharp : radius={u_best.radius} amount={u_best.amount} "
          f"threshold={int(u_best.threshold)} | PSNR={u_best.psnr:.2f} "
          f"SSIM={u_best.ssim:.4f} (subset)")

    # 5. apply best to ALL images + full-set metrics
    g_metrics = apply_gaussian_best(names, int(g_best.kernel_size), float(g_best.sigma),
                                    GAUSSIAN_DIR)
    u_metrics = apply_unsharp_best(names, float(u_best.radius), float(u_best.amount),
                                   int(u_best.threshold), UNSHARP_DIR)
    g_metrics.to_csv(SPATIAL_DIR / "gaussian_best_metrics.csv", index=False)
    u_metrics.to_csv(SPATIAL_DIR / "unsharp_best_metrics.csv", index=False)

    # 6. best_params.json
    best = {
        "sweep_size": n_sweep,
        "preprocess": {"size": args.size, "norm": args.norm},
        "selection": "average rank of PSNR, SSIM and detail-preservation (lower is better)",
        "gaussian": {
            "params": {"kernel_size": int(g_best.kernel_size), "sigma": float(g_best.sigma)},
            "sweep_psnr": float(g_best.psnr), "sweep_ssim": float(g_best.ssim),
            "sweep_detail_ratio": float(g_best.detail_ratio),
            "fullset_psnr": float(g_metrics.psnr.mean()),
            "fullset_ssim": float(g_metrics.ssim.mean()),
            "output_dir": str(GAUSSIAN_DIR.relative_to(PROJECT_ROOT)),
        },
        "unsharp": {
            "params": {"radius": float(u_best.radius), "amount": float(u_best.amount),
                       "threshold": int(u_best.threshold)},
            "sweep_psnr": float(u_best.psnr), "sweep_ssim": float(u_best.ssim),
            "sweep_detail_ratio": float(u_best.detail_ratio),
            "fullset_psnr": float(u_metrics.psnr.mean()),
            "fullset_ssim": float(u_metrics.ssim.mean()),
            "output_dir": str(UNSHARP_DIR.relative_to(PROJECT_ROOT)),
        },
    }
    (SPATIAL_DIR / "best_params.json").write_text(json.dumps(best, indent=2))

    dt = time.perf_counter() - t_start
    print("\n================ SUMMARY ================")
    print(f"Gaussian best  : {best['gaussian']['params']} | "
          f"full-set PSNR={best['gaussian']['fullset_psnr']:.2f} "
          f"SSIM={best['gaussian']['fullset_ssim']:.4f}")
    print(f"Unsharp best   : {best['unsharp']['params']} | "
          f"full-set PSNR={best['unsharp']['fullset_psnr']:.2f} "
          f"SSIM={best['unsharp']['fullset_ssim']:.4f}")
    print(f"Saved          : {GAUSSIAN_DIR} , {UNSHARP_DIR}")
    print(f"Reports        : {SPATIAL_DIR}/*.csv , best_params.json")
    print(f"Total time     : {dt:.1f}s")


if __name__ == "__main__":
    main()

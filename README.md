# Final Project — Pengolahan Citra Digital (PCD)

Chest X-Ray image processing pipeline comparing spatial filtering, frequency filtering, ROI enhancement, and classification performance.

---

## 1. Project Overview

End-to-end pipeline for:

1. **Preprocessing** — resize + MinMax/Z-score normalization
2. **Spatial Filtering** — Gaussian LPF, Unsharp Masking
3. **Frequency Filtering** — 2D FFT, Butterworth LPF/HPF, Inverse FFT
4. **ROI Enhancement** — CLAHE, Top-Hat, Opening, Closing
5. **Classification** — ResNet-18 trained on 5 scenarios (original / spatial / frequency / spatial_enhanced / frequency_enhanced)
6. **Evaluation** — PSNR, SSIM, Accuracy, Precision, Recall, F1, ROC AUC

Every stage **persists its processed images to `output/`** so the next stage (and the final evaluation) reads from disk instead of recomputing anything.

---

## 2. Dataset Structure

```
data/
├── corrupted/              ← 2500 degraded chest X-ray PNGs (pipeline input)
├── groundtruth/            ← 2500 clean reference PNGs, same filenames as corrupted/
└── balanced_2500.csv       ← labels ("Image Index", "Finding Labels", ...)
```

`data/` is **not** committed to git (see `.gitignore`). Classes present in `balanced_2500.csv`: `Atelectasis`, `Effusion`, `Infiltration`, `No Finding`, `Nodule`, `Pneumothorax`.

`groundtruth/` is the clean reference used only to score PSNR/SSIM and pick the best filter parameters — the actual pipeline always processes `corrupted/`.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional but speeds up the classification stage considerably (CUDA auto-detected).

---

## 4. Folder Structure

```
project_root/
├── data/                        ← NOT committed to git
├── pipeline_paths.py            ← shared paths/helpers used by every run_local.py
├── preprocessing/
│   ├── dataset_loader.py
│   ├── preprocessor.py
│   └── preprocessing_experiments.ipynb
├── filtering_spatial/
│   ├── gaussian_lpf.py
│   ├── unsharp_masking.py
│   ├── spatial_filter.py
│   ├── run_local.py             ← batch driver, fixed params
│   ├── optimize_spatial.py      ← sweeps params vs groundtruth, applies best to all images
│   └── spatial_experiments.ipynb
├── filtering_frequency/
│   ├── fft_transform.py
│   ├── butterworth.py
│   ├── frequency_filter.py
│   ├── run_local.py
│   ├── optimize_frequency.py
│   └── frequency_experiments.ipynb
├── enhancement_roi/
│   ├── clahe.py
│   ├── morphological.py
│   ├── roi_pipeline.py
│   ├── run_local.py
│   └── enhancement_experiments.ipynb
├── classification/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── run_local.py
│   └── classification_experiments.ipynb
├── evaluation/
│   ├── image_metrics.py
│   ├── classification_metrics.py
│   ├── evaluator.py
│   ├── run_local.py
│   └── evaluation_experiments.ipynb
├── output/                      ← generated artifacts (see §6); heavy image dirs gitignored
├── requirements.txt
├── .gitignore
└── README.md
```

Each stage ships two kinds of scripts:
- **`run_local.py`** — production batch driver. Reads from the previous stage's `output/` dir, writes to its own, fixed/CLI-provided parameters. This is what you run end-to-end.
- **`optimize_*.py`** (spatial, frequency only) — sweeps parameters against `groundtruth/`, picks the best by PSNR/SSIM rank, and applies that best configuration to the full dataset. Run this *instead of* `run_local.py` for those two stages if you want tuned parameters.
- **`*_experiments.ipynb`** — exploratory notebook for visual comparisons and parameter sweeps shown as figures (not required for the pipeline to run).

---

## 5. Pipeline Diagram

```
data/corrupted/  (2500 images)
      │
      ▼
[1. Preprocessing]  resize 224×224 + MinMax normalize
      │  -> output/01_preprocessing/normalized/
      │
      ├──▶ [2. Spatial Filtering]
      │         ├── Gaussian LPF      -> output/02_spatial_filtering/gaussian/
      │         └── Unsharp Masking   -> output/02_spatial_filtering/unsharp/
      │
      └──▶ [3. Frequency Filtering]
                ├── 2D FFT spectrum       -> output/03_frequency_filtering/fft/
                ├── Butterworth LPF       -> output/03_frequency_filtering/butterworth_lpf/
                ├── Butterworth HPF       -> output/03_frequency_filtering/butterworth_hpf/
                └── Inverse FFT (of LPF)  -> output/03_frequency_filtering/inverse_fft/

[2. gaussian/] ──▶ [4. ROI Enhancement: spatial]   -> output/04_roi_enhancement/spatial/{clahe,top_hat,opening,closing}/
[3. butterworth_lpf/] ──▶ [4. ROI Enhancement: frequency] -> output/04_roi_enhancement/frequency/{clahe,top_hat,opening,closing}/

      │
      ▼
[5. Classification — ResNet-18]   5 scenarios, each reading a different output/ dir above
   original | spatial | frequency | spatial_enhanced | frequency_enhanced
      │  -> output/05_classification/<scenario>/{checkpoints,logs}/
      ▼
[6. Evaluation]
   PSNR/SSIM (vs output/01_preprocessing/groundtruth_normalized/) + classification metrics
      -> output/06_evaluation/{psnr,classification_metrics}/, summary_report.txt
```

---

## 6. How To Run

Run every command from the project root, in this order. Each step reads the previous step's `output/` and is safe to re-run (it skips images that already exist unless `--force` is passed).

```bash
# 1+3. Frequency filtering — also produces stage-1 preprocessing output
#      (normalized/ and groundtruth_normalized/) as a side effect.
python filtering_frequency/optimize_frequency.py

# 2. Spatial filtering (skips preprocessing, already done above)
python filtering_spatial/optimize_spatial.py

# 3b. Regenerate FFT spectrum + inverse-FFT visualizations at the tuned params
#     reported by optimize_frequency.py (defaults shown match the discovered best)
python filtering_frequency/run_local.py --skip-preprocess --cutoff 30 --order 2

# 4. ROI enhancement on top of both filtered stages
python enhancement_roi/run_local.py

# 5. Classification — trains ResNet-18 for all 5 scenarios
python classification/run_local.py

# 6. Final evaluation — merges image-quality + classification metrics
python evaluation/run_local.py
```

If you'd rather run preprocessing/spatial filtering with fixed parameters instead of the auto-tuned sweep, use the plain drivers:

```bash
python filtering_spatial/run_local.py          # preprocess + gaussian + unsharp, fixed defaults
python filtering_frequency/run_local.py        # preprocess + butterworth lpf/hpf + fft/ifft, fixed defaults
```

Useful flags on every driver: `--limit N` (smoke-test on N images), `--force` (reprocess even if output exists). See `--help` on each script for the rest.

### Discovered best parameters (from the optimize scripts, full 2500-image set)

| Stage | Params | PSNR | SSIM |
|---|---|---|---|
| Butterworth LPF | cutoff=30, order=2 | 24.51 | 0.854 |
| Gaussian LPF | kernel_size=11, sigma=1.0 | 25.40 | 0.858 |
| Unsharp Masking | radius=0.5, amount=1.5, threshold=10 | 25.38 | 0.787 |

Re-running `optimize_spatial.py` / `optimize_frequency.py` regenerates these automatically and writes them to `output/02_spatial_filtering/best_params.json` and `output/03_frequency_filtering/best_params.json`.

---

## 7. Experiment Workflow (notebooks)

Each `*_experiments.ipynb` is for exploration, not production:
- Loads images from the previous stage's `output/` dir
- Runs parameter sweeps and renders comparison figures inline
- Optionally writes a small sample of processed images — the full-dataset batch processing is the job of `run_local.py` / `optimize_*.py`, not the notebooks

Open with:

```bash
jupyter notebook
```

---

## 8. Evaluation Workflow

`evaluation/run_local.py` (or `evaluation/evaluation_experiments.ipynb` for the visual version):

1. Computes PSNR/SSIM for every filtering/enhancement stage against `output/01_preprocessing/groundtruth_normalized/` (the resized+normalized clean reference, never the corrupted input — so the numbers reflect actual restoration quality).
2. Loads classification results (`output/06_evaluation/classification_metrics/all_scenarios.csv`) written by `classification/run_local.py`, if present.
3. Exports:
   - `output/06_evaluation/psnr/image_quality_metrics.csv` (per-image)
   - `output/06_evaluation/psnr/image_quality_summary.csv` (per-stage mean/std)
   - `output/06_evaluation/classification_metrics/all_scenarios.csv`, `best_scenario.json`, confusion matrices, ROC curves
   - `output/06_evaluation/summary_report.txt` (human-readable, combines both)

Run it last, after stage 5, to get the combined report. It also works if classification hasn't run yet — the classification section is simply omitted.

---

## 9. Team Responsibilities

| Person | Module | Responsibilities |
|--------|--------|-------------------|
| Person 1 | `preprocessing/` | Dataset loader, resize, normalization |
| Person 2 | `filtering_spatial/` | Gaussian LPF, Unsharp Masking |
| Person 3 | `filtering_frequency/` | FFT, Butterworth LPF/HPF, IFFT |
| Person 4 | `enhancement_roi/` | CLAHE, Top-Hat, Opening, Closing |
| Person 5 | `classification/` | ResNet-18 training, 5 scenarios |
| All | `evaluation/` | PSNR, SSIM, classification metrics |

---

## 10. Reproducibility Notes

- Python 3.10+
- `seed=42` everywhere a split or sweep subset is drawn (`ClassificationConfig.seed`, `optimize_*.py --seed`)
- Images are saved at every stage so any later stage can be re-run independently without recomputing earlier ones
- All outputs are deterministic given the same input images and parameters

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

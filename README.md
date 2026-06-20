# Final Project — Pengolahan Citra Digital (PCD)

Chest X-Ray image processing pipeline comparing spatial filtering, frequency filtering, ROI enhancement, and classification performance against a clean-groundtruth ceiling baseline.

---

## 1. Project Description

**Problem.** Chest X-rays in the dataset are corrupted (noise/degradation). The goal is to restore them closer to their clean groundtruth counterpart using classical image-processing filters, then check whether that restoration actually helps a downstream deep-learning classifier.

**Dataset.** 2500 chest X-ray images, each with a corrupted version and a clean groundtruth version (same filename), labeled with one finding from `balanced_2500.csv` ("Finding Labels"): `Atelectasis`, `Effusion`, `Infiltration`, `No Finding`, `Nodule`, `Pneumothorax` (6 classes, NIH ChestX-ray14 subset).

**Methods.**
- Spatial filtering: Gaussian LPF, Unsharp Masking
- Frequency filtering: 2D FFT, Butterworth LPF/HPF, Inverse FFT
- ROI enhancement: CLAHE (+ morphology), Global Histogram Equalization, Adaptive Gamma Correction
- Parameter selection: swept against groundtruth, ranked by PSNR + SSIM + a Laplacian-variance detail-preservation ratio (so the sweep can't win by just blurring everything)

**Classification.** ResNet-18 (ImageNet-pretrained, fc head replaced for 6 classes), trained independently per scenario — one scenario per pipeline-output directory — using one shared label mapping and train/val split across all scenarios so accuracy is comparable apples-to-apples.

**Results (full 2500-image run, see §8 for the table).** Accuracy sits in a narrow 32–37% band across every scenario, including the `00_groundtruth` ceiling (33.8%). The best classifier was `03_frequency` (Butterworth LPF, 37.4%), narrowly ahead of the unfiltered `01_original` baseline (35.6%).

**Conclusion.** None of the classical filters produce a clear, decisive accuracy gain over the unfiltered baseline — gains are within a few points of each other and of the clean-image ceiling. This suggests the bottleneck for this task is not corruption/noise in the input image but something else (class overlap in chest-finding patterns, dataset size, or model capacity), since training on perfectly clean groundtruth images does no better than training on corrupted ones.

---

## 2. Dataset Structure

```
data/
├── corrupted/              ← 2500 degraded chest X-ray PNGs (pipeline input)
├── groundtruth/            ← 2500 clean reference PNGs, same filenames as corrupted/
└── balanced_2500.csv       ← labels ("Image Index", "Finding Labels", ...)
```

`data/` is **not** committed to git (see `.gitignore`).

`groundtruth/` is the clean reference used only to score PSNR/SSIM, pick the best filter parameters, and train the `00_groundtruth` ceiling scenario — the filtering stages always process `corrupted/`.

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
├── run_full_pipeline.py         ← runs the entire pipeline end-to-end, streaming output live
├── preprocessing/
│   ├── dataset_loader.py
│   ├── preprocessor.py
│   └── preprocessing_experiments.ipynb
├── filtering_spatial/
│   ├── gaussian_lpf.py
│   ├── unsharp_masking.py
│   ├── spatial_filter.py
│   ├── run_local.py             ← batch driver, fixed params
│   ├── optimize_spatial.py      ← sweeps params vs groundtruth (PSNR+SSIM+detail), applies best to all images
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
│   ├── histogram_eq.py          ← global histogram equalization (independent of CLAHE)
│   ├── gamma_correction.py      ← adaptive/fixed gamma correction (independent of CLAHE)
│   ├── roi_pipeline.py
│   ├── run_local.py
│   └── enhancement_experiments.ipynb
├── classification/
│   ├── config.py                ← TRAINING_SCENARIOS list + scenario -> source dir map
│   ├── dataset.py                ← canonical label/split builder shared by every scenario
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
- **`run_local.py`** — production batch driver. Reads from the previous stage's `output/` dir, writes to its own, fixed/CLI-provided parameters.
- **`optimize_*.py`** (spatial, frequency only) — sweeps parameters against `groundtruth/`, picks the best by combined PSNR/SSIM/detail-preservation rank, and applies that best configuration to the full dataset. Run this *instead of* `run_local.py` for those two stages.
- **`*_experiments.ipynb`** — exploratory notebook for visual comparisons (not required for the pipeline to run).
- **`run_full_pipeline.py`** (root) — runs every stage above in correct order in one command, each subprocess's stdout/stderr streamed live to the terminal exactly as if run individually.

---

## 5. Pipeline Diagram

```
data/corrupted/  (2500 images)        data/groundtruth/  (2500 images)
      │                                       │
      ▼                                       ▼
[1. Preprocessing] resize 224×224 + MinMax normalize (both sets)
      │  -> output/01_preprocessing/normalized/        (corrupted)
      │  -> output/01_preprocessing/groundtruth_normalized/  (clean, used as PSNR/SSIM reference + 00_groundtruth scenario)
      │
      ├──▶ [2. Spatial Filtering]  (optimize_spatial.py)
      │         ├── Gaussian LPF      -> output/02_spatial_filtering/gaussian/      (-> 02_spatial scenario)
      │         └── Unsharp Masking   -> output/02_spatial_filtering/unsharp/
      │
      └──▶ [3. Frequency Filtering]  (optimize_frequency.py)
                ├── 2D FFT spectrum       -> output/03_frequency_filtering/fft/
                ├── Butterworth LPF       -> output/03_frequency_filtering/butterworth_lpf/  (-> 03_frequency scenario)
                ├── Butterworth HPF       -> output/03_frequency_filtering/butterworth_hpf/
                └── Inverse FFT (of LPF)  -> output/03_frequency_filtering/inverse_fft/

[2. gaussian/]        ──▶ [4. ROI Enhancement: spatial]   -> output/04_roi_enhancement/spatial/{clahe,top_hat,opening,closing,histeq,gamma}/
[3. butterworth_lpf/] ──▶ [4. ROI Enhancement: frequency] -> output/04_roi_enhancement/frequency/{clahe,top_hat,opening,closing,histeq,gamma}/

      │
      ▼
[5. Classification — ResNet-18]   10 numbered scenarios, one shared label/split, each reading a different dir above
   00_groundtruth | 01_original | 02_spatial | 03_frequency | 04_spatial_enhanced | 05_frequency_enhanced
   06_spatial_histeq | 07_frequency_histeq | 08_spatial_gamma | 09_frequency_gamma
      │  -> output/05_classification/<scenario>/{checkpoints,logs}/
      ▼
[6. Evaluation]
   PSNR/SSIM (vs output/01_preprocessing/groundtruth_normalized/) + classification metrics, mean±std per stage
      -> output/06_evaluation/{psnr,classification_metrics}/, summary_report.txt
```

---

## 6. How To Run

### 6.1 One command, full pipeline

```bash
python run_full_pipeline.py
```

Runs, in order, on the full dataset: `optimize_spatial.py` → `optimize_frequency.py` → `enhancement_roi/run_local.py` → `classification/run_local.py` (all 10 scenarios) → `evaluation/run_local.py`. Each script's output streams to the terminal live, same as running it by hand; if a stage fails, the run stops there with its exit code.

### 6.2 Stage by stage

```bash
# 2. Spatial filtering (+ stage-1 preprocessing as a side effect)
python filtering_spatial/optimize_spatial.py

# 3. Frequency filtering (+ stage-1 preprocessing if not already done)
python filtering_frequency/optimize_frequency.py

# 4. ROI enhancement on top of both filtered stages
python enhancement_roi/run_local.py

# 5. Classification — trains ResNet-18 for all 10 scenarios
python classification/run_local.py

# 6. Final evaluation — merges image-quality + classification metrics
python evaluation/run_local.py
```

### 6.3 Configuration options per stage

**Spatial filtering** (`filtering_spatial/optimize_spatial.py`)
```bash
python filtering_spatial/optimize_spatial.py --limit 20 --sweep-size 10       # smoke test
python filtering_spatial/optimize_spatial.py --kernels 3,5,7,9 --sigmas 0.5,1.0,1.5,2.0
python filtering_spatial/optimize_spatial.py --radii 0.5,1.0,2.0 --amounts 0.5,1.0,1.5,2.0 --thresholds 0,5,10
python filtering_spatial/optimize_spatial.py --size 224 --norm minmax --seed 42 --force
```
Fixed-parameter alternative (no sweep): `python filtering_spatial/run_local.py`

**Frequency filtering** (`filtering_frequency/optimize_frequency.py`)
```bash
python filtering_frequency/optimize_frequency.py --limit 20 --sweep-size 10   # smoke test
python filtering_frequency/optimize_frequency.py --cutoffs 30,50,80,110 --orders 1,2,4
python filtering_frequency/optimize_frequency.py --size 224 --norm minmax --seed 42 --force
```
Fixed-parameter alternative: `python filtering_frequency/run_local.py --skip-preprocess --cutoff 80 --order 2`

**ROI enhancement** (`enhancement_roi/run_local.py`)
```bash
python enhancement_roi/run_local.py --limit 20 --source spatial    # smoke test, one source only
python enhancement_roi/run_local.py --clip-limit 2.0               # CLAHE clip limit
python enhancement_roi/run_local.py --gamma 1.5                    # fixed gamma instead of adaptive
```

**Classification** (`classification/run_local.py`)
```bash
python classification/run_local.py --scenarios 01_original 02_spatial --epochs 1   # subset / smoke test
python classification/run_local.py --epochs 20 --batch-size 32 --patience 5 --val-split 0.2
```

**Evaluation** (`evaluation/run_local.py`) — no flags, run last:
```bash
python evaluation/run_local.py
```

Every driver also accepts `--force` to reprocess images that already exist on disk.

### 6.4 Classification scenarios

| Scenario               | Source image directory                                |
|------------------------|---------------------------------------------------------|
| `00_groundtruth`       | `output/01_preprocessing/groundtruth_normalized` (ceiling baseline) |
| `01_original`          | `output/01_preprocessing/normalized` (unfiltered baseline) |
| `02_spatial`           | `output/02_spatial_filtering/gaussian`                  |
| `03_frequency`         | `output/03_frequency_filtering/butterworth_lpf`         |
| `04_spatial_enhanced`  | `output/04_roi_enhancement/spatial/clahe`               |
| `05_frequency_enhanced`| `output/04_roi_enhancement/frequency/clahe`             |
| `06_spatial_histeq`    | `output/04_roi_enhancement/spatial/histeq`              |
| `07_frequency_histeq`  | `output/04_roi_enhancement/frequency/histeq`            |
| `08_spatial_gamma`     | `output/04_roi_enhancement/spatial/gamma`               |
| `09_frequency_gamma`   | `output/04_roi_enhancement/frequency/gamma`             |

### 6.5 Discovered best parameters (full 2500-image set)

| Stage | Params | PSNR | SSIM | Detail ratio |
|---|---|---|---|---|
| Gaussian LPF | kernel=3, sigma=0.5 | 25.49 | 0.821 | 0.45 |
| Unsharp Masking | radius=0.5, amount=2.0, threshold=10 | 25.38 | 0.787 | 1.00 |
| Butterworth LPF | cutoff=80, order=2 | 25.23 | 0.821 | 0.58 |

Written automatically to `output/02_spatial_filtering/best_params.json` and `output/03_frequency_filtering/best_params.json` whenever the `optimize_*.py` scripts re-run.

---

## 7. Experiment Workflow (notebooks)

Each `*_experiments.ipynb` is for exploration, not production — loads images from the previous stage's `output/` dir, runs parameter sweeps, and renders comparison figures inline. The full-dataset batch processing is the job of `run_local.py` / `optimize_*.py`, not the notebooks.

```bash
jupyter notebook
```

---

## 8. Results

Full 2500-image run, `output/06_evaluation/summary_report.txt`:

**Image quality (mean ± std PSNR / SSIM vs groundtruth):**

| Stage | PSNR | SSIM |
|---|---|---|
| Gaussian LPF | 25.49 ± 1.87 | 0.821 ± 0.047 |
| Unsharp Masking | 25.38 ± 1.82 | 0.787 ± 0.050 |
| Butterworth LPF | 25.23 ± 1.88 | 0.821 ± 0.046 |
| Butterworth HPF | 4.65 ± 1.20 | 0.037 ± 0.039 |
| ROI CLAHE (spatial/freq) | 20.06 / 19.85 | 0.604 / 0.602 |
| ROI HistEq (spatial/freq) | 19.81 / 19.77 | 0.701 / 0.702 |
| ROI Gamma (spatial/freq) | 22.93 / 22.88 | 0.789 / 0.790 |

**Classification accuracy (ResNet-18, 10 scenarios):**

| Scenario | Accuracy | F1 (weighted) | F1 (macro) |
|---|---|---|---|
| 00_groundtruth | 0.338 | 0.337 | 0.331 |
| 01_original | 0.356 | 0.362 | 0.354 |
| 02_spatial | 0.330 | 0.338 | 0.334 |
| **03_frequency (best)** | **0.374** | **0.379** | **0.371** |
| 04_spatial_enhanced | 0.342 | 0.343 | 0.338 |
| 05_frequency_enhanced | 0.342 | 0.337 | 0.330 |
| 06_spatial_histeq | 0.342 | 0.341 | 0.337 |
| 07_frequency_histeq | 0.332 | 0.333 | 0.327 |
| 08_spatial_gamma | 0.324 | 0.318 | 0.310 |
| 09_frequency_gamma | 0.368 | 0.368 | 0.360 |

Best scenario: `03_frequency` (Butterworth LPF), 37.4% accuracy. See conclusion in §1.

---

## 9. Evaluation Workflow

`evaluation/run_local.py` (or `evaluation/evaluation_experiments.ipynb` for the visual version):

1. Computes PSNR/SSIM for every filtering/enhancement stage against `output/01_preprocessing/groundtruth_normalized/` (resized+normalized clean reference, never the corrupted input).
2. Loads classification results (`output/06_evaluation/classification_metrics/all_scenarios.csv`) written by `classification/run_local.py`, if present.
3. Exports:
   - `output/06_evaluation/psnr/image_quality_metrics.csv` (per-image), `image_quality_summary.csv` (per-stage mean±std)
   - `output/06_evaluation/classification_metrics/all_scenarios.csv`, `best_scenario.json`, confusion matrices, ROC curves
   - `output/06_evaluation/summary_report.txt` (combines both, ordered by pipeline step)

Run it last. It also works if classification hasn't run yet — that section is simply omitted.

---

## 10. Team Responsibilities

| Person | Module | Responsibilities |
|--------|--------|-------------------|
| Person 1 | `preprocessing/` | Dataset loader, resize, normalization |
| Person 2 | `filtering_spatial/` | Gaussian LPF, Unsharp Masking |
| Person 3 | `filtering_frequency/` | FFT, Butterworth LPF/HPF, IFFT |
| Person 4 | `enhancement_roi/` | CLAHE, morphology, histogram equalization, gamma correction |
| Person 5 | `classification/` | ResNet-18 training, 10 scenarios |
| All | `evaluation/` | PSNR, SSIM, classification metrics |

---

## 11. Reproducibility Notes

- Python 3.10+
- `seed=42` everywhere a split or sweep subset is drawn (`ClassificationConfig.seed`, `optimize_*.py --seed`)
- Label mapping + train/val split for classification are computed **once** from the full CSV and shared across all 10 scenarios — no per-scenario split drift
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

# Final Project — Pengolahan Citra Digital (PCD)

Chest X-Ray image processing pipeline comparing spatial filtering, frequency filtering, ROI enhancement, and classification performance.

---

## 1. Project Overview

This project builds an end-to-end pipeline for:

1. **Preprocessing** — resize and normalization
2. **Spatial Filtering** — Gaussian LPF, Unsharp Masking
3. **Frequency Filtering** — 2D FFT, Butterworth LPF/HPF, Inverse FFT
4. **ROI Enhancement** — CLAHE, Top-Hat, Opening, Closing
5. **Classification** — ResNet-18 trained on 5 pipeline scenarios
6. **Evaluation** — PSNR, SSIM, Accuracy, Precision, Recall, F1, ROC AUC

---

## 2. Dataset Structure

```
dataset/
├── original/
│   ├── fp/
│   │   └── balanced/          ← original PNG images
│   └── balanced_prompts_fixed.csv
└── corrupted/
    ├── balanced_2500.csv
    ├── imbalanced_2500.csv
    └── combined/combined/
        ├── balanced/balanced/  ← corrupted balanced images
        └── Imbalanced/imbalanced/
```

Dataset classes: Atelectasis, Effusion, Infiltration, No Finding, Nodule, Pneumothorax (balanced split).

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Folder Structure

```
project_root/
├── dataset/                    ← NOT committed to git
├── preprocessing/
│   ├── dataset_loader.py
│   ├── preprocessor.py
│   └── preprocessing_experiments.ipynb
├── filtering_spatial/
│   ├── gaussian_lpf.py
│   ├── unsharp_masking.py
│   ├── spatial_filter.py
│   └── spatial_experiments.ipynb
├── filtering_frequency/
│   ├── fft_transform.py
│   ├── butterworth.py
│   ├── frequency_filter.py
│   └── frequency_experiments.ipynb
├── enhancement_roi/
│   ├── clahe.py
│   ├── morphological.py
│   ├── roi_pipeline.py
│   └── enhancement_experiments.ipynb
├── classification/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   └── classification_experiments.ipynb
├── evaluation/
│   ├── image_metrics.py
│   ├── classification_metrics.py
│   ├── evaluator.py
│   └── evaluation_experiments.ipynb
├── output/                     ← NOT committed to git
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Pipeline Diagram

```
Original Images
      │
      ▼
[Preprocessing]  →  resize (224/512/1024) + MinMax/Z-score normalization
      │
      ├──▶ [Spatial Filtering]
      │         ├── Gaussian LPF
      │         └── Unsharp Masking
      │
      ├──▶ [Frequency Filtering]
      │         ├── 2D FFT
      │         ├── Butterworth LPF
      │         ├── Butterworth HPF
      │         └── Inverse FFT
      │
      └──▶ [ROI Enhancement]
                ├── CLAHE
                ├── CLAHE + Top-Hat
                ├── CLAHE + Opening
                └── CLAHE + Closing
                      │
                      ▼
              [Classification — ResNet-18]
              5 scenarios × (Acc / Prec / Recall / F1)
                      │
                      ▼
               [Evaluation Module]
               PSNR · SSIM · CSV · JSON
```

---

## 6. How To Run

Run notebooks in order from the project root:

```bash
jupyter notebook
```

Execute in this order:

1. `preprocessing/preprocessing_experiments.ipynb`
2. `filtering_spatial/spatial_experiments.ipynb`
3. `filtering_frequency/frequency_experiments.ipynb`
4. `enhancement_roi/enhancement_experiments.ipynb`
5. `classification/classification_experiments.ipynb`
6. `evaluation/evaluation_experiments.ipynb`

---

## 7. Experiment Workflow

Each notebook:
- Loads processed images from `output/` of the previous stage
- Runs parameter sweeps and saves visualizations
- Saves processed images to its corresponding `output/` subdirectory

---

## 8. Evaluation Workflow

`evaluation/evaluation_experiments.ipynb` aggregates results from all prior stages:
- PSNR/SSIM comparing original vs. each filter stage
- Classification metrics for all 5 training scenarios
- Exports `output/06_evaluation/psnr/`, `ssim/`, `classification_metrics/`
- Generates `summary_report.txt`

---

## 9. Team Responsibilities

| Person | Module | Responsibilities |
|--------|--------|-----------------|
| Person 1 | `preprocessing/` | Dataset loader, resize, normalization |
| Person 2 | `filtering_spatial/` | Gaussian LPF, Unsharp Masking |
| Person 3 | `filtering_frequency/` | FFT, Butterworth LPF/HPF, IFFT |
| Person 4 | `enhancement_roi/` | CLAHE, Top-Hat, Opening, Closing |
| Person 5 | `classification/` | ResNet-18 training, 5 scenarios |
| All | `evaluation/` | PSNR, SSIM, classification metrics |

---

## 10. Reproducibility Notes

- Python 3.10+
- Set `seed=42` in `ClassificationConfig` for reproducible splits
- All random ops in PyTorch use the configured seed
- Images are saved at each stage so any stage can be re-run independently
- All outputs are deterministic given the same input images and parameters

```python
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

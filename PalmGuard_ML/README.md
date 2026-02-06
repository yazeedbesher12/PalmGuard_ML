# PalmGuard ML — Early Red Palm Weevil Detection (Audio ML + Streamlit)

PalmGuard is a prototype **machine-learning pipeline** for early detection of *Red Palm Weevil* infestation from **tree audio recordings**:
- **WAV → log-mel spectrogram segments**
- **Small CNN** classifier (healthy vs infected)
- **Tree-level risk scoring** by aggregating segment probabilities
- **Streamlit UI** for data exploration, training, and inference

> ⚠️ The dataset included in this repo is **synthetic** (for prototyping/demo). Replace it with real recordings for research-grade results.

---

## Project structure

```
PalmGuard_ML/
  app.py                 # Streamlit UI
  train.py               # Train using train/val/test split by tree_id
  infer.py               # Run inference + produce risk ranking + metrics
  train_onefile.py        # Single-file training script (legacy / quick demo)

  src/
    audio.py             # WAV loading, resampling, log-mel extraction
    dataset.py           # labels loader, split_by_tree, SegmentDataset
    model.py             # SmallCNN
    train_utils.py       # training loop + metrics saving
    infer_utils.py       # inference + aggregation + tree metrics
    config.py            # paths + AudioConfig

  data/
    dataset/             # dataset lives here (labels.csv + WAV files)
  models/                # best model checkpoint + training metrics
  outputs/               # inference outputs (segment predictions, risk ranking, etc.)
  assets/                # banner/icons/videos used by the UI
```

---

## Requirements

- Python **3.10+** recommended
- OS: Windows / macOS / Linux
- Packages in `requirements.txt`:
  - numpy, pandas, scipy, matplotlib
  - torch
  - scikit-learn
  - streamlit

---

## Quick start

```bash
# 1) (optional) create a venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) run the UI
streamlit run app.py
```

---

## Dataset format

Dataset lives under: `data/dataset/`

### labels.csv (required)

Minimum required columns:
- `file_path` : relative path under `data/dataset/`
- `tree_id`   : tree identifier (will be normalized to 4 digits in code)
- `label`     : `0=healthy`, `1=infected`

Extra columns are allowed (timestamp, sensor_type, etc.).

Example:
```
file_path,tree_id,label
data/tree_0001/2026-01-10T11-00.wav,1,0
data/tree_0002/2026-01-10T13-00.wav,2,1
```

### WAV files (required)

Expected path pattern:
```
data/dataset/data/tree_XXXX/*.wav
```

Audio is resampled to **16 kHz**, converted to **mono**, then segmented (default):
- `seg_s = 2.0s`
- `hop_s = 1.0s`
- `n_mels = 64`

### tree_locations.csv (optional)
If you want a map view in the UI:
- `tree_id`
- `lat`, `lon` (or any columns you use in the UI)

---

## Training

Train with **tree-level split** (avoids data leakage across recordings from the same tree):

```bash
python train.py
```

Artifacts:
- `models/best_cnn.pt` — best checkpoint on validation accuracy
- `models/metrics.json` — training history + best epoch + test segment metrics

Tuning knobs live at the top of `train.py`:
- split ratios, seed, stratify
- batch size, epochs, learning rate
- caching and DataLoader workers

---

## Inference + risk ranking

```bash
python infer.py
```

Artifacts (written to `outputs/`):
- `segment_predictions.csv` — per-segment probabilities
- `tree_risk_ranking.csv` — tree-level risk score + recommendation
- `tree_metrics.json` and `tree_confusion_matrix.csv` (if labels exist)

Tree risk definition (default):
- `risk(tree) = mean(top_k highest p_infected across that tree's segments)`
- default `top_k = 5`

---

## Notes on evaluation

- Segment-level metrics can be noisy because a single recording may contain both “quiet” and “informative” windows.
- Tree-level aggregation improves robustness (multiple segments per tree).

If you want research-grade evaluation:
- Report metrics on a **held-out test set** (not on all files).
- Consider cross-validation by **tree_id**.
- Track more metrics (ROC-AUC, PR-AUC) and calibrate risk thresholds.

---

## Reproducibility tips

Recommended additions (not required to run the repo):
- Pin package versions in `requirements.txt`
- Add a `.gitignore` to exclude:
  - `.idea/`, `__pycache__/`, large `outputs/` runs
- Add unit tests for:
  - `split_by_tree` (no leakage)
  - log-mel shapes
  - risk aggregation logic

---

## Troubleshooting

- **`labels.csv not found`**: ensure `data/dataset/labels.csv` exists.
- **Slow dataset init**: `SegmentDataset` precomputes segment counts; use smaller datasets for quick iterations or enable caching carefully.
- **CUDA not used**: verify PyTorch CUDA build and GPU availability; otherwise CPU is fine for this small CNN.

---

## License / attribution

This repository is a prototype for educational/research purposes.
Add a license if you plan to publish or commercialize.

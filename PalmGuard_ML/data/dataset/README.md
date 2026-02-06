# PalmGuard Synthetic Dataset v2 (100 trees)

This dataset is **synthetic** and intended only for prototyping / UI demos.

## Summary
- Trees: 100
- Recordings per tree: 3
- Total WAV files: 300
- Sample rate: 16000 Hz
- Duration per file: 6.0 s
- Labels:
  - 0 = healthy
  - 1 = infected

## Folder structure
```
data/dataset/
  labels.csv
  data/tree_0001/*.wav
  data/tree_0002/*.wav
  ...
  tree_locations.csv
```

## labels.csv schema

Required:
- `file_path` (string): relative path under `data/dataset/`
- `tree_id` (string/int): tree identifier
- `label` (int): 0 or 1

Optional (present in this synthetic dataset):
- `timestamp`
- `sample_rate`
- `duration_s`
- `sensor_type`

## Notes
- `labels.csv` intentionally has **no split column** so splitting happens in code (by `tree_id`).
- For real deployments, ensure labels are tree-accurate and avoid leakage (same tree in train and test).

# models/

This folder stores trained model artifacts.

## Files

- `best_cnn.pt`
  - PyTorch checkpoint saved by `src/train_utils.train_model()`.
  - Contains:
    - `state_dict`
    - `best_val_acc`
    - `best_epoch`
    - `audio_cfg` (AudioConfig used during training)

- `metrics.json`
  - Training history (train/val) and summary metrics.
  - Overwritten by subsequent training runs.

## Tip
For cleaner versioning, consider storing large checkpoints outside the repo (e.g., release assets) and keeping only metrics and configs.

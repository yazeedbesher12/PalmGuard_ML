"""PalmGuard ML — One-file training entrypoint (clean).

This script trains the model using a tree-level split to avoid leakage.

Run:
  python train_onefile.py --epochs 12 --batch-size 32 --lr 1e-3

Notes:
- Uses src.dataset.SegmentDataset and src.train_utils.train_model.
- Saves best checkpoint to models/best_cnn.pt and metrics to models/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict

import numpy as np
import torch

from src.config import BEST_MODEL_PATH, LABELS_CSV, MODEL_DIR, AudioConfig
from src.dataset import SegmentDataset, load_labels, split_by_tree
from src.train_utils import evaluate_segment_level, train_model

LOGGER = logging.getLogger("palmguard.train_onefile")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility (may reduce speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PalmGuard training (tree-level split)")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--cache-audio",
        action="store_true",
        help="Cache per-file mel segments in RAM for faster training (uses more memory)",
    )
    return p.parse_args()


def save_metrics(path: str, metrics: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # Basic validation
    ratios_sum = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {ratios_sum})")

    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing labels.csv: {LABELS_CSV}")

    _set_seed(int(args.seed))

    df = load_labels(LABELS_CSV)

    df_train, df_val, df_test = split_by_tree(
        df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        stratify=True,
    )

    LOGGER.info("Files: train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    audio_cfg = AudioConfig()
    LOGGER.info(
        "AudioConfig: sr=%d seg_s=%.2f hop_s=%.2f n_mels=%d fmin=%.1f fmax=%.1f",
        audio_cfg.sr,
        audio_cfg.seg_s,
        audio_cfg.hop_s,
        audio_cfg.n_mels,
        audio_cfg.fmin,
        audio_cfg.fmax,
    )

    train_ds = SegmentDataset(df_train, audio_cfg=audio_cfg, cache_audio=bool(args.cache_audio))
    val_ds = SegmentDataset(df_val, audio_cfg=audio_cfg, cache_audio=bool(args.cache_audio))
    test_ds = SegmentDataset(df_test, audio_cfg=audio_cfg, cache_audio=bool(args.cache_audio))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Device: %s", device)

    model, train_metrics = train_model(
        train_loader,
        val_loader,
        device=device,
        lr=float(args.lr),
        epochs=int(args.epochs),
        audio_cfg=audio_cfg,
    )

    test_metrics = evaluate_segment_level(model, test_loader, device=device)

    merged = dict(train_metrics)
    merged["test_acc_segment"] = float(test_metrics.get("acc", 0.0))
    merged["test_loss_segment"] = float(test_metrics.get("loss", 0.0))
    merged["split"] = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
        "seed": int(args.seed),
        "stratify": True,
        "tree_level": True,
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    save_metrics(metrics_path, merged)

    LOGGER.info("Saved model checkpoint: %s", BEST_MODEL_PATH)
    LOGGER.info("Saved metrics: %s", metrics_path)
    if "best_val_acc" in merged and "best_epoch" in merged:
        LOGGER.info("Best val acc: %.4f (epoch %d)", float(merged["best_val_acc"]), int(merged["best_epoch"]))
    LOGGER.info("Test acc (segment): %.4f", float(merged["test_acc_segment"]))



if __name__ == "__main__":
    main()

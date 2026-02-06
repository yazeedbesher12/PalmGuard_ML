from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

from src.config import AudioConfig, LABELS_CSV, MODEL_DIR
from src.dataset import load_labels, split_by_tree, SegmentDataset
from src.train_utils import train_model, evaluate_segment_level


def main():
    # -----------------------
    # Settings (edit these)
    # -----------------------
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    seed = 42
    stratify = True

    batch_size = 32
    epochs = 12
    lr = 1e-3

    cache_audio = False  # True = faster epochs but uses more RAM
    num_workers = 0      # set >0 if your OS supports it well

    # -----------------------
    # Load labels (NO split column needed)
    # -----------------------
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(
            f"labels.csv not found: {LABELS_CSV}\n"
            "Put your dataset under data/dataset/labels.csv"
        )

    df = load_labels(LABELS_CSV)
    print(f"Loaded labels: {df.shape[0]} files")

    # -----------------------
    # Split by tree (avoid leakage)
    # -----------------------
    df_train, df_val, df_test = split_by_tree(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=stratify,
    )

    n_trees_total = df["tree_id"].nunique()
    n_trees_inf = int(df.groupby("tree_id")["label"].max().sum())
    print(f"Trees total: {n_trees_total} | infected trees: {n_trees_inf}")
    print(
        "Split files:",
        f"train={df_train.shape[0]}",
        f"val={df_val.shape[0]}",
        f"test={df_test.shape[0]}",
    )
    print(
        "Split trees:",
        f"train={df_train['tree_id'].nunique()}",
        f"val={df_val['tree_id'].nunique()}",
        f"test={df_test['tree_id'].nunique()}",
    )

    # -----------------------
    # Datasets / loaders (segment-level)
    # -----------------------
    audio_cfg = AudioConfig()
    train_ds = SegmentDataset(df_train, audio_cfg=audio_cfg, cache_audio=cache_audio)
    val_ds = SegmentDataset(df_val, audio_cfg=audio_cfg, cache_audio=cache_audio)
    test_ds = SegmentDataset(df_test, audio_cfg=audio_cfg, cache_audio=cache_audio)

    print(f"Segments: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----------------------
    # Train
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(MODEL_DIR, exist_ok=True)

    model, metrics = train_model(
        train_loader,
        val_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        audio_cfg=audio_cfg,
    )

    print("Best val acc:", round(metrics["best_val_acc"], 4), "(epoch", metrics["best_epoch"], ")")

    # -----------------------
    # Final test metrics (segment-level)
    # -----------------------
    test_metrics = evaluate_segment_level(model, test_loader, device=device)
    print("Test acc (segment-level):", round(test_metrics["acc"], 4), "| loss:", round(test_metrics["loss"], 4))

    # -----------------------
    # Save merged run metrics for UI
    # -----------------------
    merged = dict(metrics)
    merged["test_acc_segment"] = float(test_metrics["acc"])
    merged["test_loss_segment"] = float(test_metrics["loss"])
    merged["split"] = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
        "seed": int(seed),
        "stratify": bool(stratify),
        "tree_level": True,
    }

    out_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()

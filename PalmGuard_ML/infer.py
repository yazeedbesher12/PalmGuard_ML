from __future__ import annotations

import os

import pandas as pd
import torch

from src.config import LABELS_CSV, DATASET_DIR, BEST_MODEL_PATH, OUTPUTS_DIR
from src.dataset import load_labels
from src.infer_utils import load_model, infer_segment_probs, aggregate_tree_risk, tree_level_metrics, tree_confusion_matrix_df


def main():
    # -----------------------
    # Settings (edit these)
    # -----------------------
    top_k = 5
    batch_size = 64

    # -----------------------
    # Checks
    # -----------------------
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {BEST_MODEL_PATH}. Train first using: python train.py"
        )

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # -----------------------
    # Load data + model
    # -----------------------
    df = load_labels(LABELS_CSV)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, audio_cfg, ckpt = load_model(BEST_MODEL_PATH, device=device)
    print("Loaded model. best_val_acc:", ckpt.get("best_val_acc"), "best_epoch:", ckpt.get("best_epoch"))

    # -----------------------
    # Infer segment probabilities
    # -----------------------
    seg_df = infer_segment_probs(
        model,
        df_files=df,
        dataset_dir=DATASET_DIR,
        audio_cfg=audio_cfg,
        device=device,
        batch_size=batch_size,
    )

    seg_out = os.path.join(OUTPUTS_DIR, "segment_predictions.csv")
    seg_df.to_csv(seg_out, index=False)
    print("Saved:", seg_out, "| rows:", seg_df.shape[0])

    # -----------------------
    # Aggregate to tree risk score
    # -----------------------
    tree_df = aggregate_tree_risk(seg_df, top_k=top_k)

    tree_out = os.path.join(OUTPUTS_DIR, "tree_risk_ranking.csv")
    tree_df.to_csv(tree_out, index=False)
    print("Saved:", tree_out, "| rows:", tree_df.shape[0])

    # -----------------------
    # Tree-level evaluation (if labels exist)
    # -----------------------
    if "label" in tree_df.columns:
        threshold = 0.5  # edit me
        metrics = tree_level_metrics(tree_df, threshold=threshold)
        cm_df = tree_confusion_matrix_df(tree_df, threshold=threshold)

        metrics_out = os.path.join(OUTPUTS_DIR, "tree_metrics.json")
        with open(metrics_out, "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        cm_out = os.path.join(OUTPUTS_DIR, "tree_confusion_matrix.csv")
        cm_df.to_csv(cm_out, index=True)

        print("\nTree-level metrics (threshold =", threshold, "):")
        print({k: metrics[k] for k in ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]})
        print("Saved:", metrics_out)
        print("Saved:", cm_out)

    # Quick view: top 10
    print("\nTop 10 risky trees:")
    print(tree_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

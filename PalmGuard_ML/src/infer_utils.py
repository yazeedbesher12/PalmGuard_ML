from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .audio import wav_load, logmel_segments
from .config import AudioConfig, BEST_MODEL_PATH, DATASET_DIR
from .model import SmallCNN


def load_model(model_path: str = BEST_MODEL_PATH, device: str = "cpu") -> Tuple[SmallCNN, AudioConfig, Dict]:
    """Load trained model checkpoint.

    Returns:
        model: SmallCNN in eval mode
        audio_cfg: AudioConfig used during training (if stored)
        ckpt: raw checkpoint dict
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    model = SmallCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Recover audio config if available
    cfg_dict = ckpt.get("audio_cfg", None)
    if isinstance(cfg_dict, dict):
        audio_cfg = AudioConfig(**cfg_dict)
    else:
        audio_cfg = AudioConfig()

    return model, audio_cfg, ckpt


@torch.no_grad()
def infer_segments_for_file(
    model: nn.Module,
    wav_path: str,
    *,
    label: Optional[int] = None,
    tree_id: Optional[str] = None,
    rel_path: Optional[str] = None,
    audio_cfg: AudioConfig = AudioConfig(),
    device: str = "cpu",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run inference on all segments of ONE WAV file and return a DataFrame.

    Output columns:
      - tree_id
      - file_path
      - seg_i
      - p_infected
      - label (if provided)
    """
    y, sr = wav_load(wav_path, target_sr=audio_cfg.sr)
    segs = logmel_segments(
        y,
        sr,
        seg_s=audio_cfg.seg_s,
        hop_s=audio_cfg.hop_s,
        n_mels=audio_cfg.n_mels,
        fmin=audio_cfg.fmin,
        fmax=audio_cfg.fmax,
        normalize=True,
    )

    if len(segs) == 0:
        return pd.DataFrame(columns=["tree_id", "file_path", "seg_i", "p_infected", "label"])

    X = torch.tensor(np.stack(segs, axis=0), dtype=torch.float32)  # (S, 1, n_mels, T)
    probs: List[float] = []

    for i in range(0, X.size(0), batch_size):
        xb = X[i:i + batch_size].to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]  # prob of infected
        probs.extend([float(v) for v in p.detach().cpu().numpy().tolist()])

    out = {
        "tree_id": [tree_id or ""] * len(probs),
        "file_path": [rel_path or wav_path] * len(probs),
        "seg_i": list(range(len(probs))),
        "p_infected": probs,
    }
    if label is not None:
        out["label"] = [int(label)] * len(probs)

    return pd.DataFrame(out)


@torch.no_grad()
def infer_segment_probs(
    model: nn.Module,
    df_files: pd.DataFrame,
    *,
    dataset_dir: str = DATASET_DIR,
    audio_cfg: AudioConfig = AudioConfig(),
    device: str = "cpu",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Infer segment probabilities for a dataframe of file-level rows.

    df_files must contain:
      - file_path (relative under dataset_dir)
      - tree_id
      - label (optional)
    """
    rows = []
    has_label = "label" in df_files.columns

    for row in df_files.itertuples(index=False):
        rel = str(getattr(row, "file_path"))
        tid = str(getattr(row, "tree_id"))
        lbl = int(getattr(row, "label")) if has_label else None

        wav_path = os.path.join(dataset_dir, rel)
        seg_df = infer_segments_for_file(
            model,
            wav_path,
            label=lbl,
            tree_id=tid,
            rel_path=rel,
            audio_cfg=audio_cfg,
            device=device,
            batch_size=batch_size,
        )
        if len(seg_df) > 0:
            rows.append(seg_df)

    if not rows:
        cols = ["tree_id", "file_path", "seg_i", "p_infected"]
        if "label" in df_files.columns:
            cols.append("label")
        return pd.DataFrame(columns=cols)

    return pd.concat(rows, ignore_index=True)


def recommend_action(risk: float) -> str:
    """Convert a risk score into an action recommendation."""
    if risk >= 0.75:
        return "HIGH: inspect within 48h"
    if risk >= 0.45:
        return "MEDIUM: inspect this week"
    return "LOW: routine monitoring"


def aggregate_tree_risk(seg_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Aggregate segment-level probabilities into a tree-level risk score.

    Risk definition:
      risk(tree) = mean( top_k highest p_infected across segments for that tree )

    Output columns:
      - tree_id
      - risk
      - recommendation
      - label (if present in seg_df)
      - n_segments
    """
    if seg_df.empty:
        return pd.DataFrame(columns=["tree_id", "risk", "recommendation", "label", "n_segments"])

    out_rows = []
    has_label = "label" in seg_df.columns

    for tid, g in seg_df.groupby("tree_id"):
        probs = g["p_infected"].astype(float).to_numpy()
        probs = np.sort(probs)[::-1]  # desc
        k = int(min(max(1, top_k), probs.size))
        risk = float(np.mean(probs[:k]))

        row = {
            "tree_id": tid,
            "risk": risk,
            "recommendation": recommend_action(risk),
            "n_segments": int(len(g)),
        }
        if has_label:
            row["label"] = int(g["label"].max())
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values("risk", ascending=False).reset_index(drop=True)
    return out

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def tree_level_metrics(tree_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Compute tree-level metrics using risk >= threshold as infected prediction.

    Requires columns:
      - risk
      - label (0/1)

    Returns:
      dict with accuracy, precision, recall, f1, and confusion counts.
    """
    if tree_df.empty:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "n": 0,
        }

    if "label" not in tree_df.columns:
        raise ValueError("tree_df must contain 'label' column to compute metrics")

    y_true = tree_df["label"].astype(int).to_numpy()
    y_pred = (tree_df["risk"].astype(float).to_numpy() >= float(threshold)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(y_true.size)

    acc = _safe_div(tp + tn, n)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
    }


def tree_confusion_matrix_df(tree_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Return a 2x2 confusion matrix DataFrame for tree-level predictions.

    Rows: actual (0,1)
    Cols: predicted (0,1)
    """
    if tree_df.empty:
        return pd.DataFrame([[0, 0], [0, 0]], index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])

    if "label" not in tree_df.columns:
        raise ValueError("tree_df must contain 'label' column to compute confusion matrix")

    y_true = tree_df["label"].astype(int).to_numpy()
    y_pred = (tree_df["risk"].astype(float).to_numpy() >= float(threshold)).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

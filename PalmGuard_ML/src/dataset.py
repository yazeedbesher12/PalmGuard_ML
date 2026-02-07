from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import AudioConfig, DATASET_DIR, LABELS_CSV
from .audio import wav_load, logmel_segments


def load_labels(csv_path: str = LABELS_CSV) -> pd.DataFrame:
    """Load labels.csv.
    Expected columns (minimum):
      - file_path: relative path under data/dataset/
      - tree_id
      - label (0/1)
    Note: We intentionally DO NOT require a 'split' column.
    """

    df = pd.read_csv(csv_path)
    if "file_path" not in df.columns:
        raise ValueError("labels.csv must contain 'file_path' column")

    if "tree_id" in df.columns:
        df["tree_id"] = df["tree_id"].astype(str).str.zfill(4)
    else:
        raise ValueError("labels.csv must contain 'tree_id' column")

    if "label" not in df.columns:
        raise ValueError("labels.csv must contain 'label' column")

    df["label"] = df["label"].astype(int)
    df["file_path"] = df["file_path"].astype(str)

    return df


def split_by_tree(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test by *tree_id* (no leakage).

    Why by tree?
      If we split by file, the same tree can appear in both train and test,
      which inflates metrics and is unrealistic.

    Stratified split (recommended):
      We split trees separately for label=0 and label=1 (tree-level),
      then combine, so each split has similar infection ratio.
    """
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0):
        raise ValueError("Ratios must be non-negative")
    s = float(ratios.sum())
    if s <= 0:
        raise ValueError("Sum of ratios must be > 0")
    ratios = ratios / s

    rng = np.random.default_rng(seed)

    # Tree-level label (in case a tree has multiple files and labels)
    tree_df = df.groupby("tree_id")["label"].max().reset_index()
    if stratify:
        trees0 = tree_df[tree_df["label"] == 0]["tree_id"].tolist()
        trees1 = tree_df[tree_df["label"] == 1]["tree_id"].tolist()
        rng.shuffle(trees0)
        rng.shuffle(trees1)

        def split_list(trees: List[str]) -> Tuple[List[str], List[str], List[str]]:
            n = len(trees)
            n_train = int(round(ratios[0] * n))
            n_val = int(round(ratios[1] * n))
            # remainder goes to test
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            train = trees[:n_train]
            val = trees[n_train:n_train + n_val]
            test = trees[n_train + n_val:]
            return train, val, test

        t0, v0, te0 = split_list(trees0)
        t1, v1, te1 = split_list(trees1)

        train_trees = t0 + t1
        val_trees = v0 + v1
        test_trees = te0 + te1

        rng.shuffle(train_trees)
        rng.shuffle(val_trees)
        rng.shuffle(test_trees)
    else:
        trees = tree_df["tree_id"].tolist()
        rng.shuffle(trees)
        n = len(trees)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train_trees = trees[:n_train]
        val_trees = trees[n_train:n_train + n_val]
        test_trees = trees[n_train + n_val:]

    train_df = df[df["tree_id"].isin(train_trees)].reset_index(drop=True)
    val_df = df[df["tree_id"].isin(val_trees)].reset_index(drop=True)
    test_df = df[df["tree_id"].isin(test_trees)].reset_index(drop=True)
    return train_df, val_df, test_df


@dataclass(frozen=True)
class SegmentIndex:
    row_i: int
    seg_i: int


class SegmentDataset(Dataset):
    """Segment-level dataset built from file-level WAV recordings.

    Each WAV is split into windows (segments), each becomes one sample.

    __getitem__ returns:
      x: torch.FloatTensor  shape (1, n_mels, T)
      y: torch.LongTensor   0/1
      tree_id: str
      file_path: str (relative)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_cfg: AudioConfig = AudioConfig(),
        dataset_dir: str = DATASET_DIR,
        cache_audio: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_cfg = audio_cfg
        self.dataset_dir = dataset_dir
        self.cache_audio = cache_audio

        self._index: List[SegmentIndex] = []
        self._cache: Dict[int, List[np.ndarray]] = {}

        # Build index
        for i, row in self.df.iterrows():
            rel = str(row["file_path"])
            wav_path = os.path.join(self.dataset_dir, rel)
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Missing WAV: {wav_path}")

            # Compute segments once (either cache features or just count them)
            y_wav, sr = wav_load(wav_path, target_sr=self.audio_cfg.sr)
            segs = logmel_segments(
                y_wav,
                sr,
                seg_s=self.audio_cfg.seg_s,
                hop_s=self.audio_cfg.hop_s,
                n_mels=self.audio_cfg.n_mels,
                fmin=self.audio_cfg.fmin,
                fmax=self.audio_cfg.fmax,
                normalize=True,
            )

            if self.cache_audio:
                self._cache[i] = segs

            for s_i in range(len(segs)):
                self._index.append(SegmentIndex(row_i=i, seg_i=s_i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si = self._index[idx]
        row = self.df.loc[si.row_i]

        rel = str(row["file_path"])
        tree_id = str(row["tree_id"])
        y_label = int(row["label"])

        if self.cache_audio:
            x = self._cache[si.row_i][si.seg_i]
        else:
            wav_path = os.path.join(self.dataset_dir, rel)
            y_wav, sr = wav_load(wav_path, target_sr=self.audio_cfg.sr)
            segs = logmel_segments(
                y_wav,
                sr,
                seg_s=self.audio_cfg.seg_s,
                hop_s=self.audio_cfg.hop_s,
                n_mels=self.audio_cfg.n_mels,
                fmin=self.audio_cfg.fmin,
                fmax=self.audio_cfg.fmax,
                normalize=True,
            )
            x = segs[si.seg_i]

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y_label, dtype=torch.long)
        return x_t, y_t, tree_id, rel

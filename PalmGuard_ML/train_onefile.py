# PalmGuard ML v2 — ONE FILE TRAINING SCRIPT (no split column needed)
# ---------------------------------------------------------------
# This script:
# 1) loads data/dataset/labels.csv
# 2) splits by tree_id (stratified) into train/val/test
# 3) extracts log-mel segments from WAV files
# 4) trains a small CNN
# 5) saves best model to models/best_cnn.pt + metrics to models/metrics.json
#
# Run:
#   pip install -r requirements.txt
#   python train_onefile.py
#
# Notes:
# - For fair evaluation, we split by TREE (not by file).
# - Change ratios below as you like.

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.audio import wav_load, logmel_segments

# -----------------------
# Settings (edit me)
# -----------------------
DATASET_DIR = "data/dataset"
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED = 42
STRATIFY_BY_LABEL = True

TARGET_SR = 16000
SEG_S = 2.0
HOP_S = 1.0
N_MELS = 64
FMIN = 50.0
FMAX = 3500.0

EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-3

SAVE_DIR = "models"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_cnn.pt")
METRICS_PATH = os.path.join(SAVE_DIR, "metrics.json")

# -----------------------
# Load CSV
# -----------------------
if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"Missing labels.csv: {LABELS_CSV}")

df = pd.read_csv(LABELS_CSV)

need = ["file_path", "tree_id", "label"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in labels.csv: {missing}")

df["tree_id"] = df["tree_id"].astype(str).str.zfill(4)
df["label"] = df["label"].astype(int)
df["file_path"] = df["file_path"].astype(str)

# -----------------------
# Split by TREE (no leakage)
# -----------------------
ratios = np.array([TRAIN_RATIO, VAL_RATIO, TEST_RATIO], dtype=float)
if (ratios < 0).any() or ratios.sum() <= 0:
    raise ValueError("Bad split ratios")
ratios = ratios / ratios.sum()

rng = np.random.default_rng(SEED)

tree_df = df.groupby("tree_id")["label"].max().reset_index()

if STRATIFY_BY_LABEL:
    trees0 = tree_df[tree_df["label"] == 0]["tree_id"].tolist()
    trees1 = tree_df[tree_df["label"] == 1]["tree_id"].tolist()
    rng.shuffle(trees0)
    rng.shuffle(trees1)

    # label 0
    n0 = len(trees0)
    n0_train = int(round(ratios[0] * n0))
    n0_val   = int(round(ratios[1] * n0))
    n0_train = min(n0_train, n0)
    n0_val   = min(n0_val, n0 - n0_train)
    train0 = trees0[:n0_train]
    val0   = trees0[n0_train:n0_train + n0_val]
    test0  = trees0[n0_train + n0_val:]

    # label 1
    n1 = len(trees1)
    n1_train = int(round(ratios[0] * n1))
    n1_val   = int(round(ratios[1] * n1))
    n1_train = min(n1_train, n1)
    n1_val   = min(n1_val, n1 - n1_train)
    train1 = trees1[:n1_train]
    val1   = trees1[n1_train:n1_train + n1_val]
    test1  = trees1[n1_train + n1_val:]

    train_trees = train0 + train1
    val_trees   = val0 + val1
    test_trees  = test0 + test1

    rng.shuffle(train_trees)
    rng.shuffle(val_trees)
    rng.shuffle(test_trees)
else:
    trees = tree_df["tree_id"].tolist()
    rng.shuffle(trees)
    n = len(trees)
    n_train = int(round(ratios[0] * n))
    n_val   = int(round(ratios[1] * n))
    train_trees = trees[:n_train]
    val_trees   = trees[n_train:n_train + n_val]
    test_trees  = trees[n_train + n_val:]

df_train = df[df["tree_id"].isin(train_trees)].reset_index(drop=True)
df_val   = df[df["tree_id"].isin(val_trees)].reset_index(drop=True)
df_test  = df[df["tree_id"].isin(test_trees)].reset_index(drop=True)

print("Trees:", tree_df.shape[0], "| infected trees:", int((tree_df["label"] == 1).sum()))
print("Split trees:", len(train_trees), len(val_trees), len(test_trees))
print("Split files:", df_train.shape[0], df_val.shape[0], df_test.shape[0])

# -----------------------
# Feature extraction (build segment-level arrays)
# -----------------------
def build_segments(df_part: pd.DataFrame, name: str):
    X_list = []
    y_list = []

    for row in df_part.itertuples(index=False):
        wav_path = os.path.join(DATASET_DIR, row.file_path)
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Missing WAV: {wav_path}")

        y_wav, sr = wav_load(wav_path, target_sr=TARGET_SR)
        segs = logmel_segments(
            y_wav, sr,
            seg_s=SEG_S, hop_s=HOP_S,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
            normalize=True
        )
        for s in segs:
            X_list.append(s.astype(np.float32))  # (1, n_mels, T)
            y_list.append(int(row.label))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    print(f"{name}: segments={X.shape[0]}  X={tuple(X.shape)}  y={tuple(y.shape)}")
    return X, y

# NOTE: This is written as a function for readability. If you want it truly 100% no-functions,
# we can inline it, but it becomes longer and harder to read.

X_train, y_train = build_segments(df_train, "train")
X_val,   y_val   = build_segments(df_val,   "val")
X_test,  y_test  = build_segments(df_test,  "test")

# -----------------------
# Torch tensors
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t   = torch.from_numpy(X_val)
y_val_t   = torch.from_numpy(y_val)
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

# -----------------------
# Model (CNN)
# -----------------------
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2)),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2)),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(64, 2),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -----------------------
# Train loop
# -----------------------
best_val_acc = -1.0
best_epoch = -1
history = {"train": [], "val": []}

os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    perm = torch.randperm(X_train_t.size(0))
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for i in range(0, perm.numel(), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb = X_train_t[idx].to(device)
        yb = y_train_t[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += float(loss.detach().cpu().item()) * int(yb.numel())
        pred = torch.argmax(logits, dim=1)
        train_correct += int((pred == yb).sum().detach().cpu().item())
        train_total += int(yb.numel())

    train_loss = train_loss_sum / max(1, train_total)
    train_acc = train_correct / max(1, train_total)

    # ---- VAL ----
    model.eval()
    with torch.no_grad():
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        for i in range(0, X_val_t.size(0), BATCH_SIZE):
            xb = X_val_t[i:i + BATCH_SIZE].to(device)
            yb = y_val_t[i:i + BATCH_SIZE].to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            val_loss_sum += float(loss.detach().cpu().item()) * int(yb.numel())
            pred = torch.argmax(logits, dim=1)
            val_correct += int((pred == yb).sum().detach().cpu().item())
            val_total += int(yb.numel())

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

    history["train"].append({"epoch": epoch, "loss": float(train_loss), "acc": float(train_acc)})
    history["val"].append({"epoch": epoch, "loss": float(val_loss), "acc": float(val_acc)})

    print(f"epoch {epoch:02d} | train acc {train_acc:.4f} loss {train_loss:.4f} | val acc {val_acc:.4f} loss {val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = float(val_acc)
        best_epoch = int(epoch)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "settings": {
                    "TRAIN_RATIO": TRAIN_RATIO,
                    "VAL_RATIO": VAL_RATIO,
                    "TEST_RATIO": TEST_RATIO,
                    "SEED": SEED,
                    "TARGET_SR": TARGET_SR,
                    "SEG_S": SEG_S,
                    "HOP_S": HOP_S,
                    "N_MELS": N_MELS,
                    "FMIN": FMIN,
                    "FMAX": FMAX,
                },
            },
            BEST_MODEL_PATH,
        )

# -----------------------
# TEST (load best)
# -----------------------
ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    test_correct = 0
    test_total = 0
    for i in range(0, X_test_t.size(0), BATCH_SIZE):
        xb = X_test_t[i:i + BATCH_SIZE].to(device)
        yb = y_test_t[i:i + BATCH_SIZE].to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        test_correct += int((pred == yb).sum().detach().cpu().item())
        test_total += int(yb.numel())
    test_acc = test_correct / max(1, test_total)

metrics = {
    "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    "counts": {
        "trees_total": int(tree_df.shape[0]),
        "trees_infected": int((tree_df["label"] == 1).sum()),
        "files_total": int(df.shape[0]),
        "segments_train": int(X_train_t.size(0)),
        "segments_val": int(X_val_t.size(0)),
        "segments_test": int(X_test_t.size(0)),
    },
    "best_val_acc": float(best_val_acc),
    "best_epoch": int(best_epoch),
    "test_acc": float(test_acc),
    "history": history,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("--------------------------------------------------")
print("Saved:", BEST_MODEL_PATH)
print("Saved:", METRICS_PATH)
print("Best val acc:", round(best_val_acc, 4), "at epoch", best_epoch)
print("Test acc:", round(float(test_acc), 4))

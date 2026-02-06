from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import BEST_MODEL_PATH, MODEL_DIR, AudioConfig
from .model import SmallCNN


@torch.no_grad()
def evaluate_segment_level(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses = []

    loss_fn = nn.CrossEntropyLoss()

    for x, y, *_ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.detach().cpu().item()))

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().detach().cpu().item())
        total += int(y.numel())

    acc = correct / max(1, total)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(acc),
        "n": float(total),
    }


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: str = "cpu",
    lr: float = 1e-3,
    epochs: int = 10,
    audio_cfg: AudioConfig = AudioConfig(),
) -> Tuple[SmallCNN, Dict]:
    """Train SmallCNN and save best checkpoint to models/best_cnn.pt."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train": [], "val": []}
    best_val_acc = -1.0
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for x, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            pred = torch.argmax(logits, dim=1)
            train_correct += int((pred == y).sum().detach().cpu().item())
            train_total += int(y.numel())

        train_acc = train_correct / max(1, train_total)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_metrics = evaluate_segment_level(model, val_loader, device=device)

        history["train"].append({"epoch": ep, "loss": train_loss, "acc": float(train_acc)})
        history["val"].append({"epoch": ep, **val_metrics})

        # Save best
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = ep

            ckpt = {
                "state_dict": model.state_dict(),
                "best_val_acc": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "audio_cfg": asdict(audio_cfg),
            }
            torch.save(ckpt, BEST_MODEL_PATH)

    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "history": history,
    }

    # Save metrics json
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics

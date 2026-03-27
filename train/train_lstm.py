"""
train/train_lstm.py
===================
Full training pipeline for the violence classification model.

Supports both BiLSTM and ST-Transformer architectures.
Logs to TensorBoard. Saves best model checkpoint.
Produces training curves and final metrics.

Usage:
    # Train BiLSTM (default, faster)
    python train/train_lstm.py --dataset data/datasets/RWF-2000 --arch lstm

    # Train Transformer
    python train/train_lstm.py --dataset data/datasets/RWF-2000 --arch transformer

    # Resume training
    python train/train_lstm.py --resume models/best_lstm.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm

from modules.dataset_loader import get_dataloaders
from modules.classifier import build_model


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "arch":         "lstm",         # 'lstm' or 'transformer'
    "dataset":      "data/datasets/RWF-2000",
    "seq_len":      30,
    "batch_size":   16,
    "epochs":       50,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "hidden_dim":   256,            # for LSTM
    "d_model":      128,            # for Transformer
    "dropout":      0.4,
    "patience":     10,             # early stopping
    "save_dir":     "models",
    "log_dir":      "logs",
    "num_workers":  0,              # 0 = safe on macOS
}


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for x, y in tqdm(loader, desc="  Train", leave=False, ncols=80):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_labels, all_preds),
        "f1":   f1_score(all_labels, all_preds, zero_division=0),
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for x, y in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)

        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_labels, all_preds),
        "f1":   f1_score(all_labels, all_preds, zero_division=0),
        "auc":  auc,
    }


def train(cfg: dict):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Violence Detection — Training [{cfg['arch'].upper()}]")
    print(f"  Device: {device} | Dataset: {cfg['dataset']}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, feat_dim = get_dataloaders(
        cfg["dataset"],
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        num_workers=cfg["num_workers"],
    )
    print(f"Feature dim: {feat_dim}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_kwargs = (
        {"hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"]}
        if cfg["arch"] == "lstm"
        else {"d_model": cfg["d_model"], "dropout": cfg["dropout"]}
    )
    model = build_model(cfg["arch"], input_dim=feat_dim, device=device, **model_kwargs)

    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed from {cfg['resume']}")

    # ── Class weights for imbalanced data ────────────────────────────────────
    fight_count    = sum(1 for _, l in train_loader.dataset.samples if l == 1)
    nonfight_count = sum(1 for _, l in train_loader.dataset.samples if l == 0)
    total = fight_count + nonfight_count
    weights = torch.tensor(
        [total / (2 * nonfight_count), total / (2 * fight_count)],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # ── Logging ───────────────────────────────────────────────────────────────
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True)
    log_dir = Path(cfg["log_dir"]) / f"{cfg['arch']}_{int(time.time())}"
    writer = SummaryWriter(log_dir)

    best_f1, patience_counter = 0.0, 0
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        # Log
        writer.add_scalars("Loss",     {"train": train_m["loss"],   "val": val_m["loss"]},   epoch)
        writer.add_scalars("Accuracy", {"train": train_m["acc"],    "val": val_m["acc"]},    epoch)
        writer.add_scalars("F1",       {"train": train_m["f1"],     "val": val_m["f1"]},     epoch)
        writer.add_scalar("AUC/val",   val_m["auc"], epoch)

        row = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Loss: {train_m['loss']:.4f}/{val_m['loss']:.4f} | "
              f"Acc: {train_m['acc']:.4f}/{val_m['acc']:.4f} | "
              f"F1: {train_m['f1']:.4f}/{val_m['f1']:.4f} | "
              f"AUC: {val_m['auc']:.4f} | "
              f"{elapsed:.1f}s")

        # Save best checkpoint
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            patience_counter = 0
            ckpt_path = save_dir / f"best_{cfg['arch']}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "config":      cfg,
                "feat_dim":    feat_dim,
            }, ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}  (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg['patience']} epochs)")
                break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nFinal Evaluation on Validation Set")
    ckpt = torch.load(save_dir / f"best_{cfg['arch']}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print(classification_report(all_labels, all_preds,
                                 target_names=["NonFight", "Fight"], digits=4))
    print(f"AUC-ROC: {roc_auc_score(all_labels, all_probs):.4f}")

    # Save history
    history_path = save_dir / f"history_{cfg['arch']}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"Training history saved → {history_path}")
    writer.close()
    print(f"\nTensorBoard logs: tensorboard --logdir {log_dir}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default="lstm",      choices=["lstm", "transformer"])
    parser.add_argument("--dataset",    default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--epochs",     type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--seq_len",    type=int, default=DEFAULT_CONFIG["seq_len"])
    parser.add_argument("--resume",     type=str, default=None)
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **vars(args)}
    train(cfg)

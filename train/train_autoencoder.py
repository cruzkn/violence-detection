"""
train/train_autoencoder.py
==========================
Trains the SequenceAutoencoder on NonFight clips only.
After training, calibrates the AnomalyScorer threshold
using the validation set and saves calibration data.

Usage:
    python train/train_autoencoder.py --dataset data/datasets/RWF-2000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.dataset_loader import ViolenceDataset
from modules.autoencoder import SequenceAutoencoder, AnomalyScorer


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def train(cfg):
    device = get_device()
    print(f"\n{'='*55}")
    print(f"  Autoencoder Training")
    print(f"  Device: {device} | Dataset: {cfg['dataset']}")
    print(f"{'='*55}\n")

    # ── Build datasets: NONFIGHT ONLY for training ─────────────────────────
    full_train = ViolenceDataset(cfg["dataset"], split="train",
                                 seq_len=cfg["seq_len"], augment=False)
    full_val   = ViolenceDataset(cfg["dataset"], split="val",
                                 seq_len=cfg["seq_len"], augment=False)

    nf_train_idx = [i for i, (_, l) in enumerate(full_train.samples) if l == 0]
    nf_val_idx   = [i for i, (_, l) in enumerate(full_val.samples)   if l == 0]

    train_loader = DataLoader(Subset(full_train, nf_train_idx),
                              batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(full_val, nf_val_idx),
                              batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    full_val_loader = DataLoader(full_val, batch_size=cfg["batch_size"],
                                 shuffle=False, num_workers=0)

    feat_dim = full_train.feat_dim
    print(f"NonFight train clips: {len(nf_train_idx)}")
    print(f"NonFight val clips:   {len(nf_val_idx)}")
    print(f"Feature dim:          {feat_dim}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SequenceAutoencoder(
        input_dim=feat_dim,
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        seq_len=cfg["seq_len"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Autoencoder params: {n_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(f"logs/autoencoder_{int(time.time())}")

    best_loss, patience_counter = float("inf"), 0

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"  Epoch {epoch:3d} [train]", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                val_loss += criterion(x_hat, x).item() * len(x)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_autoencoder.pt")
            print(f"  ✓ Saved best autoencoder (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Calibrate anomaly threshold ────────────────────────────────────────
    print(f"\n{'='*55}\nCalibrating AnomalyScorer threshold...")
    model.load_state_dict(torch.load(save_dir / "best_autoencoder.pt", map_location=device))
    scorer = AnomalyScorer(model, device=device, sensitivity=0.95)
    scorer.calibrate(full_val_loader)
    scorer.save(str(save_dir / "anomaly_scorer"))

    writer.close()
    print(f"\nAutoencoder training complete ✓")
    print(f"Model:     {save_dir}/best_autoencoder.pt")
    print(f"Calibration: {save_dir}/anomaly_scorer_calibration.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="data/datasets/RWF-2000")
    parser.add_argument("--epochs",     type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seq_len",    type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--save_dir",   default="models")
    args = parser.parse_args()
    train(vars(args))

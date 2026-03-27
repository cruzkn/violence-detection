"""
Module 4: Autoencoder-based Anomaly Detection
Trains on NonFight sequences only. High reconstruction error = anomaly.
Dual-stream: combines supervised score + unsupervised anomaly score.

Usage:
    python modules/04_autoencoder.py --train
    python modules/04_autoencoder.py --threshold  # calibrate threshold on val set
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import json, argparse

import sys; sys.path.insert(0, str(Path(__file__).parent))
from lstm_classifier import auto_device, CFG, SkeletonDataset

AE_CFG = {
    "latent_dim": 32,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 60,
    "patience": 10,
}

# ── Normal-only Dataset ───────────────────────────────────────────────────────
class NormalOnlyDataset(Dataset):
    """Only loads NonFight sequences for autoencoder training."""
    NON_FIGHT = {"NonFight", "no"}

    def __init__(self, data_dir):
        self.samples = []
        data_dir = Path(data_dir)
        for cls_dir in data_dir.iterdir():
            if not cls_dir.is_dir(): continue
            if cls_dir.name in self.NON_FIGHT:
                for f in cls_dir.glob("*.npy"):
                    self.samples.append(f)
        print(f"  Normal-only dataset: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq = np.load(self.samples[idx]).astype(np.float32)
        return torch.from_numpy(seq)

# ── Autoencoder ───────────────────────────────────────────────────────────────
class SkeletonAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for skeleton sequence reconstruction.
    Encoder compresses T×51 → latent_dim
    Decoder reconstructs latent_dim → T×51
    """
    def __init__(self, feat_dim=51, seq_len=30, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        # Encoder
        self.enc_lstm = nn.LSTM(feat_dim, 128, batch_first=True, bidirectional=False)
        self.enc_fc   = nn.Linear(128, latent_dim)
        # Decoder
        self.dec_fc   = nn.Linear(latent_dim, 128)
        self.dec_lstm = nn.LSTM(128, feat_dim, batch_first=True)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)         # h: (1, B, 128)
        return self.enc_fc(h.squeeze(0))      # (B, latent_dim)

    def decode(self, z, seq_len):
        h0 = self.dec_fc(z).unsqueeze(0)     # (1, B, 128)
        inp = torch.zeros(z.size(0), seq_len, 128, device=z.device)
        out, _ = self.dec_lstm(inp, (h0, torch.zeros_like(h0)))
        return out  # (B, T, feat_dim)

    def forward(self, x):
        z    = self.encode(x)
        recon = self.decode(z, x.size(1))
        return recon, z

    def reconstruction_error(self, x):
        """Per-sample MSE reconstruction error (anomaly score)."""
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])
        return err  # (B,)

# ── Dual-stream Scorer ────────────────────────────────────────────────────────
class DualStreamScorer:
    """
    Combines supervised classifier score + unsupervised AE anomaly score.
    Final violence score = alpha * sup_score + (1-alpha) * ae_score_norm
    """
    def __init__(self, classifier, autoencoder, alpha=0.7,
                 ae_threshold=None, device=None):
        self.clf  = classifier
        self.ae   = autoencoder
        self.alpha = alpha
        self.ae_threshold = ae_threshold
        self.device = device or auto_device()

    def score(self, seq_tensor):
        """
        seq_tensor: (B, T, feat_dim) or (T, feat_dim) single sample
        Returns: dict with 'violence_prob', 'ae_error', 'alert', 'confidence'
        """
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        seq = seq_tensor.to(self.device)

        # Supervised score
        self.clf.eval()
        with torch.no_grad():
            logits = self.clf(seq)
            sup_prob = F.softmax(logits, dim=1)[:, 1].item()  # Fight prob

        # Unsupervised AE score
        self.ae.eval()
        ae_err = self.ae.reconstruction_error(seq).item()
        ae_norm = min(ae_err / (self.ae_threshold or 1.0), 1.0) if self.ae_threshold else 0.0

        # Fusion
        final_score = self.alpha * sup_prob + (1 - self.alpha) * ae_norm
        alert = final_score > 0.5

        return {
            "violence_prob" : round(sup_prob,   4),
            "ae_error"      : round(ae_err,     4),
            "ae_normalized" : round(ae_norm,    4),
            "final_score"   : round(final_score, 4),
            "alert"         : bool(alert),
            "confidence"    : round(final_score, 4),
        }

# ── Training ──────────────────────────────────────────────────────────────────
class AETrainer:
    def __init__(self, train_dir, out_dir="models"):
        self.device  = auto_device()
        self.out_dir = Path(out_dir); self.out_dir.mkdir(exist_ok=True)
        ds = NormalOnlyDataset(train_dir)
        self.loader = DataLoader(ds, batch_size=AE_CFG["batch_size"], shuffle=True, num_workers=0)
        self.model  = SkeletonAutoencoder(CFG["feat_dim"], CFG["n_frames"], AE_CFG["latent_dim"]).to(self.device)
        self.optim  = torch.optim.Adam(self.model.parameters(), lr=AE_CFG["lr"])
        self.losses = []

    def train(self):
        best_loss, patience_ctr = float("inf"), 0
        print(f"\n→ Training Autoencoder on NORMAL sequences only | {AE_CFG['epochs']} epochs")
        for ep in range(1, AE_CFG["epochs"] + 1):
            epoch_loss = 0
            self.model.train()
            for seqs in self.loader:
                seqs = seqs.to(self.device)
                recon, _ = self.model(seqs)
                loss = F.mse_loss(recon, seqs)
                self.optim.zero_grad(); loss.backward(); self.optim.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(self.loader)
            self.losses.append(avg)
            print(f"Ep {ep:03d} | Recon Loss: {avg:.6f}")
            if avg < best_loss:
                best_loss = avg
                torch.save(self.model.state_dict(), self.out_dir / "autoencoder_best.pth")
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= AE_CFG["patience"]:
                    print(f"→ Early stopping"); break
        print(f"\n✓ Autoencoder trained | Best loss: {best_loss:.6f}")

    def calibrate_threshold(self, val_dir):
        """Find optimal reconstruction error threshold on val set."""
        print("\n→ Calibrating anomaly threshold on validation set...")
        ds  = SkeletonDataset(val_dir)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        self.model.eval()
        errors, labels = [], []
        with torch.no_grad():
            for seqs, lbls in loader:
                err = self.model.reconstruction_error(seqs.to(self.device))
                errors.extend(err.cpu().tolist())
                labels.extend(lbls.tolist())

        errors  = np.array(errors)
        labels  = np.array(labels)
        # Optimal threshold = mean + 2*std of normal errors
        normal_errors = errors[labels == 0]
        threshold = normal_errors.mean() + 2 * normal_errors.std()
        auc = roc_auc_score(labels, errors)
        print(f"✓ Optimal threshold: {threshold:.6f}")
        print(f"✓ AUC-ROC: {auc:.4f}")

        cfg_out = {"ae_threshold": float(threshold), "auc": auc}
        with open(self.out_dir / "ae_threshold.json", "w") as f:
            json.dump(cfg_out, f, indent=2)
        return threshold

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",     action="store_true")
    ap.add_argument("--threshold", action="store_true")
    ap.add_argument("--train-dir", default="data/processed/train")
    ap.add_argument("--val-dir",   default="data/processed/val")
    ap.add_argument("--out-dir",   default="models")
    args = ap.parse_args()
    trainer = AETrainer(args.train_dir, args.out_dir)
    if args.train:
        trainer.train()
    if args.threshold:
        trainer.calibrate_threshold(args.val_dir)

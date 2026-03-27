"""
evaluate/evaluate_model.py
==========================
Comprehensive evaluation: Accuracy, Precision, Recall, F1, AUC-ROC,
Confusion Matrix, FPS benchmark, and Grad-CAM / attention visualisation.

Usage:
    python evaluate/evaluate_model.py \
        --clf_model models/best_lstm.pt \
        --ae_model  models/anomaly_scorer \
        --dataset   data/datasets/RWF-2000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from tqdm import tqdm

from modules.dataset_loader import get_dataloaders
from modules.classifier import build_model
from modules.autoencoder import SequenceAutoencoder, AnomalyScorer


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def load_classifier(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]
    feat_dim = ckpt["feat_dim"]
    model_kwargs = (
        {"hidden_dim": cfg.get("hidden_dim", 256), "dropout": cfg.get("dropout", 0.4)}
        if cfg["arch"] == "lstm"
        else {"d_model": cfg.get("d_model", 128), "dropout": cfg.get("dropout", 0.3)}
    )
    model = build_model(cfg["arch"], input_dim=feat_dim, device=device, **model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feat_dim, cfg


def evaluate_classifier(model, loader, device):
    all_preds, all_labels, all_probs, all_attns = [], [], [], []
    t0 = time.perf_counter()
    n_frames = 0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", ncols=80):
            x = x.to(device)
            logits, attn = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            n_frames += len(x)
            if attn is not None:
                all_attns.append(attn.cpu().numpy())

    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc":   roc_auc_score(all_labels, all_probs),
        "fps":       fps,
        "n_samples": len(all_labels),
    }
    return metrics, all_labels, all_preds, all_probs, all_attns


def plot_confusion_matrix(labels, preds, save_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NonFight", "Fight"],
                yticklabels=["NonFight", "Fight"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {save_path}")


def plot_roc_curve(labels, probs, auc: float, save_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0D7377", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve → {save_path}")


def print_metrics_table(metrics: dict, arch: str):
    print(f"\n{'═'*50}")
    print(f"  Evaluation Results — {arch.upper()}")
    print(f"{'═'*50}")
    rows = [
        ("Accuracy",    f"{metrics['accuracy']:.4f}",  "≥ 0.90 ✓" if metrics['accuracy'] >= 0.90 else "< 0.90"),
        ("Precision",   f"{metrics['precision']:.4f}", "≥ 0.88 ✓" if metrics['precision'] >= 0.88 else "< 0.88"),
        ("Recall",      f"{metrics['recall']:.4f}",    "≥ 0.90 ✓" if metrics['recall'] >= 0.90 else "< 0.90"),
        ("F1-Score",    f"{metrics['f1']:.4f}",        "≥ 0.89 ✓" if metrics['f1'] >= 0.89 else "< 0.89"),
        ("AUC-ROC",     f"{metrics['auc_roc']:.4f}",   "≥ 0.92 ✓" if metrics['auc_roc'] >= 0.92 else "< 0.92"),
        ("FPS",         f"{metrics['fps']:.1f}",       "≥ 15 FPS ✓" if metrics['fps'] >= 15 else "< 15 FPS"),
        ("Samples",     str(metrics['n_samples']),     ""),
    ]
    for name, val, target in rows:
        print(f"  {name:<14} {val:<12} {target}")
    print(f"{'═'*50}\n")


def main(args):
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ── Load classifier ────────────────────────────────────────────────────
    print(f"Loading classifier from {args.clf_model}...")
    clf, feat_dim, cfg = load_classifier(args.clf_model, device)
    arch = cfg["arch"]

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_loader, _ = get_dataloaders(
        args.dataset, batch_size=32, seq_len=cfg.get("seq_len", 30)
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics, labels, preds, probs, attns = evaluate_classifier(clf, val_loader, device)
    print_metrics_table(metrics, arch)

    print(classification_report(labels, preds,
                                  target_names=["NonFight", "Fight"], digits=4))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, results_dir / f"confusion_matrix_{arch}.png")
    plot_roc_curve(labels, probs, metrics["auc_roc"], results_dir / f"roc_curve_{arch}.png")

    # ── Attention heatmap (LSTM) ───────────────────────────────────────────
    if attns and arch == "lstm":
        attn_mean = np.concatenate(attns, axis=0).mean(axis=0)   # (T,)
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(len(attn_mean)), attn_mean, color="#0D7377")
        ax.set_xlabel("Frame index", fontsize=11)
        ax.set_ylabel("Attention weight", fontsize=11)
        ax.set_title("Average Attention Weights (XAI)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(results_dir / f"attention_weights_{arch}.png", dpi=150)
        plt.close(fig)
        print(f"  Attention heatmap → results/attention_weights_{arch}.png")

    # ── Autoencoder evaluation ─────────────────────────────────────────────
    if args.ae_model:
        print(f"\nLoading AnomalyScorer from {args.ae_model}...")
        ae = SequenceAutoencoder(input_dim=feat_dim, seq_len=cfg.get("seq_len", 30))
        scorer = AnomalyScorer(ae, device=device)
        scorer.load(args.ae_model)

        ae_preds, ae_scores = [], []
        for x, y in tqdm(val_loader, desc="AE Eval", ncols=80):
            scores = scorer.score(x)
            ae_preds.extend((scores >= 0.5).astype(int).tolist())
            ae_scores.extend(scores.tolist())

        print("\nAutoencoder anomaly detection:")
        print(f"  Accuracy: {accuracy_score(labels, ae_preds):.4f}")
        print(f"  F1:       {f1_score(labels, ae_preds, zero_division=0):.4f}")
        try:
            print(f"  AUC-ROC:  {roc_auc_score(labels, ae_scores):.4f}")
        except Exception:
            pass

    # ── Save metrics JSON ─────────────────────────────────────────────────
    out_path = results_dir / f"metrics_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nMetrics saved → {out_path}")
    print("Evaluation complete ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_model", required=True, help="Path to best_*.pt")
    parser.add_argument("--ae_model",  default=None,  help="Path to anomaly_scorer (without extension)")
    parser.add_argument("--dataset",   default="data/datasets/RWF-2000")
    args = parser.parse_args()
    main(args)

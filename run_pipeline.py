"""
run_pipeline.py
===============
End-to-end pipeline runner. Runs all steps in order:
  1. Dataset download check
  2. Pose pre-extraction (optional)
  3. Classifier training
  4. Autoencoder training
  5. Evaluation + metrics report
  6. Launch dashboard (optional)

Usage:
    # Full pipeline
    python run_pipeline.py

    # Skip to evaluation (models already trained)
    python run_pipeline.py --skip_train

    # Train only, no dashboard
    python run_pipeline.py --no_dashboard

    # Quick smoke test (5 epochs, small batch)
    python run_pipeline.py --quick
"""

import sys
import os
import subprocess
import argparse
import time
import yaml
from pathlib import Path

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(msg, colour=CYAN):
    print(f"\n{colour}{BOLD}{'═'*58}")
    print(f"  {msg}")
    print(f"{'═'*58}{RESET}")

def step(n, msg):
    print(f"\n{YELLOW}{BOLD}[Step {n}]{RESET} {msg}")

def ok(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠ {msg}{RESET}")

def run(cmd, check=True):
    """Run a shell command and stream output."""
    print(f"  {CYAN}$ {cmd}{RESET}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"{RED}Command failed (exit {result.returncode}){RESET}")
        sys.exit(result.returncode)
    return result.returncode


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_datasets(cfg):
    """Return True if training data exists."""
    root = Path(cfg["paths"]["dataset_root"])
    fight_dir = root / "train" / "Fight"
    return fight_dir.exists() and any(fight_dir.iterdir())


def count_videos(cfg):
    root = Path(cfg["paths"]["dataset_root"])
    total = 0
    for p in root.rglob("*.avi"):
        total += 1
    for p in root.rglob("*.mp4"):
        total += 1
    return total


def main(args):
    banner("Violence Detection — Full Pipeline Runner", GREEN)
    cfg = load_config()

    dataset_root = cfg["paths"]["dataset_root"]
    models_dir   = cfg["paths"]["models_dir"]
    clf_arch     = cfg["classifier"]["arch"]
    Path(models_dir).mkdir(exist_ok=True)
    Path(cfg["paths"]["results_dir"]).mkdir(exist_ok=True)

    t_start = time.time()

    # ── Step 1: Dataset check ─────────────────────────────────────────────────
    step(1, "Dataset check")
    if check_datasets(cfg):
        n = count_videos(cfg)
        ok(f"Dataset found: {n} videos in {dataset_root}")
    else:
        warn("Dataset not found — running downloader...")
        run("python data/download_datasets.py --dataset all")
        if not check_datasets(cfg):
            print(f"\n{RED}Dataset still not found. "
                  f"Please download manually per README.md{RESET}")
            sys.exit(1)

    # ── Step 2: Pose pre-extraction ───────────────────────────────────────────
    step(2, "Pose pre-extraction (cache .npy files)")
    npy_count = len(list(Path(dataset_root).rglob("*.npy")))
    if npy_count > 10:
        ok(f"Found {npy_count} cached .npy files — skipping extraction")
    elif args.quick:
        warn("Quick mode — skipping pre-extraction (will extract on-the-fly)")
    else:
        warn("No cached features found — running pre-extraction (one-time, ~10-30 min)...")
        run(f"python modules/02_dataset_loader.py --root {dataset_root} --preextract")
        ok("Pose features cached")

    # ── Step 3: Train classifier ──────────────────────────────────────────────
    clf_model = Path(models_dir) / f"best_{clf_arch}.pt"
    if not args.skip_train:
        step(3, f"Training {clf_arch.upper()} classifier")
        epochs = 3 if args.quick else cfg["training"]["epochs"]
        batch  = 8 if args.quick else cfg["training"]["batch_size"]
        run(
            f"python train/train_lstm.py "
            f"--arch {clf_arch} "
            f"--dataset {dataset_root} "
            f"--epochs {epochs} "
            f"--batch_size {batch}"
        )
        ok(f"Classifier saved → {clf_model}")
    else:
        if clf_model.exists():
            ok(f"Using existing classifier: {clf_model}")
        else:
            print(f"{RED}No classifier found at {clf_model}. "
                  f"Remove --skip_train to train.{RESET}")
            sys.exit(1)

    # ── Step 4: Train autoencoder ─────────────────────────────────────────────
    ae_model = Path(models_dir) / "anomaly_scorer_calibration.json"
    if not args.skip_train:
        step(4, "Training Autoencoder (anomaly detection)")
        epochs = 3 if args.quick else cfg["autoencoder"]["epochs"]
        batch  = 8 if args.quick else cfg["training"]["batch_size"]
        run(
            f"python train/train_autoencoder.py "
            f"--dataset {dataset_root} "
            f"--epochs {epochs} "
            f"--batch_size {batch}"
        )
        ok(f"Autoencoder + AnomalyScorer saved → {models_dir}/")
    else:
        if ae_model.exists():
            ok(f"Using existing AnomalyScorer: {ae_model}")
        else:
            warn("No AnomalyScorer found — evaluation will skip anomaly metrics")

    # ── Step 5: Evaluation ────────────────────────────────────────────────────
    step(5, "Evaluation — computing metrics, plots, XAI")
    ae_flag = (f"--ae_model {models_dir}/anomaly_scorer"
               if ae_model.exists() else "")
    run(
        f"python evaluate/evaluate_model.py "
        f"--clf_model {clf_model} "
        f"{ae_flag} "
        f"--dataset {dataset_root}",
        check=False
    )
    ok("Evaluation complete — check results/ folder")

    # ── Step 6: Print summary ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"Pipeline Complete  ({elapsed/60:.1f} min)", GREEN)

    results_dir = Path(cfg["paths"]["results_dir"])
    metrics_file = results_dir / f"metrics_{clf_arch}.json"
    if metrics_file.exists():
        import json
        with open(metrics_file) as f:
            m = json.load(f)
        tgts = cfg["evaluation"]["targets"]
        print(f"\n  {'Metric':<14} {'Value':>8}   {'Target':>8}   {'Status'}")
        print(f"  {'─'*50}")
        for k, tgt in tgts.items():
            val = m.get(k.replace("-", "_"), m.get(k))
            if val is None:
                continue
            passed = val >= tgt
            icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {k:<14} {val:>8.4f}   {tgt:>8.4f}   {icon}")

    print(f"\n  Models:  {models_dir}/")
    print(f"  Results: {results_dir}/")
    print(f"  Logs:    {cfg['paths']['logs_dir']}/")

    # ── Step 7: Launch dashboard ──────────────────────────────────────────────
    if not args.no_dashboard and not args.quick:
        step(7, "Launching officer dashboard")
        port = cfg["dashboard"]["port"]
        src  = cfg["dashboard"]["source"]
        ae_flag2 = (f"--ae_model {models_dir}/anomaly_scorer"
                    if ae_model.exists() else "")
        print(f"\n  {GREEN}Dashboard starting at http://localhost:{port}{RESET}")
        print(f"  Press Ctrl+C to stop\n")
        run(
            f"python dashboard/app.py "
            f"--clf_model {clf_model} "
            f"{ae_flag2} "
            f"--source {src} "
            f"--port {port}",
            check=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full violence detection pipeline"
    )
    parser.add_argument("--skip_train",   action="store_true",
                        help="Skip training, use existing models")
    parser.add_argument("--no_dashboard", action="store_true",
                        help="Skip dashboard launch")
    parser.add_argument("--quick",        action="store_true",
                        help="Smoke test: 3 epochs, small batch, no pre-extraction")
    args = parser.parse_args()
    main(args)

"""
Dataset Downloader for Violence Detection Thesis
Datasets:
  1. RWF-2000  — via Hugging Face Hub (DanJoshua/RWF-2000)
  2. Hockey Fight — via direct URL / Kaggle
  3. CCTV-Fights  — via Kaggle
Run: python data/download_datasets.py [--dataset rwf2000|hockey|all]
"""
import os, sys, argparse, zipfile, shutil
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def banner(msg):
    print(f"\n{'='*55}\n  {msg}\n{'='*55}")

def check_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError:
        print("✗ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

# ── Dataset 1: RWF-2000 via Hugging Face ─────────────────────────────────────
def download_rwf2000():
    banner("Downloading RWF-2000 (Hugging Face)")
    snapshot_download = check_huggingface_hub()
    dest = DATA_DIR / "RWF-2000"
    if dest.exists():
        print(f"✓ RWF-2000 already exists at {dest}")
        return
    print("→ Downloading ~12 GB from HuggingFace (DanJoshua/RWF-2000)...")
    print("→ This may take 20-40 minutes depending on your connection.")
    try:
        from huggingface_hub import snapshot_download as sd
        path = sd(
            repo_id="DanJoshua/RWF-2000",
            repo_type="dataset",
            local_dir=str(dest),
            ignore_patterns=["*.md", "*.json"]
        )
        print(f"✓ RWF-2000 downloaded to {path}")
        print_rwf2000_structure(dest)
    except Exception as e:
        print(f"✗ HuggingFace download failed: {e}")
        print("\n→ ALTERNATIVE — Request access directly from authors:")
        print("   1. Email: smiip.lab@gmail.com")
        print("   2. GitHub: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection")
        print("   3. Subject: 'RWF-2000 Dataset Access Request - Academic Research'")

def print_rwf2000_structure(dest):
    print("\n→ Expected structure:")
    print("""  RWF-2000/
  ├── train/
  │   ├── Fight/      (750 videos)
  │   └── NonFight/   (750 videos)
  └── val/
      ├── Fight/      (250 videos)
      └── NonFight/   (250 videos)""")

# ── Dataset 2: Hockey Fight ───────────────────────────────────────────────────
def download_hockey():
    banner("Downloading Hockey Fight Dataset")
    dest = DATA_DIR / "HockeyFight"
    if dest.exists():
        print(f"✓ HockeyFight already exists at {dest}")
        return
    try:
        import subprocess
        print("→ Attempting Kaggle download (requires Kaggle API token)...")
        print("   Setup: https://www.kaggle.com/docs/api")
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "yassershrief/hockey-fight-vidoes",
            "--path", str(dest), "--unzip"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Hockey Fight downloaded to {dest}")
        else:
            raise RuntimeError(result.stderr)
    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("\n→ MANUAL DOWNLOAD:")
        print("   1. Go to https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes")
        print("   2. Click Download")
        print(f"   3. Extract to: {dest}/")
        print("   4. Structure should be:")
        print("       HockeyFight/")
        print("         fi/   (fight clips)")
        print("         no/   (non-fight clips)")
    print_hockey_structure(dest)

def print_hockey_structure(dest):
    print("\n→ Expected structure:")
    print("""  HockeyFight/
  ├── fi/   (500 fight videos)
  └── no/   (500 non-fight videos)""")

# ── Dataset 3: CCTV-Fights ────────────────────────────────────────────────────
def download_cctv_fights():
    banner("CCTV-Fights Dataset")
    print("→ CCTV-Fights requires manual download from Kaggle:")
    print("   URL: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset")
    dest = DATA_DIR / "CCTV-Fights"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"→ Extract to: {dest}/")

# ── Verify Datasets ───────────────────────────────────────────────────────────
def verify_datasets():
    banner("Verifying Downloaded Datasets")
    checks = {
        "RWF-2000/train/Fight": "RWF-2000 train/Fight",
        "RWF-2000/train/NonFight": "RWF-2000 train/NonFight",
        "RWF-2000/val/Fight": "RWF-2000 val/Fight",
        "RWF-2000/val/NonFight": "RWF-2000 val/NonFight",
        "HockeyFight/fi": "HockeyFight/fi",
        "HockeyFight/no": "HockeyFight/no",
    }
    all_ok = True
    for rel_path, label in checks.items():
        full = DATA_DIR / rel_path
        if full.exists():
            count = len(list(full.glob("*.avi")) + list(full.glob("*.mp4")))
            print(f"  ✓ {label:<35} {count} videos")
        else:
            print(f"  ✗ {label:<35} NOT FOUND")
            all_ok = False
    print()
    if all_ok:
        print("✓ All datasets verified and ready for training!")
    else:
        print("⚠  Some datasets missing — see instructions above to download manually.")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download violence detection datasets")
    parser.add_argument("--dataset", choices=["rwf2000", "hockey", "cctv", "all"], default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_datasets()
        sys.exit(0)

    if args.dataset in ("rwf2000", "all"):
        download_rwf2000()
    if args.dataset in ("hockey", "all"):
        download_hockey()
    if args.dataset in ("cctv", "all"):
        download_cctv_fights()

    verify_datasets()

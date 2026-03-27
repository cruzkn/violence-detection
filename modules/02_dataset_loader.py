"""
modules/02_dataset_loader.py
============================
Module 2: PyTorch Dataset — loads video files, runs pose estimation,
returns (sequence_tensor, label) pairs for training.

Caches extracted keypoint sequences as .npy files to avoid re-running
inference on every epoch.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import random


class ViolenceDataset(Dataset):
    """
    Loads Fight/NonFight video clips, extracts pose sequences via
    YOLOv8-Pose, and caches results as .npy arrays.

    Args:
        root_dir:    Path to dataset folder (must contain Fight/ NonFight/)
        split:       'train' or 'val'
        seq_len:     Number of frames sampled per clip (default 30)
        max_persons: Max persons tracked per frame (default 2)
        cache:       If True, cache .npy arrays alongside videos
        augment:     If True, apply temporal + spatial augmentation
        use_velocity:If True, append frame-to-frame velocity features
    """

    LABEL_MAP = {"Fight": 1, "NonFight": 0, "fight": 1, "nonfight": 0,
                 "Violence": 1, "NonViolence": 0}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        seq_len: int = 30,
        max_persons: int = 2,
        cache: bool = True,
        augment: bool = False,
        use_velocity: bool = True,
    ):
        self.root = Path(root_dir) / split
        self.seq_len = seq_len
        self.max_persons = max_persons
        self.cache = cache
        self.augment = augment
        self.use_velocity = use_velocity

        # Feature dimension: max_persons * 17 keypoints * 3 (x,y,conf)
        self.feat_dim = max_persons * 51
        if use_velocity:
            self.feat_dim *= 2         # append velocity

        self.samples = self._scan_directory()
        print(f"[Dataset] {split}: {len(self.samples)} clips | "
              f"Fight={sum(1 for _,l in self.samples if l==1)} | "
              f"NonFight={sum(1 for _,l in self.samples if l==0)}")

    def _scan_directory(self):
        samples = []
        extensions = {".avi", ".mp4", ".mov", ".mkv"}
        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            label = self.LABEL_MAP.get(label_name)
            if label is None:
                print(f"  [warn] Unknown label folder: {label_name} — skipping")
                continue
            for f in label_dir.iterdir():
                if f.suffix.lower() in extensions:
                    samples.append((f, label))
        random.shuffle(samples)
        return samples

    def _cache_path(self, video_path: Path) -> Path:
        return video_path.with_suffix(".npy")

    def _load_or_extract(self, video_path: Path) -> np.ndarray:
        cache_path = self._cache_path(video_path)

        # Return cached array if available
        if self.cache and cache_path.exists():
            return np.load(cache_path)

        # Extract using PoseEstimator
        from modules.pose_estimator import PoseEstimator
        if not hasattr(self, "_pe"):
            self._pe = PoseEstimator(model_size="n")

        seq = self._pe.extract_from_video(
            str(video_path),
            max_persons=self.max_persons,
            seq_len=self.seq_len,
        )

        if self.use_velocity:
            from modules.pose_estimator import PoseEstimator as PE
            seq = PE.compute_velocity(seq)

        if self.cache:
            np.save(cache_path, seq)
        return seq

    # ── Augmentation ──────────────────────────────────────────────────────────
    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Light-weight temporal + spatial jitter."""
        # Temporal shift: roll sequence by ±3 frames
        shift = random.randint(-3, 3)
        seq = np.roll(seq, shift, axis=0)

        # Horizontal flip: negate x-coordinates (every 3rd value starting at 0)
        if random.random() < 0.5:
            seq_cp = seq.copy()
            seq_cp[:, 0::3] = 1.0 - seq_cp[:, 0::3]
            seq = seq_cp

        # Gaussian noise
        seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        return seq.clip(0, 1)

    # ── Dataset API ───────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        seq = self._load_or_extract(video_path)        # (T, F)

        # Ensure consistent shape
        if seq.shape[0] != self.seq_len:
            seq = self._pad_or_trim(seq)

        if self.augment and label == 1:                # augment fight clips more
            seq = self._augment(seq)

        x = torch.tensor(seq, dtype=torch.float32)    # (T, F)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def _pad_or_trim(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T >= self.seq_len:
            return seq[:self.seq_len]
        pad = np.zeros((self.seq_len - T, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


# ── Pre-extraction helper ──────────────────────────────────────────────────────
def preextract_all(dataset_root: str, splits=("train", "val"), seq_len=30):
    """
    Run pose estimation on all videos and cache as .npy.
    Call this once before training to avoid on-the-fly extraction.
    """
    from modules.pose_estimator import PoseEstimator
    pe = PoseEstimator(model_size="n")

    root = Path(dataset_root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        videos = list(split_dir.rglob("*.avi")) + list(split_dir.rglob("*.mp4"))
        print(f"\n[PreExtract] {split}: {len(videos)} videos")
        for vp in tqdm(videos, desc=split):
            cache = vp.with_suffix(".npy")
            if cache.exists():
                continue
            try:
                seq = pe.extract_from_video(str(vp), seq_len=seq_len)
                seq = PoseEstimator.compute_velocity(seq)
                np.save(cache, seq)
            except Exception as e:
                print(f"  [warn] Failed {vp.name}: {e}")

    print("\n[PreExtract] Complete ✓")


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(
    dataset_root: str,
    batch_size: int = 16,
    seq_len: int = 30,
    num_workers: int = 0,        # 0 = main process (safer on macOS)
):
    """Returns (train_loader, val_loader, feature_dim)."""
    train_ds = ViolenceDataset(dataset_root, split="train", seq_len=seq_len, augment=True)
    val_ds   = ViolenceDataset(dataset_root, split="val",   seq_len=seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, train_ds.feat_dim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root dir")
    parser.add_argument("--preextract", action="store_true")
    args = parser.parse_args()

    if args.preextract:
        preextract_all(args.root)
    else:
        tl, vl, fdim = get_dataloaders(args.root, batch_size=4)
        x, y = next(iter(tl))
        print(f"Batch shape: {x.shape}  Labels: {y}")
        print(f"Feature dim: {fdim}")

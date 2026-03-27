"""
modules/01_pose_estimator.py
============================
Module 1: Real-time human pose estimation using YOLOv8-Pose.

Extracts 17-keypoint COCO skeleton from each detected person in a video
frame. Output is a normalised (x, y, confidence) vector per person,
ready to feed into the LSTM/Transformer classifier.

Usage:
    from modules.pose_estimator import PoseEstimator
    pe = PoseEstimator()
    keypoints = pe.extract_from_frame(frame)   # numpy (N_persons, 51)
    pe.visualise(frame, keypoints)              # draws skeleton overlay
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time

# COCO 17-keypoint skeleton pairs for visualisation
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

VIOLENCE_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12]  # arms, wrists, hips — most discriminative


class PoseEstimator:
    """
    Wraps YOLOv8-Pose for multi-person keypoint extraction.

    Args:
        model_size: 'n' (nano, fastest), 's', 'm', 'l', 'x' (largest)
        conf_threshold: minimum person detection confidence
        device: 'cpu', 'cuda', or 'mps' (Apple Silicon)
    """

    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.4,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO
        import torch

        self.conf = conf_threshold
        self.model_name = f"yolov8{model_size}-pose.pt"

        # Auto-select best device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"          # Apple Silicon GPU
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[PoseEstimator] Loading {self.model_name} on {self.device}...")
        self.model = YOLO(self.model_name)
        print(f"[PoseEstimator] Ready ✓")

    # ── Core extraction ────────────────────────────────────────────────────────
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8-Pose on one BGR frame.

        Returns:
            keypoints: float32 array of shape (N_persons, 51)
                       Each row = [x0,y0,c0, x1,y1,c1, ..., x16,y16,c16]
                       Coordinates are normalised to [0, 1].
        """
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        all_kps = []
        for r in results:
            if r.keypoints is None or len(r.keypoints.data) == 0:
                continue
            for person_kps in r.keypoints.data:  # (17, 3) tensor
                kp = person_kps.cpu().numpy().astype(np.float32)
                # Normalise x,y to [0,1]; leave confidence as-is
                kp[:, 0] /= w
                kp[:, 1] /= h
                all_kps.append(kp.flatten())      # (51,)

        return np.array(all_kps, dtype=np.float32) if all_kps else np.zeros((0, 51), dtype=np.float32)

    def extract_from_video(
        self,
        video_path: str,
        max_persons: int = 2,
        seq_len: int = 30,
    ) -> np.ndarray:
        """
        Extract pose sequence from a video file.

        Returns:
            sequence: float32 array of shape (seq_len, max_persons * 51)
                      Zero-padded if fewer persons detected than max_persons.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)

        sequence = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                sequence.append(np.zeros(max_persons * 51, dtype=np.float32))
                continue

            kps = self.extract_from_frame(frame)            # (N, 51)
            # Keep top-max_persons persons (sorted by detection confidence)
            if len(kps) == 0:
                merged = np.zeros(max_persons * 51, dtype=np.float32)
            elif len(kps) >= max_persons:
                merged = kps[:max_persons].flatten()
            else:
                pad = np.zeros((max_persons - len(kps), 51), dtype=np.float32)
                merged = np.vstack([kps, pad]).flatten()

            sequence.append(merged)

        cap.release()
        return np.array(sequence, dtype=np.float32)         # (seq_len, max_persons*51)

    # ── Velocity features ─────────────────────────────────────────────────────
    @staticmethod
    def compute_velocity(sequence: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame velocity of keypoints.
        Appends velocity as additional features.

        Input/Output shape: (T, F) → (T, 2*F)
        """
        velocity = np.zeros_like(sequence)
        velocity[1:] = sequence[1:] - sequence[:-1]
        return np.concatenate([sequence, velocity], axis=-1)

    # ── Visualisation ─────────────────────────────────────────────────────────
    def visualise(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        label: Optional[str] = None,
        color: tuple = (0, 255, 170),
    ) -> np.ndarray:
        """
        Draw skeleton overlays on frame. Returns annotated frame.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        for person_kps in keypoints:
            kp = person_kps.reshape(17, 3)

            # Draw joints
            for i, (x, y, c) in enumerate(kp):
                if c > 0.3:
                    px, py = int(x * w), int(y * h)
                    # Highlight violence-relevant keypoints
                    radius = 6 if i in VIOLENCE_KEYPOINTS else 4
                    kp_color = (0, 120, 255) if i in VIOLENCE_KEYPOINTS else color
                    cv2.circle(out, (px, py), radius, kp_color, -1)

            # Draw skeleton connections
            for a, b in SKELETON_PAIRS:
                xa, ya, ca = kp[a]
                xb, yb, cb = kp[b]
                if ca > 0.3 and cb > 0.3:
                    p1 = (int(xa * w), int(ya * h))
                    p2 = (int(xb * w), int(yb * h))
                    cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)

        if label:
            cv2.putText(out, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return out

    # ── Benchmark FPS ─────────────────────────────────────────────────────────
    def benchmark(self, n_frames: int = 100, frame_size: tuple = (640, 480)):
        """Measure inference FPS on dummy frames."""
        dummy = np.random.randint(0, 255, (*frame_size[::-1], 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(n_frames):
            self.extract_from_frame(dummy)
        elapsed = time.perf_counter() - t0
        fps = n_frames / elapsed
        print(f"[PoseEstimator] Benchmark: {fps:.1f} FPS on {self.device} ({frame_size})")
        return fps


# ── CLI demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Video file path or camera index (default: 0 = webcam)")
    parser.add_argument("--model", default="n", choices=["n", "s", "m"],
                        help="YOLOv8-Pose model size")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    pe = PoseEstimator(model_size=args.model)

    if args.benchmark:
        pe.benchmark()
        exit(0)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    print(f"[PoseEstimator] Running on source: {args.source} — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps = pe.extract_from_frame(frame)
        annotated = pe.visualise(frame, kps, label=f"Persons: {len(kps)}")
        cv2.imshow("Pose Estimation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
modules/05_alert_engine.py
==========================
Module 5: Dual-Stream Alert Engine.

Fuses signals from:
  Stream A — Supervised classifier  (BiLSTM or Transformer)
  Stream B — Unsupervised autoencoder anomaly scorer

Uses temporal smoothing (N-frame majority voting) to suppress
transient false positives before issuing an alert.
"""

import time
import json
import collections
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path


@dataclass
class AlertEvent:
    """A single triggered alert."""
    timestamp:    float
    camera_id:    str
    classifier_prob:  float          # P(Fight) from supervised model
    anomaly_score:    float          # from autoencoder [0,1]
    fused_score:      float          # final combined score
    frame_idx:        int
    keypoints:        Optional[np.ndarray] = field(default=None, repr=False)
    confirmed:        bool = False   # set True when officer confirms


class AlertEngine:
    """
    Dual-stream violence alert engine.

    Args:
        classifier:       trained BiLSTM or STTransformer model
        anomaly_scorer:   calibrated AnomalyScorer
        device:           'cpu', 'cuda', or 'mps'
        clf_threshold:    P(Fight) threshold for classifier stream (default 0.65)
        anomaly_weight:   weight for anomaly stream in fusion (0=all-classifier, 1=all-anomaly)
        fused_threshold:  final alert trigger threshold (default 0.60)
        smooth_window:    number of consecutive frames for temporal smoothing
        smooth_min_hits:  minimum positive frames in window to trigger alert
        cooldown_secs:    minimum seconds between consecutive alerts (per camera)
        on_alert:         optional callback function called with AlertEvent
    """

    def __init__(
        self,
        classifier,
        anomaly_scorer,
        device: str = "cpu",
        clf_threshold:    float = 0.65,
        anomaly_weight:   float = 0.35,
        fused_threshold:  float = 0.60,
        smooth_window:    int   = 5,
        smooth_min_hits:  int   = 3,
        cooldown_secs:    float = 10.0,
        on_alert: Optional[Callable[[AlertEvent], None]] = None,
    ):
        self.classifier    = classifier
        self.scorer        = anomaly_scorer
        self.device        = device
        self.clf_thresh    = clf_threshold
        self.ano_weight    = anomaly_weight
        self.clf_weight    = 1.0 - anomaly_weight
        self.fused_thresh  = fused_threshold
        self.smooth_window = smooth_window
        self.smooth_hits   = smooth_min_hits
        self.cooldown      = cooldown_secs
        self.on_alert      = on_alert

        # Per-camera state
        self._windows:       dict = {}   # camera_id → deque of bool
        self._last_alert:    dict = {}   # camera_id → timestamp
        self._alert_log:     list = []
        self._frame_counts:  dict = collections.defaultdict(int)

    # ── Core inference ────────────────────────────────────────────────────────
    def process_sequence(
        self,
        sequence: np.ndarray,
        camera_id: str = "cam_01",
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[AlertEvent]:
        """
        Process one pose sequence and optionally fire an alert.

        Args:
            sequence:  numpy array (T, F) — one temporal window of keypoints
            camera_id: camera identifier string
            keypoints: raw keypoints for visualisation (optional)

        Returns:
            AlertEvent if alert fires, else None
        """
        self._frame_counts[camera_id] += 1
        frame_idx = self._frame_counts[camera_id]

        # ── Stream A: Classifier ─────────────────────────────────────────────
        clf_score = self._run_classifier(sequence)       # P(Fight) in [0,1]

        # ── Stream B: Anomaly ────────────────────────────────────────────────
        ano_score = self._run_anomaly(sequence)          # [0,1]

        # ── Fusion ───────────────────────────────────────────────────────────
        fused = self.clf_weight * clf_score + self.ano_weight * ano_score

        # ── Temporal smoothing ───────────────────────────────────────────────
        is_positive = fused >= self.fused_thresh
        window = self._windows.setdefault(
            camera_id, collections.deque(maxlen=self.smooth_window)
        )
        window.append(is_positive)

        hit_count = sum(window)
        should_alert = (
            len(window) == self.smooth_window and
            hit_count >= self.smooth_hits
        )

        # ── Cooldown ─────────────────────────────────────────────────────────
        now = time.time()
        last = self._last_alert.get(camera_id, 0)
        if should_alert and (now - last) >= self.cooldown:
            self._last_alert[camera_id] = now
            event = AlertEvent(
                timestamp=now,
                camera_id=camera_id,
                classifier_prob=float(clf_score),
                anomaly_score=float(ano_score),
                fused_score=float(fused),
                frame_idx=frame_idx,
                keypoints=keypoints,
            )
            self._alert_log.append(event)
            if self.on_alert:
                self.on_alert(event)
            return event

        return None

    # ── Private helpers ───────────────────────────────────────────────────────
    def _run_classifier(self, sequence: np.ndarray) -> float:
        """Returns P(Fight) from the supervised classifier."""
        self.classifier.eval()
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, _ = self.classifier(x)
            prob = F.softmax(logits, dim=-1)[0, 1].item()   # P(Fight)
        return prob

    def _run_anomaly(self, sequence: np.ndarray) -> float:
        """Returns normalised anomaly score [0,1]."""
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        scores = self.scorer.score(x)
        return float(np.clip(scores[0], 0, 1))

    # ── Log management ────────────────────────────────────────────────────────
    def get_alerts(self, camera_id: Optional[str] = None):
        """Return alert log, optionally filtered by camera."""
        if camera_id:
            return [a for a in self._alert_log if a.camera_id == camera_id]
        return list(self._alert_log)

    def confirm_alert(self, alert: AlertEvent):
        alert.confirmed = True

    def export_log(self, path: str):
        """Export alert log to JSON."""
        data = [
            {
                "timestamp": a.timestamp,
                "camera_id": a.camera_id,
                "classifier_prob": a.classifier_prob,
                "anomaly_score":   a.anomaly_score,
                "fused_score":     a.fused_score,
                "frame_idx":       a.frame_idx,
                "confirmed":       a.confirmed,
            }
            for a in self._alert_log
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AlertEngine] Exported {len(data)} alerts to {path}")

    def reset(self, camera_id: Optional[str] = None):
        """Reset state for one or all cameras."""
        if camera_id:
            self._windows.pop(camera_id, None)
            self._last_alert.pop(camera_id, None)
        else:
            self._windows.clear()
            self._last_alert.clear()
        print(f"[AlertEngine] State reset for: {camera_id or 'ALL'}")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.classifier import build_model
    from modules.autoencoder import SequenceAutoencoder, AnomalyScorer

    T, F = 30, 204
    clf = build_model("lstm", input_dim=F)
    ae  = SequenceAutoencoder(input_dim=F, seq_len=T)
    scorer = AnomalyScorer(ae, device="cpu")
    scorer.threshold = 0.05   # dummy calibration

    def on_alert(event: AlertEvent):
        print(f"  🚨 ALERT | cam={event.camera_id} | "
              f"clf={event.classifier_prob:.2f} | "
              f"ano={event.anomaly_score:.2f} | "
              f"fused={event.fused_score:.2f}")

    engine = AlertEngine(clf, scorer, on_alert=on_alert, smooth_window=3, smooth_min_hits=2)

    print("Simulating 10 sequences (first 5 violent, last 5 normal)...")
    for i in range(10):
        seq = np.random.randn(T, F).astype(np.float32)
        if i < 5:
            seq += 0.5   # shift distribution to simulate violence
        result = engine.process_sequence(seq, camera_id="cam_01")
        print(f"  Frame {i+1}: alert={'YES' if result else 'no'}")

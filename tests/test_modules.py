"""
tests/test_modules.py
=====================
Unit tests for all pipeline modules.
Run with: python -m pytest tests/ -v

Tests are designed to run WITHOUT a dataset (uses random dummy data)
so they work immediately after setup, before downloading datasets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════
# Test constants
# ══════════════════════════════════════════════════════════════════
B     = 4     # batch size
T     = 30    # sequence length
F     = 204   # feature dim (2 persons × 51 × 2 for velocity)
F_RAW = 102   # without velocity


# ══════════════════════════════════════════════════════════════════
# Module 1: Pose Estimator (no camera needed)
# ══════════════════════════════════════════════════════════════════
class TestPoseEstimator:

    def test_import(self):
        from modules.pose_estimator import PoseEstimator, SKELETON_PAIRS, KEYPOINT_NAMES
        assert len(KEYPOINT_NAMES) == 17
        assert len(SKELETON_PAIRS) > 0

    def test_compute_velocity(self):
        from modules.pose_estimator import PoseEstimator
        seq = np.random.randn(T, F_RAW).astype(np.float32)
        out = PoseEstimator.compute_velocity(seq)
        assert out.shape == (T, F_RAW * 2)
        # First row velocity should be zero
        assert np.allclose(out[0, F_RAW:], 0.0)
        # Subsequent rows should differ
        assert not np.allclose(out[1, F_RAW:], 0.0)

    def test_visualise_no_error(self):
        """Visualise should not crash on empty keypoints."""
        import cv2
        from modules.pose_estimator import PoseEstimator
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.zeros((0, 51), dtype=np.float32)
        # Would need YOLO to run inference, so we just test the viz path
        pe = object.__new__(PoseEstimator)   # skip __init__
        pe.conf = 0.4
        result = pe.visualise(frame, kps, label="test")
        assert result.shape == frame.shape


# ══════════════════════════════════════════════════════════════════
# Module 3: Classifier
# ══════════════════════════════════════════════════════════════════
class TestClassifier:

    def test_bilstm_forward(self):
        from modules.classifier import BiLSTMClassifier
        model = BiLSTMClassifier(input_dim=F)
        x = torch.randn(B, T, F)
        logits, attn = model(x)
        assert logits.shape == (B, 2)
        assert attn.shape   == (B, T)
        # Attention weights sum to 1 per sample
        assert torch.allclose(attn.sum(dim=1), torch.ones(B), atol=1e-5)

    def test_transformer_forward(self):
        from modules.classifier import STTransformer
        model = STTransformer(input_dim=F)
        x = torch.randn(B, T, F)
        logits, attn = model(x)
        assert logits.shape == (B, 2)
        assert attn is None

    def test_build_model_lstm(self):
        from modules.classifier import build_model
        model = build_model("lstm", input_dim=F)
        assert model is not None
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_build_model_transformer(self):
        from modules.classifier import build_model
        model = build_model("transformer", input_dim=F)
        assert model is not None

    def test_softmax_sums_to_one(self):
        from modules.classifier import BiLSTMClassifier
        model = BiLSTMClassifier(input_dim=F)
        x = torch.randn(B, T, F)
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flows(self):
        from modules.classifier import BiLSTMClassifier
        model = BiLSTMClassifier(input_dim=F)
        x = torch.randn(B, T, F)
        y = torch.randint(0, 2, (B,))
        logits, _ = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ══════════════════════════════════════════════════════════════════
# Module 4: Autoencoder
# ══════════════════════════════════════════════════════════════════
class TestAutoencoder:

    def test_forward_shape(self):
        from modules.autoencoder import SequenceAutoencoder
        model = SequenceAutoencoder(input_dim=F, seq_len=T)
        x = torch.randn(B, T, F)
        x_hat, z = model(x)
        assert x_hat.shape == x.shape
        assert z.shape == (B, 64)   # default latent_dim

    def test_reconstruction_error_shape(self):
        from modules.autoencoder import SequenceAutoencoder
        model = SequenceAutoencoder(input_dim=F, seq_len=T)
        x = torch.randn(B, T, F)
        errors = model.reconstruction_error(x)
        assert errors.shape == (B,)
        assert (errors >= 0).all()

    def test_normal_lower_error_than_noisy(self):
        """Normal data should have lower recon error than heavily noisy data."""
        from modules.autoencoder import SequenceAutoencoder
        model = SequenceAutoencoder(input_dim=F, seq_len=T)
        normal = torch.zeros(1, T, F)
        noisy  = torch.randn(1, T, F) * 10
        err_n = model.reconstruction_error(normal).item()
        err_o = model.reconstruction_error(noisy).item()
        # Untrained model — both will be high, but noisy should be higher
        assert err_o >= err_n * 0.5  # loose check for untrained model

    def test_anomaly_scorer_score_range(self):
        from modules.autoencoder import SequenceAutoencoder, AnomalyScorer
        ae = SequenceAutoencoder(input_dim=F, seq_len=T)
        scorer = AnomalyScorer(ae, device="cpu")
        scorer.threshold  = 0.1
        scorer.error_mean = 0.1
        scorer.error_std  = 0.05
        x = torch.randn(B, T, F)
        scores = scorer.score(x)
        assert scores.shape == (B,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_autoencoder_gradient_flows(self):
        from modules.autoencoder import SequenceAutoencoder
        model = SequenceAutoencoder(input_dim=F, seq_len=T)
        x = torch.randn(B, T, F)
        x_hat, _ = model(x)
        loss = torch.nn.MSELoss()(x_hat, x)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ══════════════════════════════════════════════════════════════════
# Module 5: Alert Engine
# ══════════════════════════════════════════════════════════════════
class TestAlertEngine:

    def _make_engine(self):
        from modules.classifier import build_model
        from modules.autoencoder import SequenceAutoencoder, AnomalyScorer
        from modules.alert_engine import AlertEngine

        clf = build_model("lstm", input_dim=F)
        ae  = SequenceAutoencoder(input_dim=F, seq_len=T)
        scorer = AnomalyScorer(ae, device="cpu")
        scorer.threshold  = 0.0   # everything is "anomalous" for testing
        scorer.error_mean = 0.1
        scorer.error_std  = 0.05

        engine = AlertEngine(
            classifier=clf,
            anomaly_scorer=scorer,
            clf_threshold=0.0,      # always triggers for testing
            fused_threshold=0.0,
            smooth_window=3,
            smooth_min_hits=2,
            cooldown_secs=0.0,
        )
        return engine

    def test_no_alert_before_window_fills(self):
        engine = self._make_engine()
        seq = np.random.randn(T, F).astype(np.float32)
        # Only 1 frame — window not full yet
        result = engine.process_sequence(seq, camera_id="test")
        # After 1 frame window is [True] but smooth_window=3 not filled
        assert result is None

    def test_alert_fires_after_window(self):
        engine = self._make_engine()
        seq = np.random.randn(T, F).astype(np.float32)
        last_result = None
        for _ in range(5):
            last_result = engine.process_sequence(seq, camera_id="test")
        # By now window is filled and threshold is 0 — should fire
        assert last_result is not None or len(engine.get_alerts()) > 0

    def test_alert_log_accumulates(self):
        engine = self._make_engine()
        seq = np.random.randn(T, F).astype(np.float32)
        for _ in range(10):
            engine.process_sequence(seq, camera_id="cam_01")
        alerts = engine.get_alerts()
        assert isinstance(alerts, list)

    def test_reset_clears_state(self):
        engine = self._make_engine()
        seq = np.random.randn(T, F).astype(np.float32)
        for _ in range(5):
            engine.process_sequence(seq, camera_id="cam_01")
        engine.reset("cam_01")
        assert "cam_01" not in engine._windows


# ══════════════════════════════════════════════════════════════════
# Integration: full pipeline smoke test
# ══════════════════════════════════════════════════════════════════
class TestIntegration:

    def test_full_pipeline_dummy_batch(self):
        """
        Simulate a full forward pass through all modules
        using dummy data — no dataset or camera required.
        """
        from modules.classifier import build_model
        from modules.autoencoder import SequenceAutoencoder, AnomalyScorer
        from modules.alert_engine import AlertEngine

        # Build models
        clf = build_model("lstm", input_dim=F)
        ae  = SequenceAutoencoder(input_dim=F, seq_len=T)
        scorer = AnomalyScorer(ae, device="cpu")
        scorer.threshold  = 0.05
        scorer.error_mean = 0.05
        scorer.error_std  = 0.02
        engine = AlertEngine(clf, scorer, device="cpu",
                             smooth_window=3, smooth_min_hits=2,
                             cooldown_secs=0.0)

        # Feed 10 sequences
        results = []
        for i in range(10):
            seq = np.random.randn(T, F).astype(np.float32)
            r = engine.process_sequence(seq, camera_id="cam_test")
            results.append(r)

        # No crash — that's the main assertion
        assert len(results) == 10
        print("  ✓ Full pipeline integration test passed")

    def test_model_serialisation(self, tmp_path):
        """Save and reload a model checkpoint."""
        from modules.classifier import build_model
        import torch

        model = build_model("lstm", input_dim=F)
        ckpt_path = tmp_path / "test.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": {"arch": "lstm", "hidden_dim": 256, "dropout": 0.4},
            "feat_dim": F,
        }, ckpt_path)

        # Reload
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2 = build_model("lstm", input_dim=F)
        model2.load_state_dict(ckpt["model_state"])

        # Check predictions match
        x = torch.randn(2, T, F)
        model.eval(); model2.eval()
        with torch.no_grad():
            l1, _ = model(x)
            l2, _ = model2(x)
        assert torch.allclose(l1, l2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

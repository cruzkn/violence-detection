"""
Module 5: Real-Time Alert Dashboard
Flask + SocketIO web dashboard for live CCTV violence detection.
- Shows live camera feed with skeleton overlay
- Real-time violence score gauge
- Alert log with timestamp, confidence, camera ID
- Human-in-the-loop: officer must confirm each alert

Run:
    python dashboard/app.py
Then open: http://localhost:5000
"""
import cv2
import torch
import numpy as np
import json
import time
import base64
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "violence-detection-dashboard-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
camera_feed = None
alert_log   = []
current_score = {"violence_prob": 0.0, "alert": False, "confidence": 0.0, "ae_error": 0.0}
is_running  = False
model_loaded = False

# ── Load Models ───────────────────────────────────────────────────────────────
def load_models():
    global model_loaded, detector
    try:
        from ultralytics import YOLO
        from lstm_classifier import BiLSTMClassifier, CFG, auto_device
        from autoencoder import SkeletonAutoencoder, DualStreamScorer, AE_CFG

        device = auto_device()
        pose_model = YOLO("yolov8n-pose.pt")

        lstm = BiLSTMClassifier(CFG["feat_dim"], CFG["hidden_dim"],
                                CFG["n_layers"], CFG["num_classes"], CFG["dropout"])
        ckpt = MODEL_DIR / "lstm_best.pth"
        if ckpt.exists():
            lstm.load_state_dict(torch.load(ckpt, map_location=device))
            lstm = lstm.to(device)
            print("✓ Loaded LSTM classifier")
        else:
            print(f"⚠  No checkpoint at {ckpt} — using untrained model (demo only)")

        ae = SkeletonAutoencoder(CFG["feat_dim"], CFG["n_frames"], AE_CFG["latent_dim"])
        ae_ckpt = MODEL_DIR / "autoencoder_best.pth"
        ae_thresh = 0.05
        if ae_ckpt.exists():
            ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
            ae = ae.to(device)
            thresh_file = MODEL_DIR / "ae_threshold.json"
            if thresh_file.exists():
                ae_thresh = json.load(open(thresh_file))["ae_threshold"]
            print("✓ Loaded Autoencoder")

        from autoencoder import DualStreamScorer
        scorer = DualStreamScorer(lstm, ae, alpha=0.7, ae_threshold=ae_thresh, device=device)
        model_loaded = True
        print("✓ All models loaded. Dashboard ready.")
        return pose_model, scorer, device
    except Exception as e:
        print(f"⚠  Model load failed: {e}")
        print("   Dashboard running in DEMO mode (random scores)")
        return None, None, None

pose_model, scorer, device = None, None, None

# ── Video Processing Thread ───────────────────────────────────────────────────
def process_stream(source=0):
    global camera_feed, current_score, is_running, alert_log
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"✗ Cannot open camera/video source: {source}")
        return

    WINDOW   = 30           # frames to collect before scoring
    THRESH   = 0.55         # alert threshold
    N_JOINTS = 17
    FEAT_DIM = 51
    buffer   = np.zeros((WINDOW, FEAT_DIM), dtype=np.float32)
    frame_i  = 0
    fps_list = []

    while is_running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):  # video file — loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        annotated = frame.copy()

        # Pose estimation
        if pose_model:
            results = pose_model(frame, verbose=False, conf=0.3)
            annotated = results[0].plot()
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kp = r.keypoints.data[0].cpu().numpy()  # (17,3)
                    kp_norm = kp.copy()
                    kp_norm[:, 0] /= w; kp_norm[:, 1] /= h
                    buffer[frame_i % WINDOW] = kp_norm.flatten()
        else:
            # Demo mode: simulate a score
            buffer[frame_i % WINDOW] = np.random.randn(FEAT_DIM).astype(np.float32) * 0.1

        frame_i += 1

        # Score every WINDOW frames
        if frame_i % WINDOW == 0 and frame_i > 0:
            seq_tensor = torch.from_numpy(buffer.copy()).unsqueeze(0)
            if scorer:
                result = scorer.score(seq_tensor)
            else:
                # Demo mode random score
                prob = float(np.random.beta(1, 3))
                result = {"violence_prob": prob, "ae_error": 0.01,
                          "ae_normalized": 0.0, "final_score": prob,
                          "alert": prob > THRESH, "confidence": prob}

            current_score = result
            socketio.emit("score_update", result)

            if result["alert"]:
                alert = {
                    "id": len(alert_log) + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": result["confidence"],
                    "violence_prob": result["violence_prob"],
                    "camera": f"CAM-{str(source).zfill(2)}",
                    "confirmed": False,
                    "dismissed": False,
                }
                alert_log.append(alert)
                socketio.emit("new_alert", alert)

        # FPS overlay
        fps_list.append(1 / (time.time() - t0 + 1e-6))
        avg_fps = np.mean(fps_list[-30:])
        score_val = current_score.get("final_score", 0)
        color = (0, 0, 220) if score_val > THRESH else (0, 200, 80)
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(annotated, f"Violence: {score_val:.2f}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if current_score.get("alert"):
            cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 4)
            cv2.putText(annotated, "⚠ ALERT", (w//2 - 60, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Encode and emit frame
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode("utf-8")
        socketio.emit("frame", {"data": b64})

    cap.release()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "running": is_running,
        "model_loaded": model_loaded,
        "alerts_count": len(alert_log),
        "current_score": current_score,
    })

@app.route("/api/alerts")
def get_alerts():
    return jsonify(alert_log[-50:])  # last 50 alerts

@app.route("/api/alerts/<int:alert_id>/confirm", methods=["POST"])
def confirm_alert(alert_id):
    for a in alert_log:
        if a["id"] == alert_id:
            a["confirmed"] = True
            a["confirmed_by"] = request.json.get("officer", "Officer")
            a["confirmed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            socketio.emit("alert_updated", a)
            return jsonify({"status": "confirmed", "alert": a})
    return jsonify({"error": "Alert not found"}), 404

@app.route("/api/alerts/<int:alert_id>/dismiss", methods=["POST"])
def dismiss_alert(alert_id):
    for a in alert_log:
        if a["id"] == alert_id:
            a["dismissed"] = True
            socketio.emit("alert_updated", a)
            return jsonify({"status": "dismissed"})
    return jsonify({"error": "Not found"}), 404

# ── SocketIO Events ───────────────────────────────────────────────────────────
@socketio.on("start_stream")
def start_stream(data):
    global is_running, camera_feed
    if is_running: return
    is_running = True
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    t = threading.Thread(target=process_stream, args=(source,), daemon=True)
    t.start()
    emit("stream_started", {"source": source})
    print(f"→ Stream started | source={source}")

@socketio.on("stop_stream")
def stop_stream():
    global is_running
    is_running = False
    emit("stream_stopped", {})
    print("→ Stream stopped")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  Violence Detection Alert Dashboard")
    print("  Open: http://localhost:5000")
    print("="*50)
    pose_model, scorer, device = load_models()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

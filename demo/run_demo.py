"""
Quick Demo — Run violence detection on webcam or video file.
No dashboard needed — just a live OpenCV window.

Usage:
    python demo/run_demo.py                     # webcam
    python demo/run_demo.py --source video.mp4  # video file
    python demo/run_demo.py --source 0 --model-size s
"""
import cv2, torch, numpy as np, argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

def run_demo(source=0, model_size="n", show_skeleton=True):
    from ultralytics import YOLO
    from lstm_classifier import BiLSTMClassifier, CFG, auto_device
    import torch.nn.functional as F

    device = auto_device()
    print(f"→ Device: {device}")

    # Load models
    pose  = YOLO(f"yolov8{model_size}-pose.pt")
    model = BiLSTMClassifier(CFG["feat_dim"], CFG["hidden_dim"],
                             CFG["n_layers"], CFG["num_classes"], CFG["dropout"])
    ckpt  = Path(__file__).resolve().parent.parent / "models" / "lstm_best.pth"
    demo_mode = False
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device); model.eval()
        print("✓ Loaded trained LSTM model")
    else:
        print("⚠  No trained model found — running in DEMO MODE (random scores)")
        demo_mode = True

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"✗ Cannot open: {source}"); return

    WINDOW = 30
    THRESH = 0.55
    buffer = np.zeros((WINDOW, CFG["feat_dim"]), dtype=np.float32)
    frame_i = 0
    current_score = 0.0
    fps_list = []

    print("→ Running demo. Press 'q' to quit, 's' to save screenshot.")
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str): cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        annotated = frame.copy()

        # Pose estimation
        results = pose(frame, verbose=False, conf=0.3)
        if show_skeleton:
            annotated = results[0].plot()

        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kp = r.keypoints.data[0].cpu().numpy()
                kp[:, 0] /= w; kp[:, 1] /= h
                buffer[frame_i % WINDOW] = kp.flatten()

        frame_i += 1

        # Score every window
        if frame_i % WINDOW == 0:
            if demo_mode:
                current_score = float(np.random.beta(1, 4))
            else:
                seq = torch.from_numpy(buffer.copy()).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(seq)
                    current_score = F.softmax(logits, dim=1)[0, 1].item()

        # Draw HUD
        is_alert = current_score > THRESH
        bar_w = int(current_score * (w - 40))
        bar_color = (0, 0, 220) if is_alert else (0, int(200 * (1 - current_score)), int(150 * current_score + 80))
        cv2.rectangle(annotated, (20, h-40), (w-20, h-20), (30, 30, 30), -1)
        cv2.rectangle(annotated, (20, h-40), (20 + bar_w, h-20), bar_color, -1)
        cv2.putText(annotated, f"Violence: {current_score:.3f}", (25, h-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if is_alert:
            cv2.rectangle(annotated, (2, 2), (w-2, h-2), (0, 0, 255), 4)
            cv2.putText(annotated, "ALERT — VIOLENCE DETECTED", (w//2 - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        fps_list.append(1 / (time.time() - t0 + 1e-6))
        avg_fps = np.mean(fps_list[-30:])
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(annotated, f"Device: {device}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow("Violence Detection Demo", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("s"):
            fn = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fn, annotated); print(f"✓ Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Done | Average FPS: {np.mean(fps_list):.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     default="0")
    ap.add_argument("--model-size", default="n", choices=["n","s","m"])
    ap.add_argument("--no-skeleton",action="store_true")
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    run_demo(src, args.model_size, not args.no_skeleton)

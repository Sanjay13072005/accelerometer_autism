import cv2
import csv
import joblib
import numpy as np
from collections import deque
from datetime import datetime

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\models\sleep_wake_modells.pkl"
ACCEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\data\raw\accelerometer\accel_night_01.csv"

# 🔴 PHONE IP WEBCAM
IP_USERNAME = "sanjay"
IP_PASSWORD = "Sanjay@1307"
IP_CAM_URL = f"http://{IP_USERNAME}:{IP_PASSWORD}@10.20.150.236:8080/video"
# =====================================================
# CONFIG (10 SECONDS)
# =====================================================
FPS_FALLBACK = 25
WINDOW_SECONDS = 10          # 🔥 CHANGED
ACCEL_WINDOW = 50
VIDEO_SLEEP_TH = 1.2
VIDEO_WAKE_TH = 2.5
FUSION_WINDOW = 30

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD ACCEL DATA
# =====================================================
def load_accel(csv_path):
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            data.append(np.sqrt(x*x + y*y + z*z))
    return data

# =====================================================
# VIDEO MOTION
# =====================================================
def video_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.mean(diff)

# =====================================================
# LIVE MONITOR
# =====================================================
def live_sleep_monitor():

    cap = cv2.VideoCapture(IP_CAM_URL)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot connect to IP Webcam")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK)
    window_frames = fps * WINDOW_SECONDS   # 🔥 10 seconds

    accel_data = load_accel(ACCEL_PATH)
    accel_idx = 0

    accel_buf = deque(maxlen=ACCEL_WINDOW)
    fusion_votes = deque(maxlen=FUSION_WINDOW)

    prev_gray = None
    frame_counter = 0

    print("\n🟢 LIVE SLEEP MONITOR STARTED")
    print("⏱ Decision every 10 seconds\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # -------- VIDEO --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        v_motion = 0
        if prev_gray is not None:
            v_motion = video_motion(prev_gray, gray)
        prev_gray = gray

        # -------- ACCEL --------
        a_val = accel_data[accel_idx % len(accel_data)]
        accel_idx += 1
        accel_buf.append(a_val)

        if len(accel_buf) < ACCEL_WINDOW:
            continue

        mean_m = np.mean(accel_buf)
        var_m = np.var(accel_buf)

        accel_pred = model.predict([[mean_m, var_m]])[0]
        accel_state = "SLEEP" if accel_pred == 0 else "WAKE"

        # -------- FUSION --------
        if accel_state == "SLEEP" and v_motion < VIDEO_SLEEP_TH:
            fused = "SLEEP"
        elif v_motion > VIDEO_WAKE_TH:
            fused = "WAKE"
        else:
            fused = accel_state

        fusion_votes.append(fused)

        # -------- 10-SECOND DECISION --------
        if frame_counter >= window_frames:
            final_state = (
                "SLEEP"
                if fusion_votes.count("SLEEP") > fusion_votes.count("WAKE")
                else "WAKE"
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 🛌 STATE → {final_state}")

            frame_counter = 0
            fusion_votes.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    live_sleep_monitor()

import cv2
import csv
import joblib
import numpy as np
from collections import deque

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\models\sleep_wake_modells.pkl"
VIDEO_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\4980331-uhd_4096_2160_25fps.mp4"
ACCEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\data\raw\accelerometer\accel_night_01.csv"

# =====================================================
# CONFIG (TUNED)
# =====================================================
ACCEL_WINDOW = 50                 # samples (~5 sec)
VIDEO_MOTION_SLEEP = 1.2          # safe sleep zone
VIDEO_MOTION_WAKE = 2.5           # strong wake motion
FUSION_VOTE_WINDOW = 20           # smoothing
STATE_CONFIRM_SECONDS = 4         # hysteresis

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD ACCELEROMETER
# =====================================================
def load_accel(csv_path):
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            data.append(np.sqrt(x*x + y*y + z*z))
    return data

# =====================================================
# VIDEO MOTION (ENERGY)
# =====================================================
def video_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.mean(diff)

# =====================================================
# MAIN FUSION
# =====================================================
def run_fusion(video_path, accel_csv):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    accel_data = load_accel(accel_csv)
    accel_idx = 0

    accel_buffer = deque(maxlen=ACCEL_WINDOW)
    fusion_votes = deque(maxlen=FUSION_VOTE_WINDOW)

    prev_gray = None
    total_frames = 0
    sleep_frames = 0

    stable_state = "WAKE"
    stable_counter = 0
    confirm_frames = fps * STATE_CONFIRM_SECONDS

    print("\n🟢 ENHANCED FUSION SLEEP DETECTION STARTED\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # -------- VIDEO --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        v_motion = 0
        if prev_gray is not None:
            v_motion = video_motion(prev_gray, gray)
        prev_gray = gray

        # -------- ACCEL --------
        a_val = accel_data[accel_idx] if accel_idx < len(accel_data) else 0
        accel_idx += 1
        accel_buffer.append(a_val)

        if len(accel_buffer) < ACCEL_WINDOW:
            continue

        mean_motion = np.mean(accel_buffer)
        var_motion  = np.var(accel_buffer)

        ml_pred = model.predict([[mean_motion, var_motion]])[0]
        accel_state = "SLEEP" if ml_pred == 0 else "WAKE"

        # -------- FUSION LOGIC (WEIGHTED) --------
        if accel_state == "SLEEP" and v_motion < VIDEO_MOTION_SLEEP:
            fused = "SLEEP"
        elif v_motion > VIDEO_MOTION_WAKE:
            fused = "WAKE"
        else:
            fused = accel_state  # accel dominates

        fusion_votes.append(fused)

        vote_state = (
            "SLEEP" if fusion_votes.count("SLEEP") > fusion_votes.count("WAKE")
            else "WAKE"
        )

        # -------- HYSTERESIS --------
        if vote_state == stable_state:
            stable_counter += 1
        else:
            stable_counter = 1
            stable_state = vote_state

        final_state = stable_state if stable_counter >= confirm_frames else stable_state

        if final_state == "SLEEP":
            sleep_frames += 1

        print(
            f"Accel:{accel_state:<5} | "
            f"Vid:{v_motion:5.2f} | "
            f"Vote:{vote_state:<5} | "
            f"FINAL → {final_state}"
        )

    cap.release()

    total_sleep_sec = sleep_frames / fps
    total_video_sec = total_frames / fps

    return {
        "total_sleep_seconds": round(total_sleep_sec, 2),
        "total_video_seconds": round(total_video_sec, 2),
        "final_state": "SLEEP" if total_sleep_sec > 0.6 * total_video_sec else "WAKE"
    }

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    result = run_fusion(VIDEO_PATH, ACCEL_PATH)

    print("\n📊 FINAL SLEEP RESULT")
    print("----------------------")
    print(f"Total Sleep (sec): {result['total_sleep_seconds']}")
    print(f"Total Video (sec): {result['total_video_seconds']}")
    print(f"\n✅ FINAL STATE → {result['final_state']}")

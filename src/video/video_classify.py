import cv2
import csv
import joblib
import numpy as np
from collections import deque

# =====================================================
# PATHS here is the main code to analysis to detect the sleep or wake
# =====================================================
MODEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\models\sleep_wake_modells.pkl"
VIDEO_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\Standing Still (105).mp4"
ACCEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\data\raw\accelerometer\accel_night_01.csv"

# =====================================================
# CONFIG (STABLE + TUNED)
# =====================================================
ACCEL_WINDOW = 50            # ~5 sec accel window
VIDEO_SLEEP_TH = 1.2
VIDEO_WAKE_TH  = 2.5
FUSION_WINDOW  = 30          # smoothing
CONFIRM_SECONDS = 5          # state confirmation time
SLEEP_RATIO_TH = 0.6         # final decision

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
        for r in reader:
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            data.append(np.sqrt(x*x + y*y + z*z))
    return data

# =====================================================
# VIDEO MOTION ENERGY
# =====================================================
def video_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.mean(diff)

# =====================================================
# MAIN ANALYSIS
# =====================================================
def analyze_sleep(video_path, accel_path):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    accel_data = load_accel(accel_path)
    accel_idx = 0

    accel_buf = deque(maxlen=ACCEL_WINDOW)
    fusion_votes = deque(maxlen=FUSION_WINDOW)

    prev_gray = None
    total_frames = 0
    sleep_frames = 0

    stable_state = "WAKE"
    last_confirmed_state = "WAKE"
    stable_counter = 0
    confirm_frames = fps * CONFIRM_SECONDS

    sleep_events = 0
    wake_events = 0

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
        accel_buf.append(a_val)

        if len(accel_buf) < ACCEL_WINDOW:
            continue

        mean_m = np.mean(accel_buf)
        var_m  = np.var(accel_buf)

        ml_pred = model.predict([[mean_m, var_m]])[0]
        accel_state = "SLEEP" if ml_pred == 0 else "WAKE"

        # -------- FUSION --------
        if accel_state == "SLEEP" and v_motion < VIDEO_SLEEP_TH:
            fused = "SLEEP"
        elif v_motion > VIDEO_WAKE_TH:
            fused = "WAKE"
        else:
            fused = accel_state

        fusion_votes.append(fused)

        vote = (
            "SLEEP"
            if fusion_votes.count("SLEEP") > fusion_votes.count("WAKE")
            else "WAKE"
        )

        # -------- STABILITY --------
        if vote == stable_state:
            stable_counter += 1
        else:
            stable_state = vote
            stable_counter = 1

        if stable_counter >= confirm_frames:
            if stable_state != last_confirmed_state:
                if stable_state == "SLEEP":
                    sleep_events += 1
                else:
                    wake_events += 1
                last_confirmed_state = stable_state

        if last_confirmed_state == "SLEEP":
            sleep_frames += 1

    cap.release()

    sleep_ratio = sleep_frames / max(total_frames, 1)
    final_state = "SLEEP" if sleep_ratio >= SLEEP_RATIO_TH else "WAKE"

    return {
        "final_state": final_state,
        "sleep_ratio": round(sleep_ratio, 2),
        "video_seconds": round(total_frames / fps, 2),
        "sleep_events": sleep_events,
        "wake_events": wake_events,
    }

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    result = analyze_sleep(VIDEO_PATH, ACCEL_PATH)

    print("\n📊 FINAL SLEEP ANALYSIS")
    print("----------------------")
    print(f"Video Duration (sec): {result['video_seconds']}")
    print(f"Sleep Ratio          : {result['sleep_ratio']}")
    print(f"Sleep Events         : {result['sleep_events']}")
    print(f"Wake Events          : {result['wake_events']}")
    print(f"\n✅ FINAL RESULT → {result['final_state']}")

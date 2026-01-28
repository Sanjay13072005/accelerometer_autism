import cv2
import csv
import joblib
import numpy as np
import requests
import sounddevice as sd
from collections import deque
from datetime import datetime, timedelta

# ================= TELEGRAM =================

BOT_TOKEN = "7919287381:AAEeBCXf5tM7SVZ-HhAZxKZ1ZFkz74uqWZM"
CHAT_ID   = "5288367155"
def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=3
        )
    except:
        pass

# ================= IP WEBCAM =================
IP_USERNAME = "sanjay"
IP_PASSWORD = "Sanjay@1307"
IP_CAM_URL = f"http://{IP_USERNAME}:{IP_PASSWORD}@10.20.150.236:8080/video"

# ================= PATHS =================
MODEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\models\sleep_wake_modells.pkl"
ACCEL_PATH = r"C:\Users\HAI\Music\acceleromete_main_project\data\raw\accelerometer\accel_night_01.csv"

# ================= CONFIG =================
FPS_FALLBACK = 25
FRAME_SKIP = 4
WINDOW_SECONDS = 10

ACCEL_WINDOW = 50

VIDEO_WAKE_TH = 3.0          # VERY HIGH ONLY
AUDIO_WAKE_TH = 0.08         # EXTREME SOUND ONLY
NO_PERSON_SECONDS = 20

SHOW_PREVIEW = True

# ================= LOAD MODEL =================
model = joblib.load(MODEL_PATH)

# ================= LOAD ACCEL =================
def load_accel(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            out.append(np.sqrt(x*x + y*y + z*z))
    return out

# ================= AUDIO =================
def audio_energy(duration=0.3, fs=16000):
    a = sd.rec(int(duration*fs), samplerate=fs, channels=1, blocking=True)
    return float(np.sqrt(np.mean(a**2)))

# ================= FACE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def person_present(gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

# ================= MOTION =================
def motion(prev, curr):
    return np.mean(cv2.absdiff(prev, curr))

# ================= MAIN =================
def live_sleep_monitor():

    cap = cv2.VideoCapture(IP_CAM_URL)
    if not cap.isOpened():
        raise RuntimeError("IP Webcam not reachable")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK)

    accel = load_accel(ACCEL_PATH)
    accel_i = 0
    accel_buf = deque(maxlen=ACCEL_WINDOW)

    prev_gray = None
    frame_id = 0
    last_state = None
    last_person_seen = datetime.now()

    print("🟢 RELIABLE SLEEP MONITOR STARTED")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            if SHOW_PREVIEW:
                cv2.imshow("Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # -------- PERSON --------
        if person_present(gray):
            last_person_seen = datetime.now()

        # -------- VIDEO MOTION --------
        v_motion = 0 if prev_gray is None else motion(prev_gray, gray)
        prev_gray = gray

        # -------- ACCEL (PRIMARY) --------
        a = accel[accel_i % len(accel)]
        accel_i += 1
        accel_buf.append(a)

        if len(accel_buf) < ACCEL_WINDOW:
            continue

        mean_m = np.mean(accel_buf)
        var_m = np.var(accel_buf)
        ml_state = "SLEEP" if model.predict([[mean_m, var_m]])[0] == 0 else "WAKE"

        final_state = ml_state

        # -------- OVERRIDES (STRICT) --------
        if ml_state == "SLEEP":
            if v_motion > VIDEO_WAKE_TH:
                final_state = "WAKE"

            elif audio_energy() > AUDIO_WAKE_TH:
                final_state = "WAKE"

            elif (datetime.now() - last_person_seen).seconds > NO_PERSON_SECONDS:
                final_state = "WAKE"

        # -------- OUTPUT --------
        if SHOW_PREVIEW:
            cv2.putText(
                frame, final_state, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,0) if final_state=="SLEEP" else (0,0,255), 2
            )
            cv2.imshow("Monitor", frame)

        if final_state != last_state:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] → {final_state}")

            if final_state == "WAKE":
                send_telegram(f"🚨 CHILD WAKE ALERT\n🕒 {ts}")

            last_state = final_state

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    live_sleep_monitor()

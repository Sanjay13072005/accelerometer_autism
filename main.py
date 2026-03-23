import time

from realtime.accelerometer_stream import AccelerometerStream
from realtime.camera_stream import CameraStream
from realtime.eye_detector import EyeDetector
from realtime.audio_stream import AudioStream
from realtime.feature_extractor import extract_motion_features
from realtime.sleep_classifier import SleepClassifier
from realtime.decision_engine import DecisionEngine

# ===============================
# CONFIG
# ===============================
USE_LAPTOP_CAMERA = True   # ← CHANGE THIS LATER
PHONE_IP = "10.20.150.236"

ACCEL_CSV = "data/live/Accelerometer.csv"

# ===============================
# STREAM SETUP
# ===============================
accel = AccelerometerStream(ACCEL_CSV)

if USE_LAPTOP_CAMERA:
    camera = CameraStream(0)   # Laptop webcam
    audio = None               # No phone audio yet
else:
    camera = CameraStream(f"http://{PHONE_IP}:8080/video")
    audio = AudioStream(f"http://{PHONE_IP}:8080/sensors.json")

eyes = EyeDetector()
clf = SleepClassifier("models/sleep_awake_model.pkl")
decision = DecisionEngine(move_thresh=20, sound_thresh=0.08)

print("🟢 Autism Night Monitor STARTED")
print("🎥 Camera:", "Laptop Webcam" if USE_LAPTOP_CAMERA else "IP Webcam")

# ===============================
# MAIN LOOP
# ===============================
while True:
    accel.update()
    window = accel.get_window()

    frame = camera.read()
    if frame is None:
        print("⚠️ Camera frame not available")
        time.sleep(0.2)
        continue

    eyes_open = eyes.eyes_open(frame)

    # Sound
    sound = audio.get_rms() if audio else 0.0

    if window:
        feats = extract_motion_features(window)
        state = clf.predict(feats)

        distressed = decision.is_distressed(
            state,
            feats["energy"],
            sound,
            eyes_open
        )

        print(
            "STATE:",
            "SLEEP" if state == 1 else "AWAKE",
            "| Energy:", round(feats["energy"], 2),
            "| Sound:", round(sound, 3),
            "| Eyes open:", eyes_open,
            "| ALERT:", distressed
        )
# time delay in seconds
    time.sleep(0.5)

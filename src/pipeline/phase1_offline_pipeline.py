import os
import yaml
import pandas as pd
import os
import sys

# ===== PROJECT ROOT (MUST BE FIRST) =====
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)


from src.data_ingestion.accelerometer_loader import load_accelerometer_csv
from src.preprocessing.windowing import window_accelerometer
from src.feature_engineering.motion_features import extract_motion_features
from src.decision_engine.rule_based_sleep_detector import RuleBasedSleepDetector

RAW_DIR = "data/raw/accelerometer"
INTERIM_DIR = "data/interim/windowed"
FEATURE_DIR = "data/processed/features"
LOG_DIR = "logs"

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

with open(os.path.join(PROJECT_ROOT, "data/metadata/recording_info.yaml")) as f:
    meta = yaml.safe_load(f)

if meta is None:
    raise ValueError("recording_info.yaml is empty or invalid")


with open("src/config/thresholds.yaml") as f:
    th = yaml.safe_load(f)

detector = RuleBasedSleepDetector(
    th["sleep_motion_threshold"],
    th["wake_motion_threshold"],
    th["min_sleep_windows"]
)

final_log = []

for file in os.listdir(RAW_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(RAW_DIR, file)
    print(f"Processing {file}")

    df = load_accelerometer_csv(path)

    windows = window_accelerometer(
        df,
        meta["sampling_rate_hz"],
        meta["window_size_sec"]
    )

    features = extract_motion_features(windows)

    features_path = os.path.join(
        FEATURE_DIR,
        file.replace(".csv", "_features.csv")
    )
    features.to_csv(features_path, index=False)

    for _, row in features.iterrows():
        state = detector.update(row["mean_motion"])
        final_log.append({
            "timestamp": row["timestamp"],
            "state": state
        })

log_df = pd.DataFrame(final_log)
log_df.to_csv(f"{LOG_DIR}/sleep_wake_timeline.csv", index=False)

print("✅ Phase-1 completed successfully")

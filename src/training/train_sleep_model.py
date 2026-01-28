import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# PROJECT ROOT
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

FEATURE_DIR = os.path.join(PROJECT_ROOT, "data/processed/features")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs/sleep_wake_timeline.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD FEATURE DATA
# =====================================================
feature_frames = []

for file in os.listdir(FEATURE_DIR):
    if file.endswith("_features.csv"):
        df = pd.read_csv(os.path.join(FEATURE_DIR, file))
        df["timestamp"] = df["timestamp"].astype(float)
        feature_frames.append(df)

if not feature_frames:
    raise RuntimeError("❌ No feature files found")

X_df = pd.concat(feature_frames, ignore_index=True)

# =====================================================
# LOAD LABEL DATA (FROM PHASE-1)
# =====================================================
labels_df = pd.read_csv(LOG_FILE)
labels_df["timestamp"] = labels_df["timestamp"].astype(float)

# =====================================================
# ALIGN FEATURES + LABELS (TIME-SAFE)
# =====================================================
data = pd.merge_asof(
    X_df.sort_values("timestamp"),
    labels_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest"
)

data["label"] = data["state"].map({"SLEEP": 0, "WAKE": 1})

# Drop any unmapped rows
data = data.dropna(subset=["label"])

X = data[["mean_motion", "var_motion"]]
y = data["label"].astype(int)

print("\n📊 Class distribution:")
print(y.value_counts())

# =====================================================
# TRAIN / TEST SPLIT (SAFE)
# =====================================================
unique_classes = y.nunique()

if unique_classes == 1:
    print("\n⚠️ Only ONE class present. Training without test split.")
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# =====================================================
# MODEL (OPTIMIZED FOR SENSOR DATA)
# =====================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION (CRASH-SAFE)
# =====================================================
y_pred = model.predict(X_test)

if unique_classes == 1:
    print("\n⚠️ Skipping classification report (single-class data)")
else:
    print("\n📈 Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["SLEEP", "WAKE"]
        )
    )

# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump(model, os.path.join(MODEL_DIR, "sleep_wake_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(
        {
            "features": ["mean_motion", "var_motion"],
            "model": "RandomForestClassifier",
            "phase": 2,
            "single_class_training": bool(unique_classes == 1)
        },
        f,
        indent=4
    )

print("\n✅ Phase-2 ML model trained and saved successfully")

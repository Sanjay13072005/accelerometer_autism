import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# PROJECT ROOT & PATHS
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR  = os.path.join(PROJECT_ROOT, "data", "raw")
SLEEP_DIR = os.path.join(DATA_DIR, "sleep")
WAKE_DIR  = os.path.join(DATA_DIR, "vigil")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Training script started")
print("📂 SLEEP_DIR =", SLEEP_DIR)
print("📂 WAKE_DIR  =", WAKE_DIR)

# =====================================================
# FEATURE EXTRACTION (ROBUST)
# =====================================================
def extract_features(csv_path):
    df = pd.read_csv(csv_path)

    # ---- detect headerless CSV ----
    def looks_like_data(col):
        col = str(col)
        return (
            col.replace(".", "", 1).isdigit() or
            ":" in col
        )

    if all(looks_like_data(c) for c in df.columns):
        df = pd.read_csv(csv_path, header=None)

        if df.shape[1] < 4:
            raise ValueError("Not enough columns")

        df.columns = ["timestamp", "x", "y", "z"] + \
                     [f"extra_{i}" for i in range(df.shape[1] - 4)]

    df.columns = [c.lower() for c in df.columns]

    # ---- raw accelerometer ----
    if {"x", "y", "z"}.issubset(df.columns):
        motion = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        return motion.mean(), motion.var()

    # ---- precomputed features ----
    if {"mean_motion", "var_motion"}.issubset(df.columns):
        return df["mean_motion"].mean(), df["var_motion"].mean()

    for col in ["motion", "magnitude", "energy"]:
        if col in df.columns:
            return df[col].mean(), df[col].var()

    raise ValueError(f"Unsupported schema {list(df.columns)}")

# =====================================================
# LOAD DATA
# =====================================================
X, y = [], []

def load_folder(folder, label):
    print(f"\n🔍 Scanning {folder}")
    total, loaded = 0, 0

    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                total += 1
                path = os.path.join(root, f)
                try:
                    mean_m, var_m = extract_features(path)
                    X.append([mean_m, var_m])
                    y.append(label)
                    loaded += 1
                except Exception as e:
                    print(f"⚠️ Skipped {f} → {e}")

    print(f"✅ Loaded {loaded}/{total} files")

print("\n📥 Loading SLEEP data")
load_folder(SLEEP_DIR, label=0)

print("\n📥 Loading WAKE data")
load_folder(WAKE_DIR, label=1)

X = np.array(X)
y = np.array(y)

# =====================================================
# DATASET SUMMARY
# =====================================================
print("\n📊 Dataset Summary")
print("-------------------")
print("Total samples :", len(y))
print("SLEEP samples :", (y == 0).sum())
print("WAKE samples  :", (y == 1).sum())

if len(np.unique(y)) < 2:
    raise RuntimeError("❌ Need both SLEEP and WAKE data")

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

print("\n📈 Classification Report")
print("------------------------")
print(classification_report(
    y_test,
    y_pred,
    target_names=["SLEEP", "WAKE"]
))

# =====================================================
# SAVE MODEL
# =====================================================
model_path = os.path.join(MODEL_DIR, "sleep_wake_modells.pkl")
joblib.dump(model, model_path)

print(f"\n✅ Model saved successfully:\n{model_path}")

import os
import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/sleep_wake_model.pkl")
FEATURE_PATH = os.path.join(
    PROJECT_ROOT, "data/processed/features/accel_night_01_features.csv"
)

model = joblib.load(MODEL_PATH)

df = pd.read_csv(FEATURE_PATH)

X = df[["mean_motion", "var_motion"]]
df["prediction"] = model.predict(X)
df["prediction"] = df["prediction"].map({0: "SLEEP", 1: "WAKE"})

output_path = os.path.join(PROJECT_ROOT, "logs/ml_sleep_predictions.csv")
df[["timestamp", "prediction"]].to_csv(output_path, index=False)

print("✅ ML predictions generated")

import numpy as np
import pandas as pd

def extract_motion_features(windows):
    rows = []

    for w in windows:
        mag = np.sqrt(w["x"]**2 + w["y"]**2 + w["z"]**2)
        rows.append({
            "timestamp": w["timestamp"].iloc[-1],
            "mean_motion": mag.mean(),
            "var_motion": mag.var()
        })

    return pd.DataFrame(rows)

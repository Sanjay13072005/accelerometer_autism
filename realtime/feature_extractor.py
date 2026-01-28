import numpy as np

def extract_motion_features(window):
    arr = np.array(window, dtype=float)
    mag = np.linalg.norm(arr, axis=1)

    return {
        "mean": mag.mean(),
        "std": mag.std(),
        "energy": (mag ** 2).sum(),
        "max": mag.max()
    }

import numpy as np

class LiveFeatureExtractor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []

    def update(self, x, y, z):
        self.buffer.append((x, y, z))

        if len(self.buffer) < self.window_size:
            return None

        window = self.buffer[-self.window_size:]
        mags = [np.sqrt(a*a + b*b + c*c) for a,b,c in window]

        return {
            "mean_motion": np.mean(mags),
            "var_motion": np.var(mags)
        }

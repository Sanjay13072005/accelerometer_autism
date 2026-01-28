import numpy as np
import pandas as pd

def window_accelerometer(df, sampling_rate, window_size_sec):
    window_size = sampling_rate * window_size_sec
    windows = []

    for i in range(0, len(df), window_size):
        chunk = df.iloc[i:i + window_size]
        if len(chunk) == window_size:
            windows.append(chunk)

    return windows

import pandas as pd
from collections import deque
import os

class AccelerometerStream:
    def __init__(self, csv_path, window_size=40):
        self.csv_path = csv_path
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.last_row = 0

    def update(self):
        if not os.path.exists(self.csv_path):
            return False

        df = pd.read_csv(self.csv_path)

        if len(df) <= self.last_row:
            return False

        new_rows = df.iloc[self.last_row:]
        self.last_row = len(df)

        for _, row in new_rows.iterrows():
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
                self.buffer.append([x, y, z])
            except:
                continue

        return True

    def get_window(self):
        if len(self.buffer) == self.window_size:
            return list(self.buffer)
        return None

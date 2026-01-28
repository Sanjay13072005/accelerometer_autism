import pandas as pd
import time

class AccelerometerStream:
    def __init__(self, csv_path, sampling_rate=10):
        self.df = pd.read_csv(csv_path)

        # Normalize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]

        # ---- TIME COLUMN ----
        for col in ["timestamp", "time", "epoch_time", "seconds_elapsed"]:
            if col in self.df.columns:
                self.df.rename(columns={col: "timestamp"}, inplace=True)
                break
        else:
            raise ValueError(
                f"No time column found. Columns: {self.df.columns.tolist()}"
            )

        # ---- AXES ----
        axis_map = {
            "accel_x": "x", "accelerometer_x": "x", "ax": "x", "x": "x",
            "accel_y": "y", "accelerometer_y": "y", "ay": "y", "y": "y",
            "accel_z": "z", "accelerometer_z": "z", "az": "z", "z": "z",
        }

        rename_map = {}
        for col in self.df.columns:
            if col in axis_map:
                rename_map[col] = axis_map[col]

        self.df.rename(columns=rename_map, inplace=True)

        required = {"timestamp", "x", "y", "z"}
        if not required.issubset(self.df.columns):
            raise ValueError(
                f"CSV missing required columns {required}. "
                f"Found {self.df.columns.tolist()}"
            )

        self.df["timestamp"] = self.df["timestamp"].astype(float)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        self.sampling_rate = sampling_rate
        self.index = 0

    def read(self):
        if self.index >= len(self.df):
            return None

        row = self.df.iloc[self.index]
        self.index += 1
        time.sleep(1 / self.sampling_rate)

        return row["x"], row["y"], row["z"]

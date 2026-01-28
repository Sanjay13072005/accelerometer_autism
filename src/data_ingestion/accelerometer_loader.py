import pandas as pd

def load_accelerometer_csv(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # -------- TIME COLUMN --------
    time_candidates = [
        "timestamp",
        "time",
        "epoch_time",
        "seconds_elapsed"
    ]

    for col in time_candidates:
        if col in df.columns:
            df.rename(columns={col: "timestamp"}, inplace=True)
            break
    else:
        raise ValueError(
            f"No time column found in {path}. "
            f"Columns found: {df.columns.tolist()}"
        )

    # -------- ACCELEROMETER AXES --------
    axis_map = {
        "accel_x": "x",
        "accelerometer_x": "x",
        "ax": "x",
        "x": "x",

        "accel_y": "y",
        "accelerometer_y": "y",
        "ay": "y",
        "y": "y",

        "accel_z": "z",
        "accelerometer_z": "z",
        "az": "z",
        "z": "z",
    }

    rename_map = {}
    for col in df.columns:
        if col in axis_map:
            rename_map[col] = axis_map[col]

    df.rename(columns=rename_map, inplace=True)

    # -------- FINAL CHECK --------
    required = {"timestamp", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns {required}. "
            f"Found: {df.columns.tolist()}"
        )

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

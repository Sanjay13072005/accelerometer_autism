# 🌙 Accelerometer Autism - Sleep Detection System

Real-time sleep detection using accelerometer, camera, and audio streams.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

---

## 📋 What It Does

- ✅ **Detects Sleep/Awake** - Uses motion sensors + ML
- ✅ **Eye Detection** - Monitors eye closure  
- ✅ **Real-time Alerts** - Sends Telegram notifications
- ✅ **Multi-sensor** - Phone Accelerometer + Camera + Audio

---

## ⚙️ Configuration

Edit `main.py`:
```python
USE_LAPTOP_CAMERA = True              # Use laptop webcam
PHONE_IP = "10.20.150.236"           # Or use IP camera
ACCEL_CSV = "data/live/Accelerometer.csv"
```

Edit `src/config/thresholds.yaml`:
```yaml
sleep_motion_threshold: 0.02
wake_motion_threshold: 0.08
min_sleep_windows: 6
```

---

## 📁 Project Structure

```
├── main.py                    # Start here - main entry point
├── requirements.txt           # Dependencies
├── realtime/                  # Real-time components
│   ├── accelerometer_stream.py
│   ├── camera_stream.py
│   ├── eye_detector.py
│   ├── sleep_classifier.py
│   └── decision_engine.py
└── src/
    ├── config/thresholds.yaml
    ├── training/              # Train models
    ├── video/                 # Analyze video
    └── pipeline/              # Batch processing
```

---

## 📚 Other Commands

**Train a new model:**
```bash
python src/training/train_sleep_model_from_folders.py
```

**Analyze video:**
```bash
python src/video/video_classify.py
```

**Process batch data:**
```bash
python src/pipeline/phase1_offline_pipeline.py
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not working | Check camera permissions & IP address |
| Missing data | Ensure CSV file exists at `ACCEL_CSV` path |
| Import errors | Run `pip install -r requirements.txt --upgrade` |

---

## 📦 Dependencies

- `opencv-python` - Camera processing
- `mediapipe` - Eye detection
- `scikit-learn` - Sleep classification
- `sounddevice` - Audio capture
- `python-telegram-bot` - Notifications
- `librosa` - Audio analysis
- `pyyaml` - Config files

---

## 📊 Data Format

**Accelerometer CSV:**
```
timestamp, x, y, z
1000, 0.1, 0.05, 9.8
1001, 0.12, 0.06, 9.75
```

---

## 💡 Key Components

| Component | Purpose |
|-----------|---------|
| `accelerometer_stream.py` | Read sensor data |
| `eye_detector.py` | Detect eye closure |
| `feature_extractor.py` | Calculate motion features |
| `sleep_classifier.py` | ML-based classification |
| `decision_engine.py` | Make decisions + alerts |

---

**Version**: January 2026

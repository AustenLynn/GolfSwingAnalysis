# Golf Swing Analyzer

Project workspace for capturing, processing, and classifying golf swing IMU data with an ESP32 + Python.

## Hardware

- ESP32-C3 with onboard IMU (accelerometer + gyroscope)
- Sends data over BLE (UART service) at ~70 Hz
- Columns captured: `t_ms, ax, ay, az, gx, gy, gz, yaw, pitch, roll`

## Project Structure

```
firmware/               ESP32 Arduino firmware
scripts/
  capture_imu.py        Serial capture (USB)
  capture_imu_ble.py    BLE capture — saves to data/raw/ (good) or data/raw/Suboptimal_Swings/ (bad)
  analyze_swing.py      Swing segmentation + tempo scoring (standalone)
  extract_features.py   Feature extraction pipeline → data/processed/swing_features.csv
  explore_features.py   Exploratory analysis — box plots, correlation heatmap, Cohen's d
  train_classifier.py   Train good/bad classifier with LOOCV → models/swing_classifier.pkl
  classify_swing.py     Classify swings in a new capture file
data/raw/               Good swing captures (label = 1)
data/raw/Suboptimal_Swings/  Bad swing captures (label = 0)
data/processed/         Feature CSV + analysis plots
models/                 Saved classifier
notebooks/              Experiment notebooks
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Capture data

```bash
# Good swing — saves to data/raw/
python scripts/capture_imu.py

# Bad (suboptimal) swing — saves to data/raw/Suboptimal_Swings/
python scripts/capture_imu_ble.py
```

### 2. Run the ML pipeline

```bash
# Extract features from all labeled captures
python scripts/extract_features.py

# Explore feature distributions and separability
python scripts/explore_features.py

# Train classifier (LOOCV evaluation + save model)
python scripts/train_classifier.py
```

### 3. Classify a new swing

```bash
python scripts/classify_swing.py data/raw/my_new_capture.csv
```

Output:

```
File : my_new_capture.csv
Swings detected: 2

Swing    Verdict    Confidence     Top features
------------------------------------------------------------------------
  #1      GOOD       97.0%          acc_mean=0.992 ...
  #2      BAD        61.3%          acc_mean=1.610 ...
```

## How It Works

### Swing segmentation

`analyze_swing.py` computes the angular velocity magnitude (`omega_mag`) and detects swings as contiguous regions above a 60 deg/s threshold. Each swing is split into:

- **Backswing** — start → omega peak in first half
- **Downswing** — omega peak → impact (second omega peak)
- **Follow-through** — impact → end of motion

### Features extracted per swing (19 total)

| Group | Features |
|---|---|
| Tempo | `tempo_ratio` (backswing/downswing time), `T_backswing_s`, `T_downswing_s`, `T_total_s` |
| Angular velocity | `omega_peak`, `omega_mean`, `omega_std`, `omega_skew`, `omega_at_top`, `omega_rise_rate`, `transition_dip_ratio` |
| Acceleration | `acc_peak`, `acc_mean`, `acc_std`, `acc_jerk_max` |
| Euler angles | `yaw_range`, `pitch_range`, `roll_range`, `yaw_at_top` |

### Top discriminating features (Cohen's d, current dataset)

| Feature | Cohen's d | Interpretation |
|---|---|---|
| `acc_mean` | 1.89 *** | Bad swings have much higher average acceleration |
| `omega_mean` | 1.74 *** | Bad swings have ~2x higher mean angular velocity |
| `yaw_at_top` | 1.67 *** | Different yaw position at top of backswing (shoulder turn) |
| `acc_std`, `acc_peak`, `omega_at_top` | 1.2–1.3 ** | Bad swings are faster and more erratic at every phase |

### Classifier

Three models are compared using **Leave-One-Out CV** (the only statistically valid approach at this sample size):

| Model | LOOCV Accuracy | Bad swing recall |
|---|---|---|
| Rule-based thresholds | 61.9% | 80% |
| SVM (RBF kernel) | 71.4% | 0% |
| **Random Forest** | **76.2%** | **40%** |

Random Forest is saved as the default model. Features used: `acc_mean`, `omega_mean`, `yaw_at_top`, `acc_std`, `omega_at_top`, `T_backswing_s`.

## Current Dataset

| Class | Captures | Swings extracted |
|---|---|---|
| Good (label=1) | 8 files | 16 swings |
| Bad (label=0) | 5 files | 5 swings |
| **Total** | **13 files** | **21 swings** |

## Next Steps

### High priority — more data

The biggest bottleneck is the **5 bad swing samples**. Bad swing recall is 40% (misses 3 of 5). Capturing 10–15 more suboptimal swings would meaningfully improve the model and make LOOCV estimates more reliable.

Aim for a balanced dataset: **~30 good + ~30 bad swings** before investing in model tuning.

### Expand bad swing categories

Currently all bad swings are grouped together. Splitting them into specific fault types would make the feedback more actionable:

- Over-the-top (steep downswing path)
- Early extension (loss of posture)
- Casting (releasing the wrists too early)
- Over-swing (excessive backswing length)

Each fault has a distinct IMU signature and could be its own class.

### Improve swing segmentation

The current threshold-based segmenter (`START_THRESHOLD = 60 deg/s`) can merge or split swings when the sensor is moved between shots. Options:

- Tune thresholds per session using a calibration swing
- Use a peak-finding approach (`scipy.signal.find_peaks`) instead of a hysteresis threshold

### Feature engineering

- **DTW (Dynamic Time Warping) distance** to a reference good swing template — captures shape, not just statistics
- **Frequency domain features** (FFT of omega_mag) — characterizes smoothness and timing rhythm
- **Per-phase features** (backswing / downswing / follow-through separately) instead of whole-swing aggregates

### Model improvements

Once the dataset is larger (50+ swings):
- Try **cross-validation with stratified k-fold** instead of LOOCV
- Tune SVM `C` and `gamma` with a grid search — the RBF kernel collapsed to majority class at current scale
- Consider **LightGBM** or **XGBoost** which handle small imbalanced datasets well

### Real-time feedback

Extend `classify_swing.py` to run during a BLE capture session and give immediate feedback (good/bad + which feature was out of range) after each swing.

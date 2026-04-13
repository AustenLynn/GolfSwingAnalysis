# Golf Swing Analyzer

A full-stack personal golf swing analysis system. An ESP32-C3 IMU sensor worn during a swing streams data via BLE to an Android app, which sends it to a Python ML backend for classification. The verdict (GOOD/BAD), confidence score, and feature breakdown are displayed on the phone in real time.

---

## System Architecture

```
┌─────────────────────┐      BLE (NUS profile)      ┌──────────────────────────┐
│  ESP32-C3 + IMU     │ ─────────────────────────── ▶ │  Android App             │
│  (on the club/body) │  ~70 Hz CSV over 20-byte     │  (flutter_blue_plus)      │
│                     │  BLE notification chunks      │                          │
│  Sensors:           │                               │  - Live omega_mag chart  │
│  ADXL345 accel      │                               │  - Swing detection       │
│  ITG3200 gyro       │                               │  - POST /analyze →       │
│  HMC5883L compass   │                               │    show verdict          │
│  Madgwick filter    │                               │  - History & detail view │
└─────────────────────┘                               └────────────┬─────────────┘
                                                                   │
                                                      HTTP (Tailscale or LAN WiFi)
                                                                   │
                                                                   ▼
                                                      ┌────────────────────────────┐
                                                      │  FastAPI Backend (Docker)  │
                                                      │  PC on ethernet            │
                                                      │                            │
                                                      │  POST /analyze             │
                                                      │  → segmentation            │
                                                      │  → 19 feature extraction   │
                                                      │  → Random Forest classify  │
                                                      │                            │
                                                      │  GET  /swings              │
                                                      │  POST /swings/{id}/label   │
                                                      │  POST /model/retrain       │
                                                      │  GET  /health              │
                                                      │                            │
                                                      │  SQLite: swing_history.db  │
                                                      │  Model:  swing_classifier.pkl │
                                                      └────────────────────────────┘
```

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | ESP32-C3 |
| Accelerometer | ADXL345 — ±2g, 100 Hz |
| Gyroscope | ITG3200 — ±2000 deg/s, 100 Hz |
| Magnetometer | HMC5883L |
| Fusion filter | Madgwick AHRS |
| Transport | BLE NUS profile (Nordic UART Service) |
| BLE device name | `ESP32_IMU_GOLF` |
| Service UUID | `6e400001-b5a3-f393-e0a9-e50e24dcca9e` |
| TX notify UUID | `6e400003-b5a3-f393-e0a9-e50e24dcca9e` |

Data columns streamed: `t_ms, ax, ay, az, gx, gy, gz, yaw, pitch, roll`

> **BLE note:** The ESP32 accepts only one BLE connection at a time. You cannot have the Python capture scripts and the Android app connected simultaneously — disconnect one before connecting the other.

---

## Project Structure

```
GolfSwingAnalysis/
├── Dockerfile                      Docker image for the FastAPI backend
├── docker-compose.yml              Starts backend with persistent volumes
├── requirements.txt                ML pipeline dependencies
├── requirements_api.txt            FastAPI backend dependencies
│
├── firmware/
│   └── esp32_imu_capture/
│       └── esp32_imu_capture.ino  Arduino firmware for ESP32-C3
│
├── scripts/                        ML pipeline (also imported by the API)
│   ├── capture_imu.py              USB/serial capture
│   ├── capture_imu_ble.py          BLE capture → data/raw/ or Suboptimal_Swings/
│   ├── analyze_swing.py            Swing segmentation + phase detection
│   ├── extract_features.py         19-feature extraction per swing
│   ├── explore_features.py         EDA — box plots, correlation, Cohen's d
│   ├── train_classifier.py         Train + LOOCV evaluate → models/
│   └── classify_swing.py           CLI inference on a raw CSV file
│
├── api/                            FastAPI backend (served via Docker)
│   ├── main.py
│   ├── config.py
│   ├── database.py                 SQLite via aiosqlite
│   ├── schemas.py                  Pydantic request/response models
│   ├── routers/
│   │   ├── analyze.py              POST /analyze
│   │   ├── swings.py               GET/POST /swings
│   │   ├── model.py                GET /model/info, POST /model/retrain
│   │   └── health.py               GET /health
│   ├── services/
│   │   ├── pipeline_service.py     Wraps scripts/ as Python modules
│   │   └── model_service.py        Singleton model loader
│   └── swing_history.db            SQLite database (volume-mounted in Docker)
│
├── golf_swing_analyzer/            Flutter Android app
│   ├── lib/
│   │   ├── main.dart
│   │   ├── core/                   Theme, router, BLE constants
│   │   ├── models/                 Dart models (ImuSample, SwingResult, …)
│   │   ├── services/               BLE service, API client
│   │   ├── providers/              Riverpod state management
│   │   └── screens/                Home, Capture, Result, History, Settings
│   └── android/
│       └── app/src/main/
│           └── AndroidManifest.xml BLE + Internet permissions
│
├── data/
│   ├── raw/                        Good swing captures (label=1)
│   │   └── Suboptimal_Swings/      Bad swing captures (label=0)
│   └── processed/                  swing_features.csv + analysis plots
│
└── models/
    └── swing_classifier.pkl        Trained Random Forest (volume-mounted in Docker)
```

---

## Full Pipeline Explained

### 1. Data Capture

The ESP32-C3 reads all three sensors at ~100 Hz, fuses them through a Madgwick filter to produce yaw/pitch/roll, and streams each row as a CSV line over BLE in 20-byte chunks.

**Field capture (using the Android app):**
The app connects to `ESP32_IMU_GOLF` via BLE, reassembles the 20-byte chunks back into complete CSV lines, and buffers all `ImuSample` objects in memory during the session. When you tap **Stop & Analyze**, the entire buffer is POSTed to the backend as JSON.

**PC-based capture (for labelled dataset building):**
```bash
# Good swing → data/raw/
python scripts/capture_imu_ble.py

# Bad swing → data/raw/Suboptimal_Swings/
# (edit the output_dir in capture_imu_ble.py or use capture_imu.py over USB)
```

### 2. ML Pipeline (offline, run on PC)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Step 1 — extract features from all labeled captures
python scripts/extract_features.py
# → data/processed/swing_features.csv (21 rows × 21 cols)

# Step 2 — explore feature distributions
python scripts/explore_features.py
# → feature_boxplots.png, feature_correlation.png, feature_scatter.png
# → Cohen's d separability table printed to console

# Step 3 — train and evaluate classifier
python scripts/train_classifier.py
# → models/swing_classifier.pkl
# → classifier_confusion_matrices.png, rf_feature_importances.png
```

### 3. Backend API (Docker)

The API wraps the ML pipeline and exposes it over HTTP. The `scripts/` modules are imported directly — no code duplication.

```bash
# First time — build the image (~2-3 min)
docker-compose build

# Start in background
docker-compose up -d

# Check health
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"db_connected":true}

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Volume mounts (data that persists across container restarts):**
- `./models` → `/app/models` — writable so POST /model/retrain saves updated model
- `./api/swing_history.db` → `/app/api/swing_history.db` — SQLite swing history

> **First run requirement:** `api/swing_history.db` must exist as a file before `docker-compose up`. It is created automatically the first time you run the backend natively: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`

### 4. Network — Connecting Phone to Backend

The Android app communicates with the backend over HTTP. The PC is on ethernet; the phone is on WiFi. As long as both are on the same router they can reach each other directly.

**Finding the PC's ethernet IP:**
```powershell
ipconfig
# Look for "Ethernet adapter" → IPv4 Address, e.g. 192.168.100.65
```

Enter `http://192.168.100.65:8000` in the app's Settings screen and tap **Test Connection**.

**If on different networks — use Tailscale (recommended):**

This project uses [Tailscale](https://tailscale.com) to connect the PC and phone across any network configuration without opening firewall ports or caring about subnets.

1. Install Tailscale on the PC (already set up — Tailscale IP: `100.117.105.127`)
2. Install Tailscale on the Android phone from the Play Store
3. Sign in with the same Tailscale account on both devices
4. In the app Settings, enter: `http://100.117.105.127:8000`

Tailscale creates an encrypted peer-to-peer tunnel between devices regardless of whether the PC is on ethernet and the phone is on WiFi, mobile data, or a different network entirely. It also means no Windows Firewall rules are needed.

### 5. Android App

| Screen | Purpose |
|---|---|
| **Home** | Session summary (good/bad ratio bar), recent swing chips, Start Session button |
| **Capture** | BLE connection status, live `omega_mag` / `acc_mag` chart, swing detection overlay, Start/Stop button |
| **Result** | GOOD/BAD verdict banner, confidence %, phase timeline (backswing/downswing/follow-through), feature breakdown with range bars |
| **History** | Paginated list of all past swings, filterable by Good/Bad |
| **Detail** | Full feature breakdown for any historical swing, relabel for retraining |
| **Settings** | API URL, connection test, omega thresholds, model info, Retrain button |

**Build and run:**
```bash
cd golf_swing_analyzer
flutter pub get
flutter run -d <device-id>   # find device ID with: flutter devices
```

---

## How It Works

### Swing Segmentation

`analyze_swing.py` computes the angular velocity magnitude:

```
omega_mag = sqrt(gx² + gy² + gz²)
```

Smoothed with a Gaussian filter (σ=2), then thresholded:
- **Swing start:** `omega_mag > 60 deg/s`
- **Swing end:** `omega_mag < 30 deg/s` (held for minimum 25 samples / ~250 ms)

Each detected swing is split into phases:
- **Backswing** — start → first omega_mag peak in first half
- **Downswing** — backswing top → second omega_mag peak (impact)
- **Follow-through** — impact → end of motion

The Flutter app mirrors this threshold logic in Dart (`swing_detector.dart`) to give real-time visual feedback during capture without a network round-trip.

### Features Extracted per Swing (19 total)

| Group | Features |
|---|---|
| Tempo | `tempo_ratio` (backswing/downswing time — target 3:1), `T_backswing_s`, `T_downswing_s`, `T_total_s` |
| Angular velocity | `omega_peak`, `omega_mean`, `omega_std`, `omega_skew`, `omega_at_top`, `omega_rise_rate`, `transition_dip_ratio` |
| Acceleration | `acc_peak`, `acc_mean`, `acc_std`, `acc_jerk_max` |
| Euler angles | `yaw_range`, `pitch_range`, `roll_range`, `yaw_at_top` |

### Top Discriminating Features (Cohen's d, current dataset)

| Feature | Cohen's d | Good mean | Bad mean | Interpretation |
|---|---|---|---|---|
| `acc_mean` | 1.89 *** | 1.35 g | 1.91 g | Bad swings have much higher average acceleration |
| `omega_mean` | 1.74 *** | 204 °/s | 374 °/s | Bad swings rotate ~2× faster throughout |
| `yaw_at_top` | 1.67 *** | 269° | 218° | Insufficient shoulder turn at backswing top |
| `acc_std` | 1.27 ** | 0.41 | 0.73 | Bad swings are more erratic |
| `acc_peak` | 1.27 ** | 2.18 g | 3.14 g | Higher impact acceleration |
| `omega_at_top` | 1.25 ** | 405 °/s | 839 °/s | Too fast at transition point |

### Classifier

Three models evaluated using **Leave-One-Out CV** (the only valid approach at this sample size):

| Model | LOOCV Accuracy | Bad recall |
|---|---|---|
| Rule-based (acc_mean + omega_mean thresholds) | 61.9% | 80% |
| SVM (RBF kernel) | 71.4% | 0% |
| **Random Forest** | **76.2%** | **40%** |

Random Forest is saved as default. Input features: `acc_mean`, `omega_mean`, `yaw_at_top`, `acc_std`, `omega_at_top`, `T_backswing_s`.

---

## Current Dataset

| Class | Captures | Swings extracted |
|---|---|---|
| Good (label=1) | 8 files | 16 swings |
| Bad (label=0) | 5 files | 5 swings |
| **Total** | **13 files** | **21 swings** |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `POST` | `/analyze` | Segment + extract + classify raw IMU samples |
| `GET` | `/swings` | Paginated swing history (`?page&page_size&label`) |
| `GET` | `/swings/{id}` | Full swing detail |
| `POST` | `/swings/{id}/label` | Relabel a swing for retraining (`{"label": 0}`) |
| `GET` | `/model/info` | Model class, features, training set size |
| `POST` | `/model/retrain` | Re-run train_classifier.py and reload model |

---

## Next Steps

### 1. Capture more bad swing data (highest priority)

Only 5 bad swing samples exist — bad recall is 40% (misses 3 of 5). The model cannot improve meaningfully without more minority class data.

Target: **30 good + 30 bad** before any further model tuning. Use the Android app: capture a swing, relabel it in the Result screen as Good or Bad, then tap **Retrain** in Settings to update the model immediately.

### 2. Label specific fault types

Currently all bad swings are a single class. Splitting into fault categories makes feedback actionable:

| Fault | Likely IMU signature |
|---|---|
| Over-the-top | High `omega_at_top`, low `yaw_at_top` |
| Casting (early release) | Short `T_downswing_s`, high `transition_dip_ratio` |
| Over-swing | High `T_backswing_s`, high `yaw_range` |
| Early extension | Distinct `pitch_range` pattern |

Each would become its own class in a multi-class classifier.

### 3. Improve swing segmentation

The threshold-based segmenter can split one swing into two or merge practice waggles into a swing. Options:
- Use `scipy.signal.find_peaks` with `prominence` and `distance` constraints instead of hysteresis
- Add a calibration swing at the start of each session to auto-tune thresholds per sensor placement

### 4. Richer features

- **DTW distance** to a reference good swing template — captures the shape of the omega_mag curve, not just its statistics
- **Per-phase features** (backswing / downswing / follow-through computed separately) instead of whole-swing aggregates
- **FFT of omega_mag** — frequency content characterises smoothness and timing rhythm

### 5. Model improvements (once 50+ swings)

- Switch from LOOCV to **stratified k-fold** cross-validation
- Grid search SVM `C` and `gamma` — the RBF kernel collapsed at current scale
- Try **LightGBM** — handles class imbalance natively via `scale_pos_weight`

### 6. Tailscale always-on

Set the Docker container to start on PC boot and keep Tailscale active on the phone. The app then works anywhere — on the driving range, at a course — without needing the PC and phone on the same local network.

```powershell
# Set Docker container to auto-start with Windows
docker-compose up -d   # restart: unless-stopped already set in compose file
# Add Docker Desktop to Windows startup apps
```

### 7. Push notifications / session summary

After each session, generate a summary (% good swings, trend vs last session, worst feature) and display it on the Home screen. Could extend to a local notification when analysis is complete.

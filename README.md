# Golf Swing Analyzer

Project workspace for capturing, processing, and analyzing golf swing IMU data with ESP32 + Python.

## Structure

- `firmware/` ESP32 firmware
- `scripts/` Python data pipeline scripts
- `data/raw/` raw captured swings (ignored by Git)
- `data/processed/` processed datasets (ignored by Git)
- `notebooks/` experiments and visualization notebooks

## Quick start

1. Create a virtual environment and install dependencies:
   - `python -m venv venv`
   - `venv\\Scripts\\activate`
   - `pip install -r requirements.txt`
2. Capture IMU data with `scripts/capture_imu.py`.
3. Analyze with `scripts/analyze_swing.py`.
4. Add feature extraction logic in `scripts/extract_features.py`.

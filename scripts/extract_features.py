import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_swing import compute_signals, find_swings

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "swing_features.csv"


def extract_swing_features(df, start_idx, end_idx, label, source_file):
    seg = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)

    if len(seg) < 10:
        return None

    t = seg["t_ms"].values
    omega = seg["omega_smooth"].values
    acc = seg["acc_smooth"].values
    n = len(seg)

    # Phase detection: top of backswing = omega peak in first half
    first_half_end = max(2, n // 2)
    backswing_top_local = int(np.argmax(omega[:first_half_end]))

    second_part = omega[backswing_top_local + 1:]
    if len(second_part) < 3:
        return None
    impact_local = backswing_top_local + 1 + int(np.argmax(second_part))

    t_start = t[0]
    t_top = t[backswing_top_local]
    t_impact = t[impact_local]
    t_end = t[-1]

    T_backswing = (t_top - t_start) / 1000.0
    T_downswing = (t_impact - t_top) / 1000.0
    T_total = (t_end - t_start) / 1000.0

    if T_downswing <= 0 or T_backswing <= 0:
        return None

    # --- Tempo ---
    tempo_ratio = T_backswing / T_downswing

    # --- Angular velocity ---
    omega_peak = float(np.max(omega))
    omega_mean = float(np.mean(omega))
    omega_std = float(np.std(omega))
    omega_skew = float(skew(omega))
    omega_at_top = float(omega[backswing_top_local])
    omega_impact_peak = float(omega[impact_local])

    # Average slope of omega from swing start to backswing peak
    omega_rise_rate = (omega[backswing_top_local] - omega[0]) / T_backswing

    # How much omega drops at transition relative to impact peak.
    # A lower ratio = more abrupt direction reversal (suboptimal).
    if omega_impact_peak > 0:
        transition_dip_ratio = omega_at_top / omega_impact_peak
    else:
        transition_dip_ratio = float("nan")

    # --- Acceleration ---
    acc_peak = float(np.max(acc))
    acc_mean = float(np.mean(acc))
    acc_std = float(np.std(acc))

    dt_s = np.diff(t) / 1000.0
    dt_s = np.where(dt_s == 0, 1e-9, dt_s)
    jerk = np.abs(np.diff(acc)) / dt_s
    acc_jerk_max = float(np.max(jerk))

    # --- Euler angles ---
    yaw = seg["yaw"].values
    pitch = seg["pitch"].values
    roll = seg["roll"].values

    yaw_range = float(np.ptp(yaw))
    pitch_range = float(np.ptp(pitch))
    roll_range = float(np.ptp(roll))
    yaw_at_top = float(yaw[backswing_top_local])

    return {
        "source_file": source_file,
        "label": label,
        # Tempo
        "tempo_ratio": round(tempo_ratio, 4),
        "T_backswing_s": round(T_backswing, 4),
        "T_downswing_s": round(T_downswing, 4),
        "T_total_s": round(T_total, 4),
        # Angular velocity
        "omega_peak": round(omega_peak, 4),
        "omega_mean": round(omega_mean, 4),
        "omega_std": round(omega_std, 4),
        "omega_skew": round(omega_skew, 4),
        "omega_at_top": round(omega_at_top, 4),
        "omega_rise_rate": round(omega_rise_rate, 4),
        "transition_dip_ratio": round(transition_dip_ratio, 4) if not np.isnan(transition_dip_ratio) else float("nan"),
        # Acceleration
        "acc_peak": round(acc_peak, 4),
        "acc_mean": round(acc_mean, 4),
        "acc_std": round(acc_std, 4),
        "acc_jerk_max": round(acc_jerk_max, 4),
        # Euler angles
        "yaw_range": round(yaw_range, 4),
        "pitch_range": round(pitch_range, 4),
        "roll_range": round(roll_range, 4),
        "yaw_at_top": round(yaw_at_top, 4),
    }


def process_file(csv_path, label):
    df = pd.read_csv(csv_path)
    df = compute_signals(df)
    swings = find_swings(df)

    records = []
    for start_idx, end_idx in swings:
        feat = extract_swing_features(df, start_idx, end_idx, label, csv_path.name)
        if feat is not None:
            records.append(feat)
    return records


def main():
    good_files = sorted(RAW_DIR.glob("swing_capture_*.csv"))
    bad_files = sorted((RAW_DIR / "Suboptimal_Swings").glob("swing_capture_*.csv"))

    all_records = []

    print(f"Processing {len(good_files)} good swing file(s)...")
    for f in good_files:
        records = process_file(f, label=1)
        print(f"  {f.name}: {len(records)} swing(s) detected")
        all_records.extend(records)

    print(f"\nProcessing {len(bad_files)} bad swing file(s)...")
    for f in bad_files:
        records = process_file(f, label=0)
        print(f"  {f.name}: {len(records)} swing(s) detected")
        all_records.extend(records)

    if not all_records:
        print("No swings detected. Check START_THRESHOLD / END_THRESHOLD in analyze_swing.py.")
        return

    df_out = pd.DataFrame(all_records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    n_good = (df_out["label"] == 1).sum()
    n_bad = (df_out["label"] == 0).sum()
    print(f"\nTotal swings extracted: {len(df_out)}  (good={n_good}, bad={n_bad})")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

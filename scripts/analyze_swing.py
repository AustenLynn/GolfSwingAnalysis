import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "swing_features.csv"

START_THRESHOLD = 60.0   # deg/s, ajústalo
END_THRESHOLD = 30.0     # deg/s, ajústalo
MIN_SWING_SAMPLES = 25   # mínimo ~250 ms si estás a 100 Hz

def compute_signals(df):
    df = df.copy()

    df["omega_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    df["omega_smooth"] = gaussian_filter1d(df["omega_mag"].values, sigma=2)
    df["acc_smooth"] = gaussian_filter1d(df["acc_mag"].values, sigma=2)

    return df

def find_swings(df):
    signal = df["omega_smooth"].values
    swings = []

    in_swing = False
    start_idx = None

    for i, val in enumerate(signal):
        if not in_swing and val > START_THRESHOLD:
            in_swing = True
            start_idx = i

        elif in_swing and val < END_THRESHOLD:
            end_idx = i
            if end_idx - start_idx >= MIN_SWING_SAMPLES:
                swings.append((start_idx, end_idx))
            in_swing = False
            start_idx = None

    if in_swing and start_idx is not None:
        end_idx = len(signal) - 1
        if end_idx - start_idx >= MIN_SWING_SAMPLES:
            swings.append((start_idx, end_idx))

    return swings

def analyze_single_swing(df, start_idx, end_idx):
    seg = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)

    t = seg["t_ms"].values
    omega = seg["omega_smooth"].values
    acc = seg["acc_smooth"].values

    n = len(seg)
    if n < 10:
        return None

    # top of backswing = máximo en primera mitad
    first_half_end = max(2, n // 2)
    backswing_top_local = np.argmax(omega[:first_half_end])

    # impacto aproximado = máximo después del top
    second_part = omega[backswing_top_local + 1:]
    if len(second_part) < 3:
        return None

    impact_local = backswing_top_local + 1 + np.argmax(second_part)

    t_start = t[0]
    t_top = t[backswing_top_local]
    t_impact = t[impact_local]
    t_end = t[-1]

    T_backswing = (t_top - t_start) / 1000.0
    T_downswing = (t_impact - t_top) / 1000.0
    T_total = (t_end - t_start) / 1000.0

    if T_downswing <= 0:
        return None

    tempo_ratio = T_backswing / T_downswing

    result = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "t_start_ms": t_start,
        "t_top_ms": t_top,
        "t_impact_ms": t_impact,
        "t_end_ms": t_end,
        "T_backswing_s": T_backswing,
        "T_downswing_s": T_downswing,
        "T_total_s": T_total,
        "tempo_ratio": tempo_ratio,
        "omega_peak": float(np.max(omega)),
        "acc_peak": float(np.max(acc)),
        "backswing_top_local": int(backswing_top_local),
        "impact_local": int(impact_local),
    }

    return result

def score_tempo(tempo_ratio):
    # score simple respecto a 3:1
    target = 3.0
    error = abs(tempo_ratio - target)
    score = max(0, 100 - (error * 40))
    return score

def plot_swings(df, swing_results):
    plt.figure(figsize=(14, 6))
    plt.plot(df["t_ms"], df["omega_smooth"], label="omega_smooth")

    for res in swing_results:
        s = res["start_idx"]
        e = res["end_idx"]

        plt.axvspan(df.iloc[s]["t_ms"], df.iloc[e]["t_ms"], alpha=0.2)
        plt.axvline(res["t_top_ms"], linestyle="--")
        plt.axvline(res["t_impact_ms"], linestyle=":")

    plt.xlabel("Tiempo [ms]")
    plt.ylabel("Velocidad angular mag [deg/s]")
    plt.title("Detección de swings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    capture_files = sorted(RAW_DIR.glob("swing_capture_*.csv"), key=lambda p: p.stat().st_mtime)
    if not capture_files:
        print(f"No hay archivos swing_capture_*.csv en: {RAW_DIR}")
        return

    input_file = capture_files[-1]
    print(f"Analizando archivo: {input_file}")

    df = pd.read_csv(input_file)
    df = compute_signals(df)

    swings = find_swings(df)
    print(f"Swings detectados: {len(swings)}")

    results = []
    for i, (s, e) in enumerate(swings, start=1):
        res = analyze_single_swing(df, s, e)
        if res is None:
            continue

        res["swing_id"] = i
        res["tempo_score"] = score_tempo(res["tempo_ratio"])
        results.append(res)

    if not results:
        print("No se pudieron analizar swings.")
        return

    results_df = pd.DataFrame(results)
    print("\nResultados:")
    print(results_df[[
        "swing_id",
        "T_backswing_s",
        "T_downswing_s",
        "tempo_ratio",
        "tempo_score",
        "omega_peak",
        "acc_peak"
    ]])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nGuardado: {OUTPUT_FILE}")

    plot_swings(df, results)

if __name__ == "__main__":
    main()
import sys
import math
import pandas as pd
from pathlib import Path
from api.config import SCRIPTS_DIR

# Make existing scripts importable as modules
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_swing import compute_signals, find_swings       # noqa: E402
from extract_features import extract_swing_features          # noqa: E402
from classify_swing import predict_swing, feature_contributions  # noqa: E402

# Good/bad reference means from exploration (for feature verdict annotation)
_GOOD_MEANS = {
    "acc_mean": 1.347, "omega_mean": 203.7, "yaw_at_top": 269.1,
    "acc_std": 0.410,  "omega_at_top": 405.4, "T_backswing_s": 0.614,
    "tempo_ratio": 12.7, "T_downswing_s": 0.171, "T_total_s": 1.599,
    "omega_peak": 641.4, "omega_std": 163.5, "omega_skew": 0.857,
    "omega_rise_rate": 621.1, "transition_dip_ratio": 0.774,
    "acc_peak": 2.177, "acc_jerk_max": 10.2, "yaw_range": 4.21,
    "pitch_range": 7.61, "roll_range": 6.45,
}
_BAD_MEANS = {
    "acc_mean": 1.912, "omega_mean": 374.1, "yaw_at_top": 218.1,
    "acc_std": 0.734,  "omega_at_top": 839.3, "T_backswing_s": 0.907,
    "tempo_ratio": 18.4, "T_downswing_s": 0.061, "T_total_s": 1.723,
    "omega_peak": 1021.7, "omega_std": 293.4, "omega_skew": 0.718,
    "omega_rise_rate": 923.0, "transition_dip_ratio": 0.812,
    "acc_peak": 3.144, "acc_jerk_max": 14.3, "yaw_range": 5.97,
    "pitch_range": 7.92, "roll_range": 6.27,
}


def _feature_verdict(feature: str, value: float) -> str:
    """Classify a feature value as closer to good or bad mean."""
    gm = _GOOD_MEANS.get(feature)
    bm = _BAD_MEANS.get(feature)
    if gm is None or bm is None:
        return "neutral"
    dist_good = abs(value - gm)
    dist_bad = abs(value - bm)
    return "good" if dist_good <= dist_bad else "bad"


def analyze_samples(samples: list[dict], model_obj: dict) -> list[dict]:
    """
    Run the full pipeline on a list of raw IMU sample dicts.
    Returns a list of swing result dicts (one per detected swing).
    """
    df = pd.DataFrame(samples)
    df = compute_signals(df)
    swings = find_swings(df)

    results = []
    for i, (start_idx, end_idx) in enumerate(swings):
        features = extract_swing_features(
            df, start_idx, end_idx, label=None, source_file="ble_capture"
        )
        if features is None:
            continue

        label, confidence = predict_swing(model_obj, features)
        importances_raw = feature_contributions(model_obj, features) or []

        # Build feature_importances list
        feature_importances = []
        for feat_name, imp, val in importances_raw:
            if math.isnan(val):
                val = 0.0
            feature_importances.append({
                "feature": feat_name,
                "importance": round(float(imp), 4),
                "value": round(float(val), 4),
                "verdict": _feature_verdict(feat_name, float(val)),
            })

        # Phase timing
        seg = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        t = seg["t_ms"].values
        omega = seg["omega_smooth"].values
        n = len(seg)
        first_half_end = max(2, n // 2)
        top_local = int(pd.Series(omega[:first_half_end]).idxmax())
        second_part = omega[top_local + 1:]
        impact_local = top_local + 1 + int(pd.Series(second_part).idxmax()) if len(second_part) >= 3 else top_local

        t_start = float(t[0])
        t_top = float(t[top_local])
        t_impact = float(t[impact_local])
        t_end = float(t[-1])
        T_back = (t_top - t_start) / 1000.0
        T_down = (t_impact - t_top) / 1000.0
        tempo = T_back / T_down if T_down > 0 else 0.0

        # Clean features dict (handle NaN)
        clean_features = {}
        for k, v in features.items():
            if k in ("source_file", "label"):
                continue
            clean_features[k] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)

        results.append({
            "swing_index": i,
            "verdict": "GOOD" if label == 1 else "BAD",
            "label": int(label),
            "confidence": round(float(confidence) if confidence is not None else 0.5, 4),
            "features": clean_features,
            "feature_importances": feature_importances,
            "phase_timing": {
                "t_start_ms": t_start,
                "t_top_ms": t_top,
                "t_impact_ms": t_impact,
                "t_end_ms": t_end,
                "T_backswing_s": round(T_back, 4),
                "T_downswing_s": round(T_down, 4),
                "tempo_ratio": round(tempo, 4),
            },
        })

    return results

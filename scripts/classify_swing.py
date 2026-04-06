"""
Classify good/bad swings from a new raw IMU capture file.

Usage
-----
    python scripts/classify_swing.py data/raw/my_capture.csv

Output
------
    Per-swing verdict (Good / Bad) with confidence score.
    Prints the top contributing features for each prediction.
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_swing import compute_signals, find_swings
from extract_features import extract_swing_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "swing_classifier.pkl"

LABEL_STR = {1: "GOOD", 0: "BAD"}
LABEL_COLOR = {1: "\033[92m", 0: "\033[91m"}  # green / red
RESET = "\033[0m"


def load_model():
    if not MODEL_FILE.exists():
        print(f"Model not found: {MODEL_FILE}")
        print("Run train_classifier.py first.")
        sys.exit(1)
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict_swing(model_obj, feat_dict):
    features = model_obj["features"]
    model = model_obj["model"]
    model_type = model_obj["type"]

    row = pd.DataFrame([{k: feat_dict[k] for k in features}])

    if model_type == "rule_based":
        label = int(model.predict(row)[0])
        confidence = None
    else:
        label = int(model.predict(row.values)[0])
        proba = model.predict_proba(row.values)[0]
        confidence = float(proba[label])

    return label, confidence


def feature_contributions(model_obj, feat_dict):
    """Return feature values ranked by importance (RF only)."""
    model_type = model_obj["type"]
    if model_type != "sklearn":
        return None

    clf = model_obj["model"].named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None

    features = model_obj["features"]
    importances = clf.feature_importances_
    ranked = sorted(zip(features, importances, [feat_dict[f] for f in features]),
                    key=lambda x: -x[1])
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Classify swings in a new IMU capture file.")
    parser.add_argument("csv_file", help="Path to raw IMU CSV file")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    model_obj = load_model()

    df = pd.read_csv(csv_path)
    df = compute_signals(df)
    swings = find_swings(df)

    if not swings:
        print(f"No swings detected in {csv_path.name}.")
        print("Check that the file contains a full swing above the motion threshold.")
        sys.exit(0)

    print(f"\nFile : {csv_path.name}")
    print(f"Swings detected: {len(swings)}\n")
    print(f"{'Swing':<8} {'Verdict':<10} {'Confidence':<14} Top features")
    print("-" * 72)

    for i, (start_idx, end_idx) in enumerate(swings, start=1):
        feat = extract_swing_features(df, start_idx, end_idx, label=None, source_file=csv_path.name)
        if feat is None:
            print(f"  #{i:<5}  (segment too short, skipped)")
            continue

        label, confidence = predict_swing(model_obj, feat)
        label_str = LABEL_STR[label]
        conf_str = f"{confidence:.1%}" if confidence is not None else "N/A"
        color = LABEL_COLOR[label]

        contributions = feature_contributions(model_obj, feat)
        if contributions:
            top3 = "  |  ".join(f"{f}={v:.3f} (imp {imp:.2f})"
                                for f, imp, v in contributions[:3])
        else:
            top3 = "  |  ".join(f"{f}={feat[f]:.3f}" for f in model_obj["features"][:3])

        print(f"  #{i:<5}  {color}{label_str:<10}{RESET} {conf_str:<14} {top3}")

    print()


if __name__ == "__main__":
    main()

"""
Train a good/bad golf swing classifier.

Strategy
--------
- Dataset is tiny (~21 swings), so we evaluate with Leave-One-Out CV.
- Three models are compared:
    1. Rule-based threshold (interpretable, no training required)
    2. SVM with RBF kernel
    3. Random Forest
- Features are selected from the top discriminators found in explore_features.py.
- The best model is saved to models/swing_classifier.pkl for use in classify_swing.py.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FEATURES_FILE = PROCESSED_DIR / "swing_features.csv"
MODEL_FILE = MODELS_DIR / "swing_classifier.pkl"

# Top features by Cohen's d from explore_features.py
# (very large + large effect, not correlated with each other)
SELECTED_FEATURES = [
    "acc_mean",
    "omega_mean",
    "yaw_at_top",
    "acc_std",
    "omega_at_top",
    "T_backswing_s",
]


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """
    Classify as bad (0) if acc_mean and omega_mean are both above thresholds
    learned from the training set medians.
    """

    def __init__(self):
        self.acc_mean_thresh = None
        self.omega_mean_thresh = None

    def fit(self, X_df, y):
        good = X_df[y == 1]
        bad = X_df[y == 0]
        # Threshold = midpoint between class means
        self.acc_mean_thresh = (good["acc_mean"].mean() + bad["acc_mean"].mean()) / 2
        self.omega_mean_thresh = (good["omega_mean"].mean() + bad["omega_mean"].mean()) / 2
        return self

    def predict(self, X_df):
        preds = []
        for _, row in X_df.iterrows():
            if row["acc_mean"] > self.acc_mean_thresh and row["omega_mean"] > self.omega_mean_thresh:
                preds.append(0)  # bad
            else:
                preds.append(1)  # good
        return np.array(preds)


def loocv_rule_based(df, y):
    """Manual LOOCV for the rule-based classifier (no sklearn Pipeline support)."""
    preds = np.zeros(len(df), dtype=int)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(df):
        clf = RuleBasedClassifier()
        clf.fit(df.iloc[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(df.iloc[test_idx])
    return preds


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  LOOCV Accuracy : {acc:.1%}  ({int(acc * len(y_true))}/{len(y_true)} correct)")
    print(classification_report(y_true, y_pred, target_names=["Bad (0)", "Good (1)"],
                                 zero_division=0))
    return acc, cm


def plot_confusion_matrices(results, y_true):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, _, cm) in zip(axes, results):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nLOOCV Accuracy: {accuracy_score(y_true, _):.1%}")

    plt.suptitle("Confusion Matrices — Leave-One-Out CV", fontsize=13)
    plt.tight_layout()
    out = PROCESSED_DIR / "classifier_confusion_matrices.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {out}")


def plot_feature_importances(rf_model, feature_names):
    importances = rf_model.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(importances)), importances[idx], color="#4CAF50", alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = PROCESSED_DIR / "rf_feature_importances.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not FEATURES_FILE.exists():
        print(f"Features file not found: {FEATURES_FILE}")
        print("Run extract_features.py first.")
        return

    df = pd.read_csv(FEATURES_FILE)
    y = df["label"].values.astype(int)
    X_df = df[SELECTED_FEATURES].copy()
    X = X_df.values

    n_good = (y == 1).sum()
    n_bad = (y == 0).sum()
    print(f"Dataset: {len(df)} swings  (good={n_good}, bad={n_bad})")
    print(f"Features: {SELECTED_FEATURES}")
    print(f"Validation: Leave-One-Out CV\n")

    loo = LeaveOneOut()

    # --- 1. Rule-based ---
    preds_rule = loocv_rule_based(X_df, y)
    acc_rule, cm_rule = evaluate("Rule-Based (acc_mean + omega_mean thresholds)", y, preds_rule)

    # --- 2. SVM (RBF) ---
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ])
    preds_svm = cross_val_predict(svm_pipe, X, y, cv=loo)
    acc_svm, cm_svm = evaluate("SVM (RBF kernel)", y, preds_svm)

    # --- 3. Random Forest ---
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42,
                                        class_weight="balanced"))
    ])
    preds_rf = cross_val_predict(rf_pipe, X, y, cv=loo)
    acc_rf, cm_rf = evaluate("Random Forest", y, preds_rf)

    # --- Summary ---
    results = [
        ("Rule-Based", preds_rule, cm_rule),
        ("SVM (RBF)", preds_svm, cm_svm),
        ("Random Forest", preds_rf, cm_rf),
    ]
    print("\n--- Summary ---")
    for name, preds, _ in results:
        print(f"  {name:<35} {accuracy_score(y, preds):.1%}")

    # --- Pick best model by accuracy (SVM preferred on ties — lower variance) ---
    scores = [acc_rule, acc_svm, acc_rf]
    best_idx = int(np.argmax(scores))
    best_name = ["Rule-Based", "SVM (RBF)", "Random Forest"][best_idx]
    print(f"\nBest model: {best_name}  ({scores[best_idx]:.1%})")

    # --- Refit best ML model on full dataset and save ---
    if best_idx == 0:
        final_model = RuleBasedClassifier()
        final_model.fit(X_df, y)
        save_obj = {"type": "rule_based", "model": final_model, "features": SELECTED_FEATURES}
    else:
        best_pipe = svm_pipe if best_idx == 1 else rf_pipe
        best_pipe.fit(X, y)
        save_obj = {"type": "sklearn", "model": best_pipe, "features": SELECTED_FEATURES}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"Model saved to: {MODEL_FILE}")

    # --- Plots ---
    plot_confusion_matrices(results, y)

    # Feature importances only for Random Forest
    rf_pipe.fit(X, y)
    plot_feature_importances(rf_pipe, SELECTED_FEATURES)


if __name__ == "__main__":
    main()

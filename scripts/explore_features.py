import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "swing_features.csv"

FEATURE_COLS = [
    "tempo_ratio",
    "T_backswing_s",
    "T_downswing_s",
    "T_total_s",
    "omega_peak",
    "omega_mean",
    "omega_std",
    "omega_skew",
    "omega_at_top",
    "omega_rise_rate",
    "transition_dip_ratio",
    "acc_peak",
    "acc_mean",
    "acc_std",
    "acc_jerk_max",
    "yaw_range",
    "pitch_range",
    "roll_range",
    "yaw_at_top",
]

COLOR_GOOD = "#4CAF50"
COLOR_BAD = "#F44336"


def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return abs(np.mean(a) - np.mean(b)) / pooled_std


def print_separability(df):
    good = df[df["label"] == 1]
    bad = df[df["label"] == 0]

    print("\n--- Feature Separability (Cohen's d) ---")
    print(f"{'Feature':<25} {'Good mean':>12} {'Bad mean':>12} {'Cohen d':>10}  Effect")
    print("-" * 68)

    results = []
    for feat in FEATURE_COLS:
        g = good[feat].dropna().values
        b = bad[feat].dropna().values
        if len(g) < 2 or len(b) < 2:
            continue
        d = cohens_d(g, b)
        results.append((feat, float(np.mean(g)), float(np.mean(b)), d))

    results.sort(key=lambda x: -x[3])
    for feat, gm, bm, d in results:
        strength = "*** very large" if d > 1.5 else "**  large" if d > 0.8 else "*   medium" if d > 0.4 else "    small"
        print(f"{feat:<25} {gm:>12.3f} {bm:>12.3f} {d:>10.3f}  {strength}")

    print("\n  *** d>1.5   ** d>0.8   * d>0.4")
    return results


def plot_boxplots(df):
    good = df[df["label"] == 1]
    bad = df[df["label"] == 0]

    n_cols = 4
    n_rows = -(-len(FEATURE_COLS) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes[i]
        data = [good[feat].dropna().values, bad[feat].dropna().values]
        bp = ax.boxplot(data, tick_labels=["Good", "Bad"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(COLOR_GOOD)
        bp["boxes"][1].set_facecolor(COLOR_BAD)
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions: Good vs Bad Swings", fontsize=13)
    plt.tight_layout()
    out = PROCESSED_DIR / "feature_boxplots.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_correlation_heatmap(df):
    corr = df[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ticks = range(len(FEATURE_COLS))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURE_COLS, fontsize=8)
    ax.set_title("Feature Correlation Matrix", fontsize=13)

    # Annotate cells with values
    for r in range(len(FEATURE_COLS)):
        for c in range(len(FEATURE_COLS)):
            val = corr.values[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > 0.6 else "black")

    plt.tight_layout()
    out = PROCESSED_DIR / "feature_correlation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_top_features_scatter(df, separability_results, top_n=6):
    """Scatter plot of the top-N most separable feature pairs (first two features)."""
    top = [r[0] for r in separability_results[:top_n]]
    if len(top) < 2:
        return

    good = df[df["label"] == 1]
    bad = df[df["label"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (fx, fy) in zip(axes, [(top[0], top[1]), (top[2], top[3])]):
        ax.scatter(good[fx], good[fy], c=COLOR_GOOD, label="Good", alpha=0.8, edgecolors="k", linewidths=0.5)
        ax.scatter(bad[fx], bad[fy], c=COLOR_BAD, label="Bad", alpha=0.8, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{fx} vs {fy}")

    fig.suptitle("Top Discriminative Feature Pairs", fontsize=13)
    plt.tight_layout()
    out = PROCESSED_DIR / "feature_scatter.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def main():
    if not FEATURES_FILE.exists():
        print(f"Features file not found: {FEATURES_FILE}")
        print("Run extract_features.py first.")
        return

    df = pd.read_csv(FEATURES_FILE)
    n_good = (df["label"] == 1).sum()
    n_bad = (df["label"] == 0).sum()
    print(f"Loaded {len(df)} swings: {n_good} good, {n_bad} bad")
    print(f"Files: {df['source_file'].nunique()} unique capture files")

    sep = print_separability(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    if len(sep) >= 4:
        plot_top_features_scatter(df, sep)


if __name__ == "__main__":
    main()

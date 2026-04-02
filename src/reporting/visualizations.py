"""
visualizations.py
-----------------
All charts for the A/B testing framework.

Usage:
    from src.reporting.visualizations import plot_all_ab
    plot_all_ab(df, results, experiment_name="homepage_cta")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict

COLORS = {
    "control":   "#6B7280",
    "treatment": "#2563EB",
    "positive":  "#059669",
    "negative":  "#EF4444",
    "neutral":   "#F59E0B",
    "ci":        "#BFDBFE",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {path}")
    return path


# ── 1. Conversion Rate Comparison ────────────────────────────────────────────
def plot_conversion_rates(df: pd.DataFrame, metric: str, exp_name: str) -> str:
    ctrl_rate = df[df["group"] == "control"][metric].mean()
    trt_rate  = df[df["group"] == "treatment"][metric].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Control", "Treatment"],
        [ctrl_rate, trt_rate],
        color=[COLORS["control"], COLORS["treatment"]],
        width=0.5, edgecolor="white",
    )
    for bar, val in zip(bars, [ctrl_rate, trt_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.2%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    lift = trt_rate - ctrl_rate
    color = COLORS["positive"] if lift > 0 else COLORS["negative"]
    ax.annotate(
        f"Δ = {lift:+.2%}",
        xy=(1, trt_rate), xytext=(1.35, (ctrl_rate + trt_rate)/2),
        fontsize=11, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color),
    )
    ax.set_ylim(0, max(ctrl_rate, trt_rate) * 1.3)
    ax.set_ylabel(f"{metric} Rate")
    ax.set_title(f"Conversion Rate by Group\n{exp_name.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout()
    return _save(fig, f"{exp_name}_01_conversion_rates.png")


# ── 2. Bootstrap Distribution ─────────────────────────────────────────────────
def plot_bootstrap_distribution(
    df: pd.DataFrame, metric: str, exp_name: str, n_bootstrap: int = 500
) -> str:
    np.random.seed(42)
    ctrl = df[df["group"] == "control"][metric].values
    trt  = df[df["group"] == "treatment"][metric].values
    # Vectorized bootstrap — much faster than a loop
    ctrl_boot = np.random.choice(ctrl, (n_bootstrap, len(ctrl)), replace=True).mean(axis=1)
    trt_boot  = np.random.choice(trt,  (n_bootstrap, len(trt)),  replace=True).mean(axis=1)
    diffs = trt_boot - ctrl_boot
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diffs, bins=60, color=COLORS["treatment"], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="No effect")
    ax.axvline(diffs.mean(), color=COLORS["positive"], lw=2, label=f"Observed Δ = {diffs.mean():+.4f}")
    ax.axvspan(lo, hi, alpha=0.15, color=COLORS["ci"], label=f"95% CI [{lo:+.4f}, {hi:+.4f}]")

    ax.set_xlabel("Bootstrapped Difference in Means (Treatment − Control)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution of Treatment Effect",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _save(fig, f"{exp_name}_02_bootstrap_distribution.png")


# ── 3. Segment Lift Heatmap ───────────────────────────────────────────────────
def plot_segment_lift(df: pd.DataFrame, metric: str, exp_name: str) -> str:
    seg_cols = [c for c in ["device", "user_type", "region"] if c in df.columns]
    if not seg_cols:
        return None

    fig, axes = plt.subplots(1, len(seg_cols), figsize=(5 * len(seg_cols), 5))
    if len(seg_cols) == 1:
        axes = [axes]

    for ax, seg_col in zip(axes, seg_cols):
        rows = []
        for val in df[seg_col].unique():
            sub = df[df[seg_col] == val]
            c_rate = sub[sub["group"] == "control"][metric].mean()
            t_rate = sub[sub["group"] == "treatment"][metric].mean()
            rows.append({"segment": val, "lift": t_rate - c_rate,
                         "control": c_rate, "treatment": t_rate})
        seg_df = pd.DataFrame(rows).sort_values("lift", ascending=True)

        colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in seg_df["lift"]]
        bars = ax.barh(seg_df["segment"], seg_df["lift"], color=colors, height=0.5, edgecolor="white")
        ax.axvline(0, color="black", lw=1, linestyle="--")
        for bar, val in zip(bars, seg_df["lift"]):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                    f"{val:+.2%}", va="center", fontsize=9)
        ax.set_title(f"Lift by {seg_col.replace('_',' ').title()}", fontweight="bold")
        ax.set_xlabel("Absolute Lift")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.suptitle(f"Segment-Level Treatment Effect — {exp_name.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, f"{exp_name}_03_segment_lift.png")


# ── 4. Propensity Score Overlap ───────────────────────────────────────────────
def plot_ps_overlap(df: pd.DataFrame, ps_col: str = "propensity_score",
                    treatment_col: str = "treated", exp_name: str = "observational") -> str:
    if ps_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for val, label, color in [(0, "Control", COLORS["control"]),
                               (1, "Treatment", COLORS["treatment"])]:
        ps = df[df[treatment_col] == val][ps_col]
        ax.hist(ps, bins=40, alpha=0.6, color=color, label=label, edgecolor="white", density=True)

    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap (Before Matching)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _save(fig, f"{exp_name}_04_ps_overlap.png")


# ── Master caller ─────────────────────────────────────────────────────────────
def plot_all_ab(df: pd.DataFrame, results: Dict, experiment_name: str):
    metric = results["experiment_summary"]["primary_metric"]
    print(f"\n📊 Generating A/B test figures for: {experiment_name}")
    plot_conversion_rates(df, metric, experiment_name)
    plot_bootstrap_distribution(df, metric, experiment_name)
    plot_segment_lift(df, metric, experiment_name)
    print("✅ All A/B figures saved to outputs/figures/")

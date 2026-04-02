"""
data_generator.py
-----------------
Generates realistic synthetic A/B experiment datasets
for multiple experiment types (proportion, continuous, observational).

Usage:
    from src.pipeline.data_generator import generate_experiment_data
    df = generate_experiment_data(n=5000, true_effect=0.025)
"""

import numpy as np
import pandas as pd
import os
from typing import Optional

RANDOM_SEED = 42


def generate_experiment_data(
    n: int = 10000,
    true_effect: float = 0.025,
    baseline_rate: float = 0.12,
    control_pct: float = 0.50,
    add_segments: bool = True,
    experiment_name: str = "homepage_cta",
    output_dir: str = "data/raw",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic A/B experiment dataset.

    Parameters
    ----------
    n            : Total number of experiment units
    true_effect  : True treatment effect on conversion rate (absolute)
    baseline_rate: Control group conversion rate
    control_pct  : Fraction of units in control
    add_segments : Include user segments (device, region, user_type)
    experiment_name : Used in filename

    Returns
    -------
    pd.DataFrame with columns:
        user_id, group, device, region, user_type,
        days_since_signup, converted, revenue, sessions
    """
    np.random.seed(seed)

    n_control   = int(n * control_pct)
    n_treatment = n - n_control

    groups  = ["control"] * n_control + ["treatment"] * n_treatment
    user_ids = [f"u_{i:06d}" for i in range(n)]

    # User features (potential confounders — important for PSM)
    devices    = np.random.choice(["mobile", "desktop", "tablet"],
                                  size=n, p=[0.55, 0.37, 0.08])
    regions    = np.random.choice(["Northeast", "Southeast", "Midwest", "West"],
                                  size=n, p=[0.30, 0.25, 0.20, 0.25])
    user_types = np.random.choice(["new", "returning", "power"],
                                  size=n, p=[0.40, 0.45, 0.15])
    days_since = np.random.exponential(scale=120, size=n).clip(1, 1500).astype(int)

    # Conversion probabilities
    convert_prob = np.where(
        np.array(groups) == "control",
        baseline_rate,
        baseline_rate + true_effect,
    )
    # Heterogeneous effects: mobile users benefit more from treatment
    mobile_mask = devices == "mobile"
    convert_prob = np.where(
        (np.array(groups) == "treatment") & mobile_mask,
        convert_prob + 0.008,
        convert_prob,
    )
    converted = np.random.binomial(1, convert_prob)

    # Revenue (only for converters, log-normal)
    revenue = np.where(
        converted == 1,
        np.random.lognormal(mean=3.5, sigma=0.8, size=n).clip(5, 500),
        0.0,
    )
    # Treatment group gets slightly higher revenue per conversion
    revenue = np.where(
        (np.array(groups) == "treatment") & (converted == 1),
        revenue * 1.06,
        revenue,
    )

    # Sessions (engagement metric — guardrail)
    sessions = np.random.poisson(lam=3.2, size=n).clip(1, 20)

    df = pd.DataFrame({
        "user_id":         user_ids,
        "group":           groups,
        "device":          devices,
        "region":          regions,
        "user_type":       user_types,
        "days_since_signup": days_since,
        "converted":       converted,
        "revenue":         revenue.round(2),
        "sessions":        sessions,
    })

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}.csv")
    df.to_csv(path, index=False)

    n_c = (df["group"] == "control").sum()
    n_t = (df["group"] == "treatment").sum()
    print(f"✅ Experiment data saved → {path}")
    print(f"   N={len(df):,}  |  Control={n_c:,}  Treatment={n_t:,}")
    print(f"   True effect: {true_effect:+.1%}  (baseline={baseline_rate:.1%})")
    print(f"   Control conversion:   {df[df['group']=='control']['converted'].mean():.1%}")
    print(f"   Treatment conversion: {df[df['group']=='treatment']['converted'].mean():.1%}")

    return df


def generate_observational_data(
    n: int = 8000,
    true_ate: float = 0.03,
    output_dir: str = "data/raw",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate observational (non-randomized) data with confounding,
    suitable for propensity score matching.

    Here, 'power users' are more likely to self-select into treatment
    AND more likely to convert — a classic confounder.
    """
    np.random.seed(seed)

    user_type  = np.random.choice(["new", "returning", "power"], size=n, p=[0.40, 0.45, 0.15])
    device     = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.37, 0.08])
    region     = np.random.choice(["Northeast", "Southeast", "Midwest", "West"],
                                  size=n, p=[0.30, 0.25, 0.20, 0.25])
    days_since = np.random.exponential(120, n).clip(1, 1500).astype(int)

    # Treatment selection — confounded by user_type and days_since_signup
    power_mask     = user_type == "power"
    returning_mask = user_type == "returning"
    treat_prob = (
        0.25
        + 0.25 * power_mask
        + 0.10 * returning_mask
        + 0.10 * (device == "desktop")
        - 0.05 * (days_since > 365)
    ).clip(0.05, 0.90)
    treated = np.random.binomial(1, treat_prob)

    # Outcome — confounded by user_type + treatment
    baseline = 0.10 + 0.08 * power_mask + 0.03 * returning_mask
    outcome_prob = (baseline + true_ate * treated).clip(0.01, 0.99)
    converted = np.random.binomial(1, outcome_prob)

    df = pd.DataFrame({
        "user_id":          [f"o_{i:06d}" for i in range(n)],
        "treated":          treated,
        "user_type":        user_type,
        "device":           device,
        "region":           region,
        "days_since_signup": days_since,
        "converted":        converted,
    })

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "observational_study.csv")
    df.to_csv(path, index=False)
    print(f"✅ Observational data saved → {path}  N={n:,}")
    print(f"   Naive treatment effect (confounded): "
          f"{df[df['treated']==1]['converted'].mean() - df[df['treated']==0]['converted'].mean():+.3f}")
    print(f"   True ATE: {true_ate:+.3f}")
    return df


if __name__ == "__main__":
    generate_experiment_data(n=10000, true_effect=0.025)
    generate_observational_data(n=8000, true_ate=0.030)

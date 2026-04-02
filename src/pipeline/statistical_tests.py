"""
statistical_tests.py
--------------------
All hypothesis testing methods for A/B experiment analysis.
Covers: t-test, z-test for proportions, Mann-Whitney U,
chi-squared, and bootstrap confidence intervals.

Usage:
    from src.pipeline.statistical_tests import run_hypothesis_tests
    results = run_hypothesis_tests(df, metric="converted", group_col="group")
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─── Individual Tests ─────────────────────────────────────────────────────────

def two_sample_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,  # Welch's t-test by default
) -> Dict:
    """Two-sample Welch's t-test for continuous metrics."""
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=equal_var)
    df_deg   = len(control) + len(treatment) - 2
    effect   = treatment.mean() - control.mean()
    pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
    cohens_d = effect / pooled_std if pooled_std > 0 else 0

    # 95% CI on the difference
    se = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
    z  = stats.norm.ppf(1 - alpha/2)
    ci = (effect - z*se, effect + z*se)

    return {
        "test":           "Two-sample t-test (Welch)",
        "control_mean":   round(control.mean(), 4),
        "treatment_mean": round(treatment.mean(), 4),
        "absolute_lift":  round(effect, 4),
        "relative_lift":  round(effect / control.mean() * 100, 2) if control.mean() != 0 else None,
        "t_statistic":    round(t_stat, 4),
        "p_value":        round(p_val, 4),
        "significant":    p_val < alpha,
        "cohens_d":       round(cohens_d, 4),
        "ci_lower":       round(ci[0], 4),
        "ci_upper":       round(ci[1], 4),
        "n_control":      len(control),
        "n_treatment":    len(treatment),
    }


def z_test_proportions(
    n_control: int,
    n_treatment: int,
    conv_control: int,
    conv_treatment: int,
    alpha: float = 0.05,
) -> Dict:
    """Two-proportion z-test for binary metrics (conversion rates)."""
    p_c = conv_control / n_control
    p_t = conv_treatment / n_treatment
    p_pool = (conv_control + conv_treatment) / (n_control + n_treatment)

    se_pool = np.sqrt(p_pool * (1-p_pool) * (1/n_control + 1/n_treatment))
    z_stat  = (p_t - p_c) / se_pool if se_pool > 0 else 0
    p_val   = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # CI on the difference
    se_diff = np.sqrt(p_c*(1-p_c)/n_control + p_t*(1-p_t)/n_treatment)
    z_crit  = stats.norm.ppf(1 - alpha/2)
    diff    = p_t - p_c
    ci      = (diff - z_crit*se_diff, diff + z_crit*se_diff)

    return {
        "test":                  "Two-proportion z-test",
        "control_rate":          round(p_c, 4),
        "treatment_rate":        round(p_t, 4),
        "absolute_lift":         round(diff, 4),
        "relative_lift_pct":     round(diff / p_c * 100, 2) if p_c != 0 else None,
        "z_statistic":           round(z_stat, 4),
        "p_value":               round(p_val, 4),
        "significant":           p_val < alpha,
        "ci_lower":              round(ci[0], 4),
        "ci_upper":              round(ci[1], 4),
        "n_control":             n_control,
        "n_treatment":           n_treatment,
        "conversions_control":   conv_control,
        "conversions_treatment": conv_treatment,
    }


def mann_whitney_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Mann-Whitney U test — non-parametric alternative to t-test."""
    u_stat, p_val = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    n1, n2  = len(treatment), len(control)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "test":           "Mann-Whitney U",
        "u_statistic":    round(u_stat, 2),
        "p_value":        round(p_val, 4),
        "significant":    p_val < alpha,
        "effect_size_r":  round(rank_biserial, 4),
        "n_control":      n2,
        "n_treatment":    n1,
    }


def bootstrap_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    stat_fn=np.mean,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence interval for any statistic."""
    np.random.seed(seed)
    diffs = []
    for _ in range(n_bootstrap):
        bc = np.random.choice(control,   size=len(control),   replace=True)
        bt = np.random.choice(treatment, size=len(treatment), replace=True)
        diffs.append(stat_fn(bt) - stat_fn(bc))

    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100 * alpha/2)
    hi = np.percentile(diffs, 100 * (1 - alpha/2))
    p_val = np.mean(diffs <= 0) * 2  # two-tailed

    return {
        "test":          "Bootstrap CI",
        "observed_diff": round(stat_fn(treatment) - stat_fn(control), 4),
        "ci_lower":      round(lo, 4),
        "ci_upper":      round(hi, 4),
        "p_value":       round(min(p_val, 1.0), 4),
        "significant":   (lo > 0) or (hi < 0),
        "n_bootstrap":   n_bootstrap,
    }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_hypothesis_tests(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "group",
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05,
    guardrail_metrics: Optional[list] = None,
) -> Dict:
    """
    Run the full suite of hypothesis tests for a primary metric,
    plus guardrail metric checks.

    Returns a dict with all test results + metadata.
    """
    ctrl = df[df[group_col] == control_label][metric].values
    trt  = df[df[group_col] == treatment_label][metric].values

    results = {
        "experiment_summary": {
            "n_control":   len(ctrl),
            "n_treatment": len(trt),
            "primary_metric": metric,
            "alpha":       alpha,
        }
    }

    # Choose test based on metric type
    is_binary = set(df[metric].unique()).issubset({0, 1})

    if is_binary:
        results["primary_test"] = z_test_proportions(
            n_control=len(ctrl), n_treatment=len(trt),
            conv_control=ctrl.sum(), conv_treatment=trt.sum(),
            alpha=alpha,
        )
    else:
        results["primary_test"]    = two_sample_ttest(ctrl, trt, alpha=alpha)
        results["nonparametric"]   = mann_whitney_test(ctrl, trt, alpha=alpha)

    results["bootstrap_ci"] = bootstrap_ci(ctrl, trt, alpha=alpha)

    # Guardrail metrics
    if guardrail_metrics:
        results["guardrails"] = {}
        for gm in guardrail_metrics:
            g_ctrl = df[df[group_col] == control_label][gm].values
            g_trt  = df[df[group_col] == treatment_label][gm].values
            is_bin = set(df[gm].unique()).issubset({0, 1})
            if is_bin:
                r = z_test_proportions(len(g_ctrl), len(g_trt),
                                       g_ctrl.sum(), g_trt.sum(), alpha)
            else:
                r = two_sample_ttest(g_ctrl, g_trt, alpha)
            results["guardrails"][gm] = r

    # Segment breakdown (if segment cols available)
    results["segment_breakdown"] = {}
    for seg_col in ["device", "user_type", "region"]:
        if seg_col in df.columns:
            seg_results = {}
            for seg_val in df[seg_col].unique():
                seg_df = df[df[seg_col] == seg_val]
                sc = seg_df[seg_df[group_col] == control_label][metric].values
                st = seg_df[seg_df[group_col] == treatment_label][metric].values
                if len(sc) < 30 or len(st) < 30:
                    continue
                if is_binary:
                    r = z_test_proportions(len(sc), len(st), sc.sum(), st.sum(), alpha)
                else:
                    r = two_sample_ttest(sc, st, alpha)
                seg_results[seg_val] = {
                    "lift": r.get("absolute_lift") or r.get("absolute_lift"),
                    "p_value": r["p_value"],
                    "significant": r["significant"],
                }
            results["segment_breakdown"][seg_col] = seg_results

    return results


def print_results_summary(results: Dict) -> None:
    """Print a clean human-readable summary of test results."""
    pt   = results["primary_test"]
    summ = results["experiment_summary"]

    print("\n" + "=" * 60)
    print("  A/B TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Primary Metric  : {summ['primary_metric']}")
    print(f"  N Control       : {summ['n_control']:,}")
    print(f"  N Treatment     : {summ['n_treatment']:,}")
    print(f"  Test Used       : {pt['test']}")
    print("-" * 60)

    if "control_rate" in pt:
        print(f"  Control Rate    : {pt['control_rate']:.2%}")
        print(f"  Treatment Rate  : {pt['treatment_rate']:.2%}")
        print(f"  Absolute Lift   : {pt['absolute_lift']:+.2%}")
        print(f"  Relative Lift   : {pt['relative_lift_pct']:+.1f}%")
    else:
        print(f"  Control Mean    : {pt['control_mean']:.4f}")
        print(f"  Treatment Mean  : {pt['treatment_mean']:.4f}")
        print(f"  Absolute Lift   : {pt['absolute_lift']:+.4f}")

    ci = results["bootstrap_ci"]
    print(f"  95% CI          : [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
    print(f"  p-value         : {pt['p_value']:.4f}")
    sig = pt["significant"]
    print(f"  Significant     : {'✅ YES' if sig else '❌ NO'}")
    print("=" * 60)

    rec = "🟢 SHIP — statistically significant positive lift" if sig and pt["absolute_lift"] > 0 \
        else "🔴 NO-GO — not significant or negative effect"
    print(f"\n  RECOMMENDATION: {rec}")
    print("=" * 60)


if __name__ == "__main__":
    from src.pipeline.data_generator import generate_experiment_data
    df = generate_experiment_data(n=10000, true_effect=0.025)
    results = run_hypothesis_tests(
        df, metric="converted",
        guardrail_metrics=["sessions", "revenue"]
    )
    print_results_summary(results)

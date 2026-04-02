"""
experiment_design.py
--------------------
Power analysis, sample size calculation, randomization,
and pre-experiment validation (A/A test, SRM detection).

Usage:
    from src.pipeline.experiment_design import ExperimentDesigner
    designer = ExperimentDesigner(baseline_rate=0.12, mde=0.02)
    print(designer.sample_size_report())
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple
import hashlib


class ExperimentDesigner:
    """
    Handles all pre-experiment decisions:
    - Sample size calculation via power analysis
    - Traffic allocation recommendations
    - Minimum runtime estimation
    - Unit randomization
    """

    def __init__(
        self,
        baseline_rate: float,
        mde: float,                  # Minimum Detectable Effect (absolute)
        alpha: float = 0.05,         # Type I error rate
        power: float = 0.80,         # 1 - Type II error rate
        two_tailed: bool = True,
        metric_type: str = "proportion",  # "proportion" or "continuous"
        baseline_std: Optional[float] = None,  # required for continuous metrics
    ):
        self.baseline_rate = baseline_rate
        self.mde           = mde
        self.alpha         = alpha
        self.power         = power
        self.two_tailed    = two_tailed
        self.metric_type   = metric_type
        self.baseline_std  = baseline_std

    def _z_alpha(self) -> float:
        tail = self.alpha / 2 if self.two_tailed else self.alpha
        return stats.norm.ppf(1 - tail)

    def _z_beta(self) -> float:
        return stats.norm.ppf(self.power)

    def sample_size_per_group(self) -> int:
        """
        Calculate required sample size per group using the standard
        power analysis formula.
        """
        z_a = self._z_alpha()
        z_b = self._z_beta()

        if self.metric_type == "proportion":
            p1 = self.baseline_rate
            p2 = self.baseline_rate + self.mde
            pooled = (p1 + p2) / 2
            numerator   = (z_a * np.sqrt(2 * pooled * (1 - pooled)) +
                           z_b * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2
            denominator = (p2 - p1) ** 2
        else:
            # Continuous metric
            if self.baseline_std is None:
                raise ValueError("baseline_std required for continuous metrics")
            numerator   = 2 * (self.baseline_std ** 2) * (z_a + z_b) ** 2
            denominator = self.mde ** 2

        return int(np.ceil(numerator / denominator))

    def total_sample_size(self, n_variants: int = 2) -> int:
        return self.sample_size_per_group() * n_variants

    def runtime_days(self, daily_traffic: int, traffic_pct: float = 1.0) -> float:
        """Estimate days needed to collect sufficient sample."""
        eligible_daily = daily_traffic * traffic_pct
        return self.total_sample_size() / eligible_daily

    def effect_size(self) -> float:
        """Cohen's h (proportions) or Cohen's d (continuous)."""
        if self.metric_type == "proportion":
            p2 = self.baseline_rate + self.mde
            return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(self.baseline_rate))
        return self.mde / self.baseline_std

    def sample_size_report(self, daily_traffic: int = 10000) -> str:
        n   = self.sample_size_per_group()
        tot = self.total_sample_size()
        rt  = self.runtime_days(daily_traffic)
        es  = self.effect_size()

        lines = [
            "=" * 55,
            "  EXPERIMENT DESIGN — POWER ANALYSIS REPORT",
            "=" * 55,
            f"  Metric Type          : {self.metric_type}",
            f"  Baseline Rate        : {self.baseline_rate:.1%}",
            f"  Min Detectable Effect: {self.mde:+.1%}",
            f"  Target Rate (control+MDE): {self.baseline_rate + self.mde:.1%}",
            f"  Significance Level α : {self.alpha}",
            f"  Statistical Power    : {self.power:.0%}",
            f"  Tails                : {'Two-tailed' if self.two_tailed else 'One-tailed'}",
            "-" * 55,
            f"  Effect Size          : {es:.3f}",
            f"  Sample Size/Group    : {n:,}",
            f"  Total Sample Needed  : {tot:,}",
            f"  Est. Runtime         : {rt:.1f} days (@ {daily_traffic:,}/day traffic)",
            "=" * 55,
        ]
        return "\n".join(lines)

    @staticmethod
    def randomize(
        user_ids: pd.Series,
        experiment_name: str,
        control_pct: float = 0.50,
        salt: str = "",
    ) -> pd.Series:
        """
        Deterministic, hash-based randomization.
        Same user always gets same assignment for same experiment.
        """
        def _assign(uid):
            key   = f"{experiment_name}{salt}{uid}"
            hash_ = int(hashlib.md5(key.encode()).hexdigest(), 16)
            bucket = (hash_ % 10000) / 10000.0
            return "control" if bucket < control_pct else "treatment"

        return user_ids.apply(_assign)

    @staticmethod
    def check_srm(
        n_control: int,
        n_treatment: int,
        expected_split: float = 0.50,
        alpha: float = 0.01,
    ) -> dict:
        """
        Sample Ratio Mismatch (SRM) detection via chi-squared test.
        Returns dict with p-value and verdict.
        """
        total    = n_control + n_treatment
        expected = [total * expected_split, total * (1 - expected_split)]
        observed = [n_control, n_treatment]
        chi2, pval = stats.chisquare(observed, f_exp=expected)
        srm_detected = pval < alpha
        return {
            "chi2":         round(chi2, 4),
            "p_value":      round(pval, 4),
            "srm_detected": srm_detected,
            "verdict":      "⚠️  SRM DETECTED — investigate randomization" if srm_detected
                            else "✅ No SRM — randomization looks healthy",
            "n_control":    n_control,
            "n_treatment":  n_treatment,
            "actual_split": round(n_control / total, 3),
        }


if __name__ == "__main__":
    # Example: Homepage CTA button test (conversion rate)
    designer = ExperimentDesigner(
        baseline_rate=0.12,
        mde=0.02,        # Detect a 2pp lift (12% → 14%)
        alpha=0.05,
        power=0.80,
        metric_type="proportion",
    )
    print(designer.sample_size_report(daily_traffic=8000))

    # SRM check
    srm = ExperimentDesigner.check_srm(n_control=4820, n_treatment=4980)
    print(f"\nSRM Check: {srm['verdict']}")
    print(f"  p-value={srm['p_value']}  actual split={srm['actual_split']:.1%}")

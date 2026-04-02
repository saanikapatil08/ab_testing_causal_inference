"""
tests/test_framework.py
-----------------------
Unit tests for the A/B Testing & Causal Inference Framework.

Run: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.experiment_design   import ExperimentDesigner
from src.pipeline.data_generator      import generate_experiment_data, generate_observational_data
from src.pipeline.statistical_tests   import (
    two_sample_ttest, z_test_proportions, mann_whitney_test,
    bootstrap_ci, run_hypothesis_tests,
)
from src.pipeline.propensity_matching import PropensityMatcher


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def experiment_df():
    return generate_experiment_data(
        n=4000, true_effect=0.025, baseline_rate=0.12,
        output_dir="/tmp/ab_test_raw", seed=42
    )

@pytest.fixture(scope="module")
def observational_df():
    return generate_observational_data(n=3000, true_ate=0.030,
                                       output_dir="/tmp/ab_obs_raw", seed=42)

@pytest.fixture(scope="module")
def ctrl_arr():
    np.random.seed(0)
    return np.random.binomial(1, 0.12, 2000).astype(float)

@pytest.fixture(scope="module")
def trt_arr():
    np.random.seed(1)
    return np.random.binomial(1, 0.145, 2000).astype(float)


# ─── ExperimentDesigner Tests ─────────────────────────────────────────────────

class TestExperimentDesigner:

    def test_sample_size_positive(self):
        d = ExperimentDesigner(baseline_rate=0.12, mde=0.02)
        assert d.sample_size_per_group() > 0

    def test_larger_mde_needs_less_sample(self):
        d_small = ExperimentDesigner(baseline_rate=0.12, mde=0.01)
        d_large = ExperimentDesigner(baseline_rate=0.12, mde=0.04)
        assert d_small.sample_size_per_group() > d_large.sample_size_per_group()

    def test_higher_power_needs_more_sample(self):
        d_80 = ExperimentDesigner(baseline_rate=0.12, mde=0.02, power=0.80)
        d_90 = ExperimentDesigner(baseline_rate=0.12, mde=0.02, power=0.90)
        assert d_90.sample_size_per_group() > d_80.sample_size_per_group()

    def test_total_sample_is_double_per_group(self):
        d = ExperimentDesigner(baseline_rate=0.12, mde=0.02)
        assert d.total_sample_size() == d.sample_size_per_group() * 2

    def test_runtime_decreases_with_more_traffic(self):
        d = ExperimentDesigner(baseline_rate=0.12, mde=0.02)
        assert d.runtime_days(5000) > d.runtime_days(20000)

    def test_randomization_deterministic(self):
        ids = pd.Series([f"u_{i}" for i in range(100)])
        a1 = ExperimentDesigner.randomize(ids, "exp_xyz")
        a2 = ExperimentDesigner.randomize(ids, "exp_xyz")
        assert (a1 == a2).all()

    def test_randomization_approx_50_50(self):
        ids = pd.Series([f"u_{i}" for i in range(10000)])
        assignments = ExperimentDesigner.randomize(ids, "test_exp")
        control_pct = (assignments == "control").mean()
        assert 0.48 < control_pct < 0.52

    def test_srm_detection_catches_imbalance(self):
        result = ExperimentDesigner.check_srm(n_control=3000, n_treatment=7000)
        assert result["srm_detected"] is True

    def test_srm_passes_balanced(self):
        result = ExperimentDesigner.check_srm(n_control=5010, n_treatment=4990)
        assert result["srm_detected"] is False

    def test_sample_size_report_contains_key_info(self):
        d = ExperimentDesigner(baseline_rate=0.12, mde=0.02)
        report = d.sample_size_report()
        assert "MAPE" not in report  # sanity — not the forecasting report
        assert "Sample Size" in report
        assert "Power" in report


# ─── Data Generator Tests ─────────────────────────────────────────────────────

class TestDataGenerator:

    def test_output_columns(self, experiment_df):
        expected = {"user_id", "group", "converted", "revenue", "sessions"}
        assert expected.issubset(set(experiment_df.columns))

    def test_two_groups_only(self, experiment_df):
        assert set(experiment_df["group"].unique()) == {"control", "treatment"}

    def test_no_negative_revenue(self, experiment_df):
        assert (experiment_df["revenue"] >= 0).all()

    def test_converted_is_binary(self, experiment_df):
        assert set(experiment_df["converted"].unique()).issubset({0, 1})

    def test_treatment_rate_higher(self, experiment_df):
        ctrl_rate = experiment_df[experiment_df["group"] == "control"]["converted"].mean()
        trt_rate  = experiment_df[experiment_df["group"] == "treatment"]["converted"].mean()
        assert trt_rate > ctrl_rate  # true effect = +0.025

    def test_observational_has_confounder(self, observational_df):
        # Power users should be over-represented in treatment
        power_treat = (observational_df[observational_df["treated"] == 1]["user_type"] == "power").mean()
        power_ctrl  = (observational_df[observational_df["treated"] == 0]["user_type"] == "power").mean()
        assert power_treat > power_ctrl


# ─── Statistical Tests ────────────────────────────────────────────────────────

class TestStatisticalTests:

    def test_ttest_detects_large_effect(self):
        np.random.seed(0)
        ctrl = np.random.normal(100, 10, 500)
        trt  = np.random.normal(115, 10, 500)
        result = two_sample_ttest(ctrl, trt)
        assert result["significant"] is True
        assert result["absolute_lift"] > 0

    def test_ttest_no_effect(self):
        np.random.seed(0)
        ctrl = np.random.normal(100, 10, 500)
        trt  = np.random.normal(100.1, 10, 500)
        result = two_sample_ttest(ctrl, trt)
        assert result["significant"] is False

    def test_ztest_proportions_significant(self, ctrl_arr, trt_arr):
        result = z_test_proportions(
            n_control=len(ctrl_arr), n_treatment=len(trt_arr),
            conv_control=ctrl_arr.sum(), conv_treatment=trt_arr.sum(),
        )
        assert result["significant"] is True
        assert result["absolute_lift"] > 0
        assert 0 < result["p_value"] < 0.05

    def test_ztest_ci_contains_true_effect(self, ctrl_arr, trt_arr):
        true_effect = 0.145 - 0.12  # = 0.025
        result = z_test_proportions(
            len(ctrl_arr), len(trt_arr),
            int(ctrl_arr.sum()), int(trt_arr.sum()),
        )
        assert result["ci_lower"] < true_effect < result["ci_upper"]

    def test_mann_whitney_returns_keys(self, ctrl_arr, trt_arr):
        result = mann_whitney_test(ctrl_arr, trt_arr)
        assert "u_statistic" in result
        assert "p_value" in result
        assert "effect_size_r" in result

    def test_bootstrap_ci_excludes_zero_for_large_effect(self):
        np.random.seed(42)
        ctrl = np.random.binomial(1, 0.10, 2000).astype(float)
        trt  = np.random.binomial(1, 0.18, 2000).astype(float)
        result = bootstrap_ci(ctrl, trt, n_bootstrap=2000)
        assert result["ci_lower"] > 0  # large effect → CI should exclude 0

    def test_run_hypothesis_tests_structure(self, experiment_df):
        result = run_hypothesis_tests(
            experiment_df, metric="converted",
            guardrail_metrics=["sessions"],
        )
        assert "primary_test" in result
        assert "bootstrap_ci" in result
        assert "guardrails" in result
        assert "segment_breakdown" in result

    def test_guardrail_flagged_correctly(self):
        # Simulate a case where sessions degrades significantly
        np.random.seed(0)
        df = pd.DataFrame({
            "group":     ["control"] * 1000 + ["treatment"] * 1000,
            "converted": np.random.binomial(1, 0.14, 2000),
            "sessions":  np.concatenate([
                np.random.poisson(3.5, 1000),
                np.random.poisson(1.0, 1000),  # dramatic drop
            ]),
        })
        result = run_hypothesis_tests(df, "converted", guardrail_metrics=["sessions"])
        guard = result["guardrails"]["sessions"]
        assert guard["significant"] is True
        assert guard["absolute_lift"] < 0  # sessions dropped


# ─── Propensity Score Matching Tests ─────────────────────────────────────────

class TestPropensityMatcher:

    def test_ps_scores_between_0_and_1(self, observational_df):
        pm = PropensityMatcher(observational_df, treatment_col="treated",
                               outcome_col="converted", caliper=0.10)
        ps = pm.estimate_propensity_scores()
        assert (ps >= 0).all() and (ps <= 1).all()

    def test_matching_returns_dataframe(self, observational_df):
        pm = PropensityMatcher(observational_df, treatment_col="treated",
                               outcome_col="converted", caliper=0.10)
        matched = pm.fit_and_match()
        assert isinstance(matched, pd.DataFrame)
        assert len(matched) > 0

    def test_matching_reduces_confounder_imbalance(self, observational_df):
        pm = PropensityMatcher(observational_df, treatment_col="treated",
                               outcome_col="converted", caliper=0.10)
        pm.fit_and_match()

        # Check that numeric covariates are available for SMD
        num_cols = [c for c in pm.covariate_cols if c in observational_df.columns]
        if not num_cols:
            pytest.skip("No numeric covariates available for SMD check")

        before_smd = pm.standardized_mean_diff(pm.df, cols=num_cols[:3])["SMD"].mean()
        after_smd  = pm.standardized_mean_diff(pm.matched_df_, cols=num_cols[:3])["SMD"].mean()
        assert after_smd <= before_smd  # balance should improve

    def test_att_estimation_returns_dict(self, observational_df):
        pm = PropensityMatcher(observational_df, treatment_col="treated",
                               outcome_col="converted", caliper=0.10)
        pm.fit_and_match()
        att = pm.estimate_att()
        assert "att" in att
        assert "p_value" in att
        assert "ci_95" in att

    def test_att_closer_to_true_effect_than_naive(self, observational_df):
        true_ate = 0.030
        naive = (
            observational_df[observational_df["treated"] == 1]["converted"].mean()
            - observational_df[observational_df["treated"] == 0]["converted"].mean()
        )
        pm = PropensityMatcher(observational_df, treatment_col="treated",
                               outcome_col="converted", caliper=0.10)
        pm.fit_and_match()
        att = pm.estimate_att()["att"]
        assert abs(att - true_ate) <= abs(naive - true_ate)

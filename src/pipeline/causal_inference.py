"""
causal_inference.py
-------------------
Full causal inference pipeline:
  - Randomized A/B: statistical tests + segment breakdown
  - Observational: propensity score matching + ATT estimation

CLI Usage:
    python src/pipeline/causal_inference.py --experiment homepage_cta --n 10000
    python src/pipeline/causal_inference.py --observational --n 8000
"""

import argparse
import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.data_generator     import generate_experiment_data, generate_observational_data
from src.pipeline.experiment_design  import ExperimentDesigner
from src.pipeline.statistical_tests  import run_hypothesis_tests, print_results_summary
from src.pipeline.propensity_matching import PropensityMatcher
from src.reporting.report_generator  import generate_report
from src.reporting.visualizations    import plot_all_ab


EXPERIMENT_CATALOG = {
    "homepage_cta":       {"baseline": 0.12, "true_effect": 0.025, "mde": 0.02},
    "checkout_flow":      {"baseline": 0.35, "true_effect": 0.041, "mde": 0.03},
    "email_subject":      {"baseline": 0.22, "true_effect": 0.008, "mde": 0.02},
    "pricing_layout":     {"baseline": 0.08, "true_effect": 0.018, "mde": 0.015},
    "onboarding_steps":   {"baseline": 0.60, "true_effect": 0.011, "mde": 0.02},
}


def run_ab_experiment(
    experiment_name: str,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """End-to-end randomized A/B test pipeline."""

    config = EXPERIMENT_CATALOG.get(experiment_name, {"baseline": 0.12, "true_effect": 0.025, "mde": 0.02})

    print(f"\n{'━'*60}")
    print(f"  EXPERIMENT: {experiment_name.replace('_',' ').title()}")
    print(f"{'━'*60}")

    # Step 1: Design
    print("\n[1/5] Experiment Design & Power Analysis")
    designer = ExperimentDesigner(
        baseline_rate=config["baseline"],
        mde=config["mde"],
        alpha=alpha, power=power,
        metric_type="proportion",
    )
    print(designer.sample_size_report(daily_traffic=8000))
    required_n = designer.total_sample_size()

    # Step 2: Data
    print(f"\n[2/5] Generating experiment data (N={n:,})")
    df = generate_experiment_data(
        n=n,
        true_effect=config["true_effect"],
        baseline_rate=config["baseline"],
        experiment_name=experiment_name,
    )

    # Step 3: SRM check
    print("\n[3/5] Randomization Integrity (SRM Check)")
    n_ctrl = (df["group"] == "control").sum()
    n_trt  = (df["group"] == "treatment").sum()
    srm = ExperimentDesigner.check_srm(n_ctrl, n_trt)
    print(f"  {srm['verdict']}")

    # Step 4: Statistical tests
    print("\n[4/5] Hypothesis Testing")
    results = run_hypothesis_tests(
        df, metric="converted",
        guardrail_metrics=["sessions", "revenue"],
        alpha=alpha,
    )
    results["experiment_name"] = experiment_name
    results["config"]          = config
    results["srm_check"]       = srm
    results["required_n"]      = required_n
    results["actual_n"]        = n
    results["powered"]         = n >= required_n

    print_results_summary(results)

    # Step 5: Save + report
    print("\n[5/5] Generating outputs")
    os.makedirs("outputs/results", exist_ok=True)
    result_path = f"outputs/results/{experiment_name}_results.json"
    with open(result_path, "w") as f:
        # Serialize — convert non-serializable types
        safe = {}
        for k, v in results.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
        json.dump(safe, f, indent=2)
    print(f"  ✅ Results saved → {result_path}")

    plot_all_ab(df, results, experiment_name)
    generate_report(results, experiment_name)

    return results


def run_observational(n: int) -> dict:
    """Observational study pipeline with PSM."""
    print(f"\n{'━'*60}")
    print(f"  OBSERVATIONAL STUDY — Propensity Score Matching")
    print(f"{'━'*60}")

    df = generate_observational_data(n=n, true_ate=0.030)

    pm = PropensityMatcher(
        df, treatment_col="treated", outcome_col="converted", caliper=0.05
    )
    matched = pm.fit_and_match()
    pm.balance_report()
    att = pm.estimate_att()

    return att


def main():
    parser = argparse.ArgumentParser(description="A/B Testing & Causal Inference Framework")
    parser.add_argument("--experiment", default="homepage_cta",
                        choices=list(EXPERIMENT_CATALOG.keys()) + ["custom"],
                        help="Experiment to run")
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--power", type=float, default=0.80, help="Statistical power")
    parser.add_argument("--observational", action="store_true",
                        help="Run observational PSM study instead of A/B")
    parser.add_argument("--all", action="store_true", help="Run all experiments in catalog")
    args = parser.parse_args()

    start = time.time()

    if args.observational:
        run_observational(args.n)
    elif args.all:
        for exp in EXPERIMENT_CATALOG:
            run_ab_experiment(exp, args.n, args.alpha, args.power)
    else:
        run_ab_experiment(args.experiment, args.n, args.alpha, args.power)

    print(f"\n✅ Done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

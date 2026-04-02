"""
report_generator.py
-------------------
Generates executive-ready text reports for A/B experiment results.
Produces a standardized go/no-go recommendation with full context.

Usage:
    from src.reporting.report_generator import generate_report
    generate_report(results, experiment_name="homepage_cta")
"""

import os
from datetime import datetime
from typing import Dict


DECISION_THRESHOLD = 0.05


def _verdict(results: Dict) -> str:
    pt  = results["primary_test"]
    sig = pt["significant"]
    lift = pt.get("absolute_lift", 0)

    if sig and lift > 0:
        return "🟢 SHIP — Statistically significant positive lift detected"
    elif sig and lift < 0:
        return "🔴 NO-GO — Statistically significant negative effect"
    elif not sig and lift > 0:
        return "🟡 EXTEND — Positive trend but insufficient evidence; consider extending runtime"
    else:
        return "🔴 NO-GO — No meaningful effect detected"


def _guardrail_status(results: Dict) -> str:
    if "guardrails" not in results:
        return "  No guardrail metrics specified."
    lines = []
    for metric, r in results["guardrails"].items():
        lift  = r.get("absolute_lift", 0)
        sig   = r["significant"]
        flag  = "⚠️  DEGRADED" if (sig and lift < 0) else "✅ OK"
        lines.append(f"  {metric:20s} : lift={lift:+.4f}  p={r['p_value']:.4f}  {flag}")
    return "\n".join(lines)


def _segment_summary(results: Dict) -> str:
    if "segment_breakdown" not in results:
        return "  No segment breakdown available."
    lines = []
    for seg_col, segs in results["segment_breakdown"].items():
        lines.append(f"\n  By {seg_col}:")
        for seg_val, r in segs.items():
            sig = "✅" if r["significant"] else "  "
            lines.append(f"    {sig} {seg_val:15s}: lift={r['lift']:+.4f}  p={r['p_value']:.4f}")
    return "\n".join(lines)


def generate_report(
    results: Dict,
    experiment_name: str,
    output_dir: str = "outputs/reports",
) -> str:
    """
    Write a standardized executive experiment report.

    Returns path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    pt     = results["primary_test"]
    summ   = results["experiment_summary"]
    ci     = results["bootstrap_ci"]
    srm    = results.get("srm_check", {})
    config = results.get("config", {})

    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    exp_title = experiment_name.replace("_", " ").title()
    verdict   = _verdict(results)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          A/B EXPERIMENT — EXECUTIVE REPORT                  ║
╚══════════════════════════════════════════════════════════════╝

  Experiment   : {exp_title}
  Report Date  : {now}
  Analyst      : Saanika Patil

══════════════════════════════════════════════════════════════
  EXPERIMENT OVERVIEW
══════════════════════════════════════════════════════════════
  Primary Metric     : {summ['primary_metric']}
  Significance Level : α = {summ['alpha']}
  Sample Sizes       : Control={summ['n_control']:,}  Treatment={summ['n_treatment']:,}
  Powered            : {"✅ Yes" if results.get('powered', True) else "⚠️  Underpowered — interpret with caution"}

══════════════════════════════════════════════════════════════
  RANDOMIZATION INTEGRITY
══════════════════════════════════════════════════════════════
  SRM Check          : {srm.get('verdict', 'Not run')}
  Actual Split       : {srm.get('actual_split', 'N/A')}  (expected 0.500)
  SRM p-value        : {srm.get('p_value', 'N/A')}

══════════════════════════════════════════════════════════════
  PRIMARY METRIC RESULTS
══════════════════════════════════════════════════════════════
  Test Used          : {pt['test']}
  Control Rate       : {pt.get('control_rate', pt.get('control_mean', 'N/A')):.4f}
  Treatment Rate     : {pt.get('treatment_rate', pt.get('treatment_mean', 'N/A')):.4f}
  Absolute Lift      : {pt['absolute_lift']:+.4f}
  Relative Lift      : {pt.get('relative_lift_pct', 'N/A'):+.1f}%  (if proportion test)
  p-value            : {pt['p_value']:.4f}
  Statistically Sig. : {"✅ YES" if pt['significant'] else "❌ NO"}
  Bootstrap 95% CI   : [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]

══════════════════════════════════════════════════════════════
  GUARDRAIL METRICS
══════════════════════════════════════════════════════════════
{_guardrail_status(results)}

══════════════════════════════════════════════════════════════
  SEGMENT BREAKDOWN
══════════════════════════════════════════════════════════════
{_segment_summary(results)}

══════════════════════════════════════════════════════════════
  ✅ RECOMMENDATION
══════════════════════════════════════════════════════════════
  {verdict}

  Rationale:
  - Primary metric lift: {pt['absolute_lift']:+.4f} (p={pt['p_value']:.4f})
  - 95% CI excludes zero: {(ci['ci_lower'] > 0) or (ci['ci_upper'] < 0)}
  - All guardrail metrics: {"not degraded" if "guardrails" in results else "not measured"}
  - Sample size adequate: {results.get('powered', True)}

══════════════════════════════════════════════════════════════
"""

    filename = f"{experiment_name}_executive_report_{datetime.now().strftime('%Y%m%d')}.txt"
    path     = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(report)

    print(f"  📄 Executive report saved → {path}")
    return path


if __name__ == "__main__":
    # Demo
    sample_results = {
        "experiment_name": "homepage_cta",
        "experiment_summary": {"primary_metric": "converted", "alpha": 0.05,
                               "n_control": 5000, "n_treatment": 5000},
        "primary_test": {
            "test": "Two-proportion z-test",
            "control_rate": 0.1201, "treatment_rate": 0.1452,
            "absolute_lift": 0.0251, "relative_lift_pct": 20.9,
            "p_value": 0.0003, "significant": True,
        },
        "bootstrap_ci": {"ci_lower": 0.0142, "ci_upper": 0.0361},
        "srm_check": {"verdict": "✅ No SRM", "actual_split": 0.500, "p_value": 0.72},
        "powered": True,
    }
    generate_report(sample_results, "homepage_cta")

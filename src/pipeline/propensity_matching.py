"""
propensity_matching.py
----------------------
Propensity Score Matching (PSM) for causal inference
in observational (non-randomized) studies.

Steps:
  1. Estimate propensity scores via logistic regression
  2. Match treatment/control units (nearest-neighbor within caliper)
  3. Assess covariate balance (SMD)
  4. Estimate ATE and ATT on matched sample

Usage:
    from src.pipeline.propensity_matching import PropensityMatcher
    pm = PropensityMatcher(df, treatment_col="treated", outcome_col="converted")
    matched_df = pm.fit_and_match()
    pm.balance_report()
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial import KDTree
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


class PropensityMatcher:
    """
    Propensity Score Matching with covariate balance diagnostics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treated",
        outcome_col: str = "converted",
        covariate_cols: Optional[List[str]] = None,
        caliper: float = 0.05,         # Max PS distance for a valid match
        ratio: int = 1,                # 1:1 matching
        with_replacement: bool = False,
    ):
        self.df              = df.copy()
        self.treatment_col   = treatment_col
        self.outcome_col     = outcome_col
        self.caliper         = caliper
        self.ratio           = ratio
        self.with_replacement = with_replacement
        self.matched_df_     = None
        self.ps_model_       = None
        self.covariate_cols  = covariate_cols or self._auto_detect_covariates()

    def _auto_detect_covariates(self) -> List[str]:
        """Auto-detect numeric/encoded covariate columns."""
        exclude = {self.treatment_col, self.outcome_col, "user_id"}
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in exclude]

    def _encode_categoricals(self) -> pd.DataFrame:
        """One-hot encode categorical covariates for logistic regression."""
        df = self.df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in {"user_id", self.treatment_col, self.outcome_col}]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            # Update covariate cols to include dummies
            new_dummies = [c for c in df.columns if any(c.startswith(cat) for cat in cat_cols)]
            self.covariate_cols = self.covariate_cols + [c for c in new_dummies if c not in self.covariate_cols]
        return df

    def estimate_propensity_scores(self) -> pd.Series:
        """Fit logistic regression and return propensity scores."""
        df_enc = self._encode_categoricals()
        available_covs = [c for c in self.covariate_cols if c in df_enc.columns]

        X = df_enc[available_covs].fillna(0).values
        y = df_enc[self.treatment_col].values

        self.ps_model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self.ps_model_.fit(X, y)
        ps = self.ps_model_.predict_proba(X)[:, 1]
        self.df["propensity_score"] = ps
        print(f"✅ Propensity scores estimated | "
              f"min={ps.min():.3f}  max={ps.max():.3f}  mean={ps.mean():.3f}")
        return pd.Series(ps, index=self.df.index)

    def match(self) -> pd.DataFrame:
        """
        Nearest-neighbor matching within caliper.
        Returns matched dataset (treatment + matched controls).
        """
        if "propensity_score" not in self.df.columns:
            self.estimate_propensity_scores()

        treated  = self.df[self.df[self.treatment_col] == 1].copy()
        controls = self.df[self.df[self.treatment_col] == 0].copy()

        control_ps  = controls["propensity_score"].values.reshape(-1, 1)
        treated_ps  = treated["propensity_score"].values.reshape(-1, 1)
        tree        = KDTree(control_ps)

        used_controls = set()
        matched_pairs = []
        n_unmatched   = 0

        for i, (idx, row) in enumerate(treated.iterrows()):
            query_ps = [[row["propensity_score"]]]
            k        = min(self.ratio * 10, len(controls))  # search wider, filter by caliper
            dists, positions = tree.query(query_ps, k=k)

            matched_for_this = 0
            for dist, pos in zip(dists[0], positions[0]):
                if dist > self.caliper:
                    break
                ctrl_idx = controls.index[pos]
                if not self.with_replacement and ctrl_idx in used_controls:
                    continue
                used_controls.add(ctrl_idx)
                matched_pairs.append({
                    "treatment_idx": idx,
                    "control_idx":   ctrl_idx,
                    "ps_distance":   dist,
                })
                matched_for_this += 1
                if matched_for_this >= self.ratio:
                    break

            if matched_for_this == 0:
                n_unmatched += 1

        print(f"✅ Matching complete:")
        print(f"   Matched pairs    : {len(matched_pairs):,}")
        print(f"   Unmatched treated: {n_unmatched:,} ({n_unmatched/len(treated)*100:.1f}%)")

        if not matched_pairs:
            raise ValueError("No matches found. Try increasing the caliper.")

        t_idxs = [p["treatment_idx"] for p in matched_pairs]
        c_idxs = [p["control_idx"]   for p in matched_pairs]
        matched_df = pd.concat([
            self.df.loc[t_idxs],
            self.df.loc[c_idxs],
        ]).reset_index(drop=True)

        self.matched_df_ = matched_df
        return matched_df

    def fit_and_match(self) -> pd.DataFrame:
        self.estimate_propensity_scores()
        return self.match()

    def standardized_mean_diff(
        self,
        df: Optional[pd.DataFrame] = None,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute Standardized Mean Difference (SMD) for covariate balance.
        SMD < 0.1 is generally considered well-balanced.
        """
        df   = self.df if df is None else df
        cols = cols or [c for c in self.covariate_cols if c in df.columns][:8]

        rows = []
        for col in cols:
            t_vals = df[df[self.treatment_col] == 1][col].dropna()
            c_vals = df[df[self.treatment_col] == 0][col].dropna()
            pooled_std = np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2)
            smd = (t_vals.mean() - c_vals.mean()) / pooled_std if pooled_std > 0 else 0
            rows.append({
                "covariate":     col,
                "mean_treated":  round(t_vals.mean(), 4),
                "mean_control":  round(c_vals.mean(), 4),
                "SMD":           round(abs(smd), 4),
                "balanced":      abs(smd) < 0.1,
            })
        return pd.DataFrame(rows).set_index("covariate")

    def balance_report(self) -> None:
        """Print before/after balance comparison."""
        print("\n" + "=" * 60)
        print("  COVARIATE BALANCE REPORT")
        print("=" * 60)

        print("\n--- BEFORE Matching ---")
        before = self.standardized_mean_diff(self.df)
        print(before.to_string())

        if self.matched_df_ is not None:
            print("\n--- AFTER Matching ---")
            after = self.standardized_mean_diff(self.matched_df_)
            print(after.to_string())
            pct_balanced = after["balanced"].mean() * 100
            print(f"\n✅ {pct_balanced:.0f}% of covariates balanced (SMD < 0.1) after matching")

    def estimate_att(self) -> Dict:
        """
        Estimate Average Treatment Effect on the Treated (ATT)
        on the matched sample.
        """
        if self.matched_df_ is None:
            raise ValueError("Run fit_and_match() first")

        mdf = self.matched_df_
        t_outcomes = mdf[mdf[self.treatment_col] == 1][self.outcome_col].values
        c_outcomes = mdf[mdf[self.treatment_col] == 0][self.outcome_col].values

        att   = t_outcomes.mean() - c_outcomes.mean()
        se    = np.sqrt(t_outcomes.var()/len(t_outcomes) + c_outcomes.var()/len(c_outcomes))
        z     = att / se if se > 0 else 0
        from scipy import stats as _stats
        p_val = 2 * (1 - _stats.norm.cdf(abs(z)))
        ci    = (att - 1.96*se, att + 1.96*se)

        result = {
            "estimand":         "ATT (Average Treatment Effect on Treated)",
            "att":              round(att, 4),
            "se":               round(se, 4),
            "z_statistic":      round(z, 4),
            "p_value":          round(p_val, 4),
            "ci_95":            (round(ci[0], 4), round(ci[1], 4)),
            "significant":      p_val < 0.05,
            "n_matched_treated": len(t_outcomes),
            "n_matched_control": len(c_outcomes),
        }

        print("\n" + "=" * 55)
        print("  CAUSAL EFFECT ESTIMATE (PSM)")
        print("=" * 55)
        print(f"  Estimand    : {result['estimand']}")
        print(f"  ATT         : {att:+.4f}  ({att*100:+.2f} percentage points)")
        print(f"  95% CI      : [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"  p-value     : {p_val:.4f}")
        print(f"  Significant : {'✅ YES' if result['significant'] else '❌ NO'}")
        print("=" * 55)
        return result


if __name__ == "__main__":
    from src.pipeline.data_generator import generate_observational_data
    df = generate_observational_data(n=8000, true_ate=0.030)

    pm = PropensityMatcher(
        df,
        treatment_col="treated",
        outcome_col="converted",
        caliper=0.05,
    )
    matched = pm.fit_and_match()
    pm.balance_report()
    pm.estimate_att()

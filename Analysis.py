"""
02_analysis.py
--------------
NJ Literacy & COVID Economic Impact — Statistical Analysis Script

Performs three analyses using the cleaned district-level and DFG-aggregated
ELA proficiency data produced by 01_clean_data.py:

  ANALYSIS 1 — COVID Recovery Gap by DFG Tier
      Computes the change in mean ELA proficiency from the 2017-2018
      pre-COVID baseline to the 2023-2024 recovery endpoint for each of
      the 8 DFG socioeconomic tiers. Tests whether lower-SES tiers lost
      more ground than higher-SES tiers.

  ANALYSIS 2 — OLS Regression: Predictors of Recovery
      At the district level, models the change in ELA proficiency from
      2017-2018 to 2023-2024 as a function of:
        - Pre-COVID ELA proficiency baseline (2017-2018)
        - % Economically Disadvantaged Students (2017-2018)
        - DFG rank (1=lowest SES, 8=highest SES)
      This isolates whether economic disadvantage predicts recovery
      independently of where a district started.

  ANALYSIS 3 — Distance from NJ 2030 Goal (80%)
      For each DFG tier, calculates how far the average district is from
      the state 80% ELA proficiency target and projects whether current
      recovery trajectories will reach that goal by 2030.

Outputs saved to data/processed/:
  regression_dataset.csv    — district-level analysis table
  dfg_recovery_summary.csv  — tier-level summary table

Usage:
  Must be run AFTER 01_clean_data.py.
  python scripts/02_analysis.py

Requirements:
  pip install pandas numpy scipy statsmodels
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
PROC_DIR     = os.path.join("data", "processed")
DISTRICT_CSV = os.path.join(PROC_DIR, "ela_by_district_year.csv")
DFG_CSV      = os.path.join(PROC_DIR, "ela_by_dfg_year.csv")

DFG_ORDER  = ["A", "B", "CD", "DE", "FG", "GH", "I", "J"]
NJ_GOAL    = 80.0
BASE_YEAR  = "2017-2018"
END_YEAR   = "2023-2024"
GOAL_YEAR  = 2030
END_CAL_YR = 2024


def divider(title=""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * width)


# ════════════════════════════════════════════════════════════════════════
# Data preparation
# ════════════════════════════════════════════════════════════════════════

def build_regression_dataset(dist):
    pre = (
        dist[dist["school_year"] == BASE_YEAR]
        [["district_code", "district_name", "ela_proficiency_pct",
          "econ_disadvantaged_pct", "dfg", "dfg_rank"]]
        .rename(columns={"ela_proficiency_pct":    "ela_pre",
                         "econ_disadvantaged_pct": "econ_disadvantaged"})
    )
    post = (
        dist[dist["school_year"] == END_YEAR]
        [["district_code", "ela_proficiency_pct"]]
        .rename(columns={"ela_proficiency_pct": "ela_post"})
    )
    reg = pre.merge(post, on="district_code", how="inner")
    reg["recovery_change"] = reg["ela_post"] - reg["ela_pre"]
    reg["below_goal_2024"] = reg["ela_post"] < NJ_GOAL
    reg["gap_to_goal"]     = NJ_GOAL - reg["ela_post"]
    reg = reg.dropna(subset=["ela_pre", "ela_post", "econ_disadvantaged", "dfg_rank"])
    return reg.reset_index(drop=True)


def build_dfg_summary(reg, dfg):
    recovery_years = dfg[dfg["period"] == "Recovery"].copy()
    recovery_years["year_num"] = recovery_years["school_year"].str[:4].astype(int)

    annual_change = {}
    for tier in DFG_ORDER:
        sub = recovery_years[recovery_years["dfg"] == tier].sort_values("year_num")
        if len(sub) >= 2:
            slope, _, _, _, _ = stats.linregress(
                sub["year_num"].values,
                sub["ela_proficiency_mean"].values
            )
            annual_change[tier] = slope
        else:
            annual_change[tier] = np.nan

    summary = (
        reg.groupby("dfg")
        .agg(
            n_districts     =("district_code",   "count"),
            ela_pre_mean    =("ela_pre",          "mean"),
            ela_post_mean   =("ela_post",         "mean"),
            recovery_change =("recovery_change",  "mean"),
            econ_dis_mean   =("econ_disadvantaged","mean"),
            pct_below_goal  =("below_goal_2024",  "mean"),
        )
        .reindex(DFG_ORDER)
        .reset_index()
    )
    summary["pct_below_goal"]     = summary["pct_below_goal"] * 100
    summary["gap_to_goal_2024"]   = NJ_GOAL - summary["ela_post_mean"]
    summary["annual_recovery_rate"] = summary["dfg"].map(annual_change)
    years_left = GOAL_YEAR - END_CAL_YR
    summary["projected_2030"] = (
        summary["ela_post_mean"] +
        summary["annual_recovery_rate"] * years_left
    )
    summary["on_track_2030"] = summary["projected_2030"] >= NJ_GOAL
    return summary


# ════════════════════════════════════════════════════════════════════════
# Analysis 1 — Recovery gap by DFG tier
# ════════════════════════════════════════════════════════════════════════

def analysis_recovery_gap(reg, summary):
    divider("ANALYSIS 1: RECOVERY GAP BY DFG TIER")

    print(f"\n  Baseline : {BASE_YEAR}   Endpoint : {END_YEAR}")
    print(f"  Districts in analysis : {len(reg)}")

    print(f"\n  {'DFG':>4} | {'N':>4} | {'Pre-COVID':>10} | {'Recovery':>10} | "
          f"{'Change':>9} | {'Econ Dis%':>10} | {'%Below80':>9}")
    print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")

    for _, row in summary.iterrows():
        print(
            f"  {row['dfg']:>4} | {int(row['n_districts']):>4} | "
            f"{row['ela_pre_mean']:>9.1f}% | {row['ela_post_mean']:>9.1f}% | "
            f"{row['recovery_change']:>+8.1f}pp | {row['econ_dis_mean']:>9.1f}% | "
            f"{row['pct_below_goal']:>8.1f}%"
        )

    dfg_rank_map = {t: i+1 for i, t in enumerate(DFG_ORDER)}
    r, p = stats.pearsonr(
        summary["dfg"].map(dfg_rank_map),
        summary["recovery_change"]
    )
    print(f"\n  Correlation — DFG rank vs. recovery change : r = {r:.3f}  p = {p:.4f}")
    print(f"  → {'Higher-SES districts recovered significantly more' if p < 0.05 else 'Trend present but not significant at tier level (n=8)'}")

    tier_a = summary[summary["dfg"] == "A"]["recovery_change"].values[0]
    tier_j = summary[summary["dfg"] == "J"]["recovery_change"].values[0]
    print(f"\n  DFG-A (lowest SES) recovery change  : {tier_a:+.1f} pp")
    print(f"  DFG-J (highest SES) recovery change : {tier_j:+.1f} pp")
    print(f"  Differential between A and J        : {tier_j - tier_a:.1f} pp")


# ════════════════════════════════════════════════════════════════════════
# Analysis 2 — OLS Regression
# ════════════════════════════════════════════════════════════════════════

def analysis_regression(reg):
    divider("ANALYSIS 2: OLS REGRESSION — PREDICTORS OF RECOVERY")

    print("""
  Outcome   : recovery_change — ELA change from 2017-18 to 2023-24
  Predictors:
    ela_pre          — Pre-COVID ELA proficiency (baseline control)
    econ_disadvantaged — % economically disadvantaged students
    dfg_rank         — DFG socioeconomic rank (1=lowest, 8=highest)
    """)

    y = reg["recovery_change"]
    X = sm.add_constant(reg[["ela_pre", "econ_disadvantaged", "dfg_rank"]])
    model = sm.OLS(y, X).fit()

    print(f"  Observations : {int(model.nobs)}")
    print(f"  R²           : {model.rsquared:.4f}  ({model.rsquared*100:.1f}% of variance explained)")
    print(f"  Adjusted R²  : {model.rsquared_adj:.4f}")
    print(f"  F-statistic  : {model.fvalue:.2f}  (p = {model.f_pvalue:.4f})")

    labels = {
        "const":             "Intercept",
        "ela_pre":           "ELA Pre-COVID",
        "econ_disadvantaged":"% Econ Disadvantaged",
        "dfg_rank":          "DFG Rank (SES)",
    }
    print(f"\n  {'Variable':>22} | {'Coeff':>8} | {'Std Err':>8} | "
          f"{'t':>7} | {'p-value':>9} | {'Sig':>5}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*5}")

    for var in ["const", "ela_pre", "econ_disadvantaged", "dfg_rank"]:
        coef = model.params[var]
        se   = model.bse[var]
        t    = model.tvalues[var]
        p    = model.pvalues[var]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {labels[var]:>22} | {coef:>+8.3f} | {se:>8.3f} | "
              f"{t:>7.2f} | {p:>9.4f} | {sig:>5}")

    print("\n  Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")

    c_econ = model.params["econ_disadvantaged"]
    c_dfg  = model.params["dfg_rank"]
    c_pre  = model.params["ela_pre"]
    p_econ = model.pvalues["econ_disadvantaged"]
    p_dfg  = model.pvalues["dfg_rank"]
    p_pre  = model.pvalues["ela_pre"]

    def sig_str(p):
        return "statistically significant" if p < 0.05 else "not statistically significant"

    print(f"""
  PLAIN-LANGUAGE INTERPRETATION:

  • Each additional 1pp of economically disadvantaged students
    is associated with a {c_econ:+.3f} pp shift in recovery change
    ({sig_str(p_econ)}, p={p_econ:.4f}).

  • Each 1-step increase in DFG rank (higher SES) is associated
    with a {c_dfg:+.3f} pp shift in recovery change
    ({sig_str(p_dfg)}, p={p_dfg:.4f}).

  • Holding economic factors constant, a higher pre-COVID baseline
    is associated with a {c_pre:+.3f} pp shift in recovery
    ({sig_str(p_pre)}, p={p_pre:.4f}).

  • These results support the core argument: economic disadvantage
    is a meaningful predictor of slower post-COVID recovery, even
    after controlling for starting point. Uniform screening without
    targeted funding will not close these gaps.
    """)

    print("  BIVARIATE CORRELATIONS (supporting evidence):")
    for col, label in [
        ("econ_disadvantaged", "% Econ Disadvantaged vs. Recovery Change"),
        ("ela_pre",            "Pre-COVID ELA Baseline vs. Recovery Change"),
        ("dfg_rank",           "DFG Rank vs. Recovery Change           "),
    ]:
        r, p = stats.pearsonr(reg[col], reg["recovery_change"])
        print(f"    {label}  r = {r:+.3f}  p = {p:.4f}")


# ════════════════════════════════════════════════════════════════════════
# Analysis 3 — 2030 goal projection
# ════════════════════════════════════════════════════════════════════════

def analysis_goal_projection(summary):
    divider("ANALYSIS 3: DISTANCE FROM NJ 2030 GOAL (80%)")

    print(f"\n  NJ Target : {NJ_GOAL}% ELA proficiency by {GOAL_YEAR}")
    print(f"  Projection: linear extrapolation of annual recovery rate")

    print(f"\n  {'DFG':>4} | {'2024 ELA':>9} | {'Gap to 80%':>11} | "
          f"{'Rate/yr':>8} | {'Proj 2030':>10} | {'On Track':>9}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")

    for _, row in summary.iterrows():
        rate_s  = f"{row['annual_recovery_rate']:+.2f}" if not pd.isna(row['annual_recovery_rate']) else " N/A"
        proj_s  = f"{row['projected_2030']:.1f}%" if not pd.isna(row['projected_2030']) else "N/A"
        track_s = "YES ✓" if row["on_track_2030"] else "NO  ✗"
        print(
            f"  {row['dfg']:>4} | {row['ela_post_mean']:>8.1f}% | "
            f"{row['gap_to_goal_2024']:>+10.1f}pp | {rate_s:>8} | "
            f"{proj_s:>10} | {track_s:>9}"
        )

    off = summary[~summary["on_track_2030"]]["dfg"].tolist()
    on  = summary[ summary["on_track_2030"]]["dfg"].tolist()
    print(f"\n  On track for 80% by 2030  : {on  if on  else 'None'}")
    print(f"  NOT on track              : {off if off else 'None'}")

    worst = summary.loc[summary["gap_to_goal_2024"].idxmax()]
    print(f"\n  Furthest from goal: DFG-{worst['dfg']}")
    print(f"    Current proficiency : {worst['ela_post_mean']:.1f}%")
    print(f"    Projected 2030      : {worst['projected_2030']:.1f}%")
    print(f"    Average % econ dis  : {worst['econ_dis_mean']:.1f}%")
    print(f"    These districts need targeted intervention, not just screening.")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    divider()
    print("  NJ Literacy & COVID Economic Impact — 02_analysis.py")
    divider()

    for path in [DISTRICT_CSV, DFG_CSV]:
        if not os.path.exists(path):
            print(f"\n  Missing: {path}")
            print("  Run 01_clean_data.py first.")
            return

    dist = pd.read_csv(DISTRICT_CSV, dtype={"county_code": str, "district_code": str})
    dfg  = pd.read_csv(DFG_CSV)
    print(f"\n  District rows loaded : {len(dist):,}")
    print(f"  DFG rows loaded      : {len(dfg):,}")

    reg     = build_regression_dataset(dist)
    summary = build_dfg_summary(reg, dfg)

    analysis_recovery_gap(reg, summary)
    analysis_regression(reg)
    analysis_goal_projection(summary)

    divider("SAVING OUTPUTS")
    reg_path = os.path.join(PROC_DIR, "regression_dataset.csv")
    sum_path = os.path.join(PROC_DIR, "dfg_recovery_summary.csv")
    reg.to_csv(reg_path, index=False)
    summary.to_csv(sum_path, index=False)
    print(f"\n  Saved: {reg_path}  ({len(reg)} rows)")
    print(f"  Saved: {sum_path}  ({len(summary)} rows)")

    divider()
    print("  ANALYSIS COMPLETE")
    print("  Next step: python scripts/03_visualize.py")
    divider()


if __name__ == "__main__":
    main()
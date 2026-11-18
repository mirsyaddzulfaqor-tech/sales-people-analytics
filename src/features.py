"""Feature engineering: turn the monthly panel into a model-ready table.

The split between HR and sales features is deliberate. This project's thesis
is that each is weak alone and strong combined, so the feature lists have to
be usable independently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 3   # predict resignation within the next 3 months


def _roll(s: pd.Series, w: int, fn: str = "mean") -> pd.Series:
    return getattr(s.rolling(w, min_periods=1), fn)()


def build_features(employees: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.merge(
        employees[["rep_id", "region", "level", "employment_status", "commute_km",
                   "manager_span", "territory_potential", "territory_id", "city",
                   "exit_date", "join_date"]],
        on="rep_id", how="left")
    df["month"] = pd.to_datetime(df["month"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values(["rep_id", "month"]).reset_index(drop=True)

    g = df.groupby("rep_id", sort=False)

    # --- Sales signals ---
    df["attain_3m"] = g["attainment"].transform(lambda s: _roll(s, 3))
    df["attain_6m"] = g["attainment"].transform(lambda s: _roll(s, 6))
    df["attain_trend"] = df["attain_3m"] - g["attainment"].transform(
        lambda s: _roll(s.shift(3), 3))
    df["quota_fairness_3m"] = g["quota_fairness"].transform(lambda s: _roll(s, 3))
    df["incentive_ratio"] = df["incentive_idr"] / df["base_salary_idr"]
    df["incentive_ratio_3m"] = g["incentive_ratio"].transform(lambda s: _roll(s, 3))
    df["months_no_incentive_3m"] = g["incentive_idr"].transform(
        lambda s: _roll((s == 0).astype(float), 3, "sum"))
    df["coverage_3m"] = g["outlet_coverage"].transform(lambda s: _roll(s, 3))
    df["strike_rate_3m"] = g["strike_rate"].transform(lambda s: _roll(s, 3))

    # --- HR signals (shape of an HRIS export) ---
    df["pct_late"] = df["days_late"] / df["working_days"]
    df["pct_absent"] = df["days_absent"] / df["working_days"]
    df["pct_late_3m"] = g["pct_late"].transform(lambda s: _roll(s, 3))
    df["pct_absent_3m"] = g["pct_absent"].transform(lambda s: _roll(s, 3))
    df["late_trend"] = df["pct_late_3m"] - g["pct_late"].transform(
        lambda s: _roll(s.shift(3), 3))
    df["overtime_3m"] = g["overtime_hours"].transform(lambda s: _roll(s, 3))
    df["leave_taken_3m"] = g["leave_taken"].transform(lambda s: _roll(s, 3, "sum"))

    # --- Label: resigns within HORIZON months ---
    months_to_exit = (
        (df["exit_date"].dt.year - df["month"].dt.year) * 12
        + (df["exit_date"].dt.month - df["month"].dt.month))
    df["months_to_exit"] = months_to_exit
    df["label_attrition"] = ((months_to_exit >= 1) & (months_to_exit <= HORIZON)).astype(int)

    # Right censoring: the final HORIZON months are dropped from the labelled
    # set, because nobody's future is observed there yet. Dropping only the
    # still-active reps would leave those months filled almost entirely with
    # leavers and bias the evaluation. The raw rows are kept separately so the
    # app can still score the most recent month.
    cutoff = df["month"].max() - pd.DateOffset(months=HORIZON)
    labelled = df[df["month"] <= cutoff].copy()
    labelled.attrs["label_cutoff"] = cutoff
    labelled.attrs["scoring_rows"] = df[df["month"] > cutoff].copy()
    return labelled.replace([np.inf, -np.inf], np.nan)


HR_FEATURES = [
    "pct_late_3m", "pct_absent_3m", "late_trend", "overtime_3m", "leave_taken_3m",
    "leave_balance", "tenure_months", "commute_km", "manager_span",
    "status_Contract", "status_Probation",
    "level_Sales Rep (Senior)", "level_Team Leader",
]

SALES_FEATURES = [
    "attain_3m", "attain_6m", "attain_trend", "months_below_quota",
    "quota_fairness_3m", "incentive_ratio_3m", "months_no_incentive_3m",
    "coverage_3m", "strike_rate_3m", "lines_per_call",
]

ALL_FEATURES = HR_FEATURES + SALES_FEATURES

FRIENDLY_LABEL = {
    "pct_late_3m": "Lateness rate (3 mo)",
    "pct_absent_3m": "Unexcused absence (3 mo)",
    "late_trend": "Lateness getting worse",
    "overtime_3m": "Overtime hours (3 mo)",
    "leave_taken_3m": "Leave taken (3 mo)",
    "leave_balance": "Leave balance",
    "tenure_months": "Tenure (months)",
    "commute_km": "Commute to territory",
    "manager_span": "Reports per supervisor",
    "status_Contract": "Contract status",
    "status_Probation": "Probation status",
    "level_Sales Rep (Senior)": "Level: Senior rep",
    "level_Team Leader": "Level: Team leader",
    "attain_3m": "Quota attainment (3 mo)",
    "attain_6m": "Quota attainment (6 mo)",
    "attain_trend": "Attainment trend",
    "months_below_quota": "Consecutive months below quota",
    "quota_fairness_3m": "Quota unfairness",
    "incentive_ratio_3m": "Incentive vs base salary",
    "months_no_incentive_3m": "Months with zero incentive (of 3)",
    "coverage_3m": "Outlet coverage (3 mo)",
    "strike_rate_3m": "Strike rate (3 mo)",
    "lines_per_call": "Lines per call",
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Simple one-hot encoding for the categorical columns the model uses."""
    out = df.copy()
    for s in ["Contract", "Probation"]:
        out[f"status_{s}"] = (out["employment_status"] == s).astype(int)
    for lv in ["Sales Rep (Senior)", "Team Leader"]:
        out[f"level_{lv}"] = (out["level"] == lv).astype(int)
    out.attrs.update(df.attrs)
    return out

"""Translate attrition risk into money.

This is the part that separates people analytics from a machine-learning
exercise. Managers do not buy AUC; they buy an answer to "what does it cost
if this person leaves, and is keeping them worth it".

Every assumption below is exposed and adjustable in the app - see
ASSUMPTIONS.md for the source behind each reference figure.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Assumptions:
    # Recruitment + onboarding cost as a multiple of annual OTE.
    # Industry reference is 1.0-2.0x OTE; the lower bound is used to stay
    # conservative.
    recruitment_multiple: float = 1.0
    # Months the territory sits vacant before a replacement starts.
    # Reference time-to-fill is 45-70 days.
    vacancy_months: float = 2.0
    # Months until a replacement reaches full productivity.
    ramp_months: int = 6
    # Contribution margin, used to convert lost sales into lost profit rather
    # than counting gross revenue as a cost.
    contribution_margin: float = 0.12
    # Share of at-risk reps an intervention actually retains.
    intervention_success: float = 0.35
    # Cost of intervening per person (coaching, quota rebalance, retention).
    intervention_cost_idr: float = 3_500_000


def _ramp_gap(ramp_months: int) -> float:
    """Total productive months lost while a replacement ramps up.

    Uses the same ramp curve as the data generator: 30% in month zero,
    full productivity by month six.
    """
    lost = 0.0
    for t in range(ramp_months):
        ramp = min(1.0, 0.30 + 0.70 * (t / 6.0) ** 0.75) if t > 0 else 0.30
        lost += (1.0 - ramp)
    return lost


def cost_per_rep(row: pd.Series, a: Assumptions) -> dict[str, float]:
    """Cost of one resignation, broken into components that can be argued with."""
    monthly_salary = float(row.get("base_salary_idr", 0.0))
    monthly_incentive = float(row.get("incentive_idr", 0.0))
    annual_ote = (monthly_salary + monthly_incentive) * 12

    monthly_sales = float(row.get("actual_sales_idr", 0.0))
    monthly_profit = monthly_sales * a.contribution_margin

    recruitment = annual_ote * a.recruitment_multiple
    vacancy = monthly_profit * a.vacancy_months
    ramp = monthly_profit * _ramp_gap(a.ramp_months)

    return {
        "Recruitment & onboarding": recruitment,
        "Vacant territory": vacancy,
        "Replacement ramp-up": ramp,
        "Total": recruitment + vacancy + ramp,
    }


def cost_table(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    detail = df.apply(lambda r: pd.Series(cost_per_rep(r, a)), axis=1)
    return pd.concat([df.reset_index(drop=True), detail.reset_index(drop=True)], axis=1)


def simulate_intervention(risk_df: pd.DataFrame, n_target: int,
                          a: Assumptions) -> dict[str, float]:
    """If we intervene on the N highest-risk reps, what is saved?

    Deliberately does NOT assume everyone treated stays. The expected value is
    risk x intervention success x cost, which leaves a genuinely negative net
    result possible - and it does turn negative under harsher assumptions.
    """
    if risk_df.empty or n_target <= 0:
        return dict(n=0, programme_cost=0.0, loss_avoided=0.0,
                    net=0.0, roi=float("nan"))

    target = risk_df.nlargest(min(n_target, len(risk_df)), "risk")
    expected_leavers = float(target["risk"].sum())
    mean_cost = float(target["Total"].mean()) if "Total" in target else 0.0

    loss_avoided = expected_leavers * a.intervention_success * mean_cost
    programme_cost = len(target) * a.intervention_cost_idr
    net = loss_avoided - programme_cost
    roi = (net / programme_cost) if programme_cost > 0 else float("nan")
    return dict(n=len(target), programme_cost=programme_cost,
                loss_avoided=loss_avoided, net=net, roi=roi)

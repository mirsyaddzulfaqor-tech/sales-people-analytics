"""Synthetic data generator: field sales force + HRIS records.

IMPORTANT: every row here is COMPUTER GENERATED. Nothing originates from a
real company, distributor or employee. Column shapes deliberately mirror an
HRIS export (attendance, leave, payroll, employment status) joined to a CRM /
quota export, so the analysis reflects real people-analytics work.

Latent structure the model is meant to recover:
  - HR signals alone    -> weak
  - Sales signals alone -> weak
  - The two combined    -> strong
That is the thesis of this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261118
N_REPS = 420
N_MONTHS = 36
START = pd.Timestamp("2023-12-01")

REGIONS = {
    "Greater Jakarta":   dict(potential=1.25, cities=["Jakarta", "Bekasi", "Tangerang", "Depok"]),
    "West Java":         dict(potential=1.05, cities=["Bandung", "Cirebon", "Sukabumi"]),
    "Central Java":      dict(potential=0.92, cities=["Semarang", "Solo", "Yogyakarta"]),
    "East Java":         dict(potential=1.00, cities=["Surabaya", "Malang", "Jember"]),
    "South Sumatra":     dict(potential=0.78, cities=["Palembang", "Lampung", "Jambi"]),
}
LEVELS = ["Sales Rep (Junior)", "Sales Rep (Senior)", "Team Leader"]
EMPLOYMENT_STATUS = ["Probation", "Contract", "Permanent"]

# Monthly base salary by level, in IDR. Illustrative figures, not real payroll.
BASE_SALARY = {
    "Sales Rep (Junior)": 4_800_000,
    "Sales Rep (Senior)": 6_500_000,
    "Team Leader": 9_200_000,
}

# Baseline monthly selling capacity by level, before ramp and territory potential.
CAPACITY = {
    "Sales Rep (Junior)": 165_000_000,
    "Sales Rep (Senior)": 235_000_000,
    "Team Leader": 300_000_000,
}

# National average territory potential, used by the company when setting quotas.
AVG_POTENTIAL = 1.00
# Stretch factor: quotas are deliberately set slightly above fair capacity.
STRETCH = 1.03


def _ramp(tenure_months: int) -> float:
    """Ramp-up curve: a new rep is not productive on day one.

    Reference: for field sales, roughly 50% productivity around month 3 and
    full productivity around month 6 (see ASSUMPTIONS.md).
    """
    if tenure_months <= 0:
        return 0.30
    return float(min(1.0, 0.30 + 0.70 * (tenure_months / 6.0) ** 0.75))


def _incentive(attainment: float, base_salary: float) -> float:
    """Tiered incentive scheme with a cliff at 100% of quota.

    That cliff later becomes one of the attrition drivers: a rep who keeps
    landing at 90-99% never reaches the large tier.
    """
    if attainment < 0.80:
        pct = 0.0
    elif attainment < 0.90:
        pct = 0.15
    elif attainment < 1.00:
        pct = 0.35
    elif attainment < 1.15:
        pct = 0.85          # sharp jump the moment 100% is reached
    else:
        pct = 1.20
    return float(base_salary * pct)


def build(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.date_range(START, periods=N_MONTHS, freq="MS")

    # ---------- Employee master ----------
    region_names = list(REGIONS)
    reps = []
    for i in range(N_REPS):
        region = region_names[rng.integers(0, len(region_names))]
        meta = REGIONS[region]
        level = str(rng.choice(LEVELS, p=[0.55, 0.33, 0.12]))

        # Some reps were already employed before the data window opens.
        starting_tenure = int(rng.choice([0, 0, 0, 1, 3, 6, 10, 16, 24, 38, 55],
                                         p=[.10, .07, .07, .08, .09, .10, .11, .12, .10, .09, .07]))
        join = START - pd.DateOffset(months=starting_tenure)

        if starting_tenure < 3:
            status = "Probation"
        elif starting_tenure < 14:
            status = str(rng.choice(["Contract", "Permanent"], p=[0.75, 0.25]))
        else:
            status = str(rng.choice(["Contract", "Permanent"], p=[0.25, 0.75]))

        reps.append(dict(
            rep_id=f"EMP-{i+1:04d}",
            region=region,
            city=str(rng.choice(meta["cities"])),
            territory_id=f"TR-{region[:3].upper()}-{rng.integers(1, 40):02d}",
            level=level,
            employment_status=status,
            join_date=join,
            # Territory potential: how much demand actually exists there.
            territory_potential=float(np.clip(rng.normal(meta["potential"], 0.18), 0.60, 1.55)),
            # Individual skill, never visible in the data - this is honest noise.
            skill=float(np.clip(rng.normal(1.0, 0.13), 0.65, 1.40)),
            commute_km=float(np.clip(rng.gamma(2.2, 6.0), 2, 70)),
            base_salary_idr=float(BASE_SALARY[level] * rng.normal(1.0, 0.06)),
            supervisor_id=f"SPV-{rng.integers(1, 23):02d}",
        ))
    reps = pd.DataFrame(reps)

    # Span of control: how many reports each supervisor carries.
    span = reps.groupby("supervisor_id")["rep_id"].count().rename("manager_span")
    reps = reps.merge(span, on="supervisor_id", how="left")

    # ---------- Monthly panel ----------
    active = {r: True for r in reps["rep_id"]}
    exit_date: dict[str, pd.Timestamp | None] = {r: None for r in reps["rep_id"]}
    below_streak = {r: 0 for r in reps["rep_id"]}
    discipline_drift = {r: 0.0 for r in reps["rep_id"]}   # attendance slowly degrading
    rows = []

    rmeta = reps.set_index("rep_id").to_dict("index")

    for m_idx, month in enumerate(months):
        # Seasonality: Ramadan/Eid and year-end lift FMCG sell-out.
        mm = month.month
        seasonal = {3: 1.18, 4: 1.14, 11: 1.08, 12: 1.12}.get(mm, 1.0)
        seasonal *= {1: 0.90, 2: 0.95}.get(mm, 1.0)

        for rep_id, meta in rmeta.items():
            if not active[rep_id]:
                continue

            tenure = max(0, (month.year - meta["join_date"].year) * 12
                         + (month.month - meta["join_date"].month))
            ramp = _ramp(tenure)

            # --- Quota: the company sets targets only PARTLY informed by
            # territory potential; the rest uses a national average. Low
            # potential territories therefore get over-quota'd. This is the
            # source of quota unfairness analysed later. ---
            fair_capacity = CAPACITY[meta["level"]] * meta["territory_potential"]
            potential_weight = 0.45     # 0 = fully blind, 1 = perfectly fair
            basis = CAPACITY[meta["level"]] * (
                potential_weight * meta["territory_potential"]
                + (1 - potential_weight) * AVG_POTENTIAL)
            seasonal_quota = 1 + 0.6 * (seasonal - 1)   # quotas seasonalised, imperfectly
            quota = float(max(60_000_000,
                              basis * STRETCH * rng.normal(1.0, 0.09)
                              * seasonal_quota * (1.003 ** m_idx)))

            # --- Actual sales ---
            actual = (CAPACITY[meta["level"]] * meta["territory_potential"] * ramp
                      * meta["skill"] * seasonal * (1.002 ** m_idx)
                      * rng.normal(1.0, 0.13))
            actual = float(max(0.0, actual))
            attainment = actual / quota

            # Quota fairness ratio: >1 means the quota exceeds what the
            # territory can fairly yield. Core metric of the fairness view.
            quota_fairness = float(quota / (fair_capacity * STRETCH))

            # --- Field activity ---
            outlets_assigned = int(np.clip(rng.normal(185, 35), 90, 320))
            discipline = float(np.clip(1.0 - discipline_drift[rep_id] + rng.normal(0, 0.05), 0.45, 1.12))
            coverage = float(np.clip(rng.normal(0.86, 0.09) * discipline, 0.35, 1.0))
            outlets_visited = int(outlets_assigned * coverage)
            strike_rate = float(np.clip(rng.normal(0.66, 0.10) * (0.85 + 0.30 * ramp), 0.20, 0.95))
            productive_calls = int(outlets_visited * strike_rate)
            lines_per_call = float(np.clip(rng.normal(4.8, 1.0) * (0.9 + 0.2 * meta["skill"]), 1.5, 9))

            # --- HRIS side (shape of an HRIS export) ---
            working_days = int(np.busday_count(month.date(),
                                               (month + pd.offsets.MonthBegin(1)).date()))
            p_late = float(np.clip(0.06 + discipline_drift[rep_id] * 0.55 + rng.normal(0, 0.02), 0.0, 0.75))
            days_late = int(rng.binomial(working_days, p_late))
            p_absent = float(np.clip(discipline_drift[rep_id] * 0.14 + rng.normal(0, 0.006), 0.0, 0.25))
            days_absent = int(rng.binomial(working_days, p_absent))
            leave_taken = int(rng.poisson(0.9 + discipline_drift[rep_id] * 1.3))
            leave_balance = int(np.clip(12 - (m_idx % 12) * 0.9 - leave_taken, 0, 12))
            overtime_hours = float(np.clip(
                rng.gamma(2.0, 4.5) * (1 + 0.5 * max(0, quota_fairness - 1)), 0, 70))

            incentive = _incentive(attainment, meta["base_salary_idr"])

            if attainment < 0.85:
                below_streak[rep_id] += 1
            else:
                below_streak[rep_id] = max(0, below_streak[rep_id] - 1)

            rows.append(dict(
                rep_id=rep_id, month=month, tenure_months=tenure,
                quota_idr=quota, actual_sales_idr=actual, attainment=attainment,
                quota_fairness=quota_fairness,
                outlets_assigned=outlets_assigned, outlets_visited=outlets_visited,
                outlet_coverage=coverage, strike_rate=strike_rate,
                productive_calls=productive_calls, lines_per_call=lines_per_call,
                working_days=working_days, days_late=days_late, days_absent=days_absent,
                leave_taken=leave_taken, leave_balance=leave_balance,
                overtime_hours=overtime_hours,
                base_salary_idr=meta["base_salary_idr"], incentive_idr=incentive,
                months_below_quota=below_streak[rep_id],
            ))

            # --- Discipline degrades slowly when the job feels futile ---
            pressure = (0.35 * max(0.0, quota_fairness - 1.05)
                        + 0.10 * min(4, below_streak[rep_id]) / 4
                        + 0.05 * (meta["commute_km"] > 35))
            discipline_drift[rep_id] = float(np.clip(
                discipline_drift[rep_id] + rng.normal(pressure * 0.11, 0.012) - 0.004, 0.0, 0.85))

            # --- Attrition hazard ---
            # Built from HR signals, sales signals AND an interaction term.
            # That interaction is why the combined model beats either alone.
            s_hr = (2.4 * p_late + 6.0 * p_absent + 0.020 * overtime_hours
                    + 0.45 * (meta["employment_status"] == "Contract")
                    + 0.70 * (meta["employment_status"] == "Probation"))
            s_sales = (0.50 * min(5, below_streak[rep_id])
                       + 1.80 * max(0.0, quota_fairness - 1.05)
                       + 1.0 * (incentive == 0.0))
            # Interaction: discipline degrading AND quota repeatedly missed.
            # Deliberately keyed on the slow-moving drift so it is picked up by
            # 3-month rolling features rather than by a single lucky month.
            interaction = (1.9 * (discipline_drift[rep_id] > 0.14)
                           * (below_streak[rep_id] >= 2))

            z = (-6.45 + 0.72 * s_hr + 0.78 * s_sales + 0.95 * interaction
                 + 0.70 * np.exp(-tenure / 7.0)          # first-year attrition is high
                 + 0.014 * meta["commute_km"]
                 + 0.038 * meta["manager_span"]
                 + rng.normal(0, 0.30))                   # honest noise: not everything is in the data
            if rng.random() < 1 / (1 + np.exp(-z)):
                active[rep_id] = False
                exit_date[rep_id] = month

    monthly = pd.DataFrame(rows)
    reps["exit_date"] = reps["rep_id"].map(exit_date)
    reps["is_active"] = reps["exit_date"].isna()
    return reps, monthly


if __name__ == "__main__":
    import pathlib
    reps, monthly = build()
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    reps.to_csv(out / "employees.csv", index=False)
    monthly.to_csv(out / "monthly_performance.csv", index=False)
    leavers = int((~reps["is_active"]).sum())
    rate = leavers / len(reps)
    print(f"employees    : {len(reps)}")
    print(f"monthly rows : {len(monthly):,}")
    print(f"attrition    : {leavers} ({rate:.1%} over {N_MONTHS} months, "
          f"~{1 - (1 - rate) ** (12 / N_MONTHS):.1%}/year)")
    print(f"attainment   : median {monthly['attainment'].median():.2f}")

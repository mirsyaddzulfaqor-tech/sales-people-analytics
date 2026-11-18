"""Sales Force People Analytics - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme as T
from src.economics import Assumptions, cost_table, simulate_intervention
from src.features import ALL_FEATURES, build_features, encode
from src.generate_data import build as build_data
from src.model import (check_calibration, evaluate_folds, score_current,
                       top_reasons)

st.set_page_config(page_title="Sales Force People Analytics",
                   page_icon="📊", layout="wide")


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner="Generating synthetic data...")
def load():
    employees, monthly = build_data()
    feats = encode(build_features(employees, monthly))
    scoring = encode(feats.attrs["scoring_rows"])
    return employees, monthly, feats, scoring


@st.cache_data(show_spinner="Training and evaluating models...")
def model_results(feats: pd.DataFrame):
    return evaluate_folds(feats), check_calibration(feats)


@st.cache_data(show_spinner="Scoring current risk...")
def risk_scores(feats: pd.DataFrame, scoring: pd.DataFrame):
    return score_current(feats, scoring)


employees, monthly, feats, scoring = load()
folds, calibration = model_results(feats)
scores = risk_scores(feats, scoring)

st.title("Sales Force People Analytics")
st.caption(
    "Joining HRIS records (attendance, leave, employment status, payroll) to "
    "sales performance data to answer one question: **who is going to resign "
    "in the next three months, and what does it cost to let it happen.**")
st.warning(
    "Every row in this app is **computer generated**. Nothing originates from a "
    "real company, distributor or employee.", icon="⚠️")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Attrition Risk", "Cost & Simulation", "Quota, Incentives & Ramp"])


# ------------------------------------------------------------- OVERVIEW
with tab1:
    active = int(employees["is_active"].sum())
    leavers = int((~employees["is_active"]).sum())
    n_months = monthly["month"].nunique()
    annual_turnover = 1 - (1 - leavers / len(employees)) ** (12 / n_months)
    expected = float(scores["risk"].sum())

    costs = cost_table(scores, Assumptions())
    mean_cost = float(costs["Total"].mean())

    c = st.columns(4)
    c[0].metric("Active reps", f"{active}")
    c[1].metric("Annual turnover", f"{annual_turnover:.1%}",
                help="Industry reference for sales forces: 20-35% a year.")
    c[2].metric("Expected leavers (3 mo)", f"{expected:.0f}",
                help="Sum of model probabilities across all active reps.")
    c[3].metric("Mean cost per resignation", T.idr(mean_cost))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Resignations per month")
        ex = (employees.dropna(subset=["exit_date"])
              .assign(m=lambda d: pd.to_datetime(d["exit_date"]).dt.to_period("M").dt.to_timestamp())
              .groupby("m").size().rename("leavers").reset_index())
        fig = go.Figure(go.Bar(
            x=ex["m"], y=ex["leavers"], name="Resignations",
            marker_color=T.S1, marker_line_width=0,
            hovertemplate="%{x|%b %Y}<br>%{y} resignations<extra></extra>"))
        T.apply_layout(fig, 320, showlegend=False, ylab="people")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Quota attainment vs outlet coverage")
        bl = (monthly.assign(month=pd.to_datetime(monthly["month"]))
              .groupby("month")[["attainment", "outlet_coverage"]].mean().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bl["month"], y=bl["attainment"], name="Quota attainment",
            mode="lines", line=dict(color=T.S1, width=2),
            hovertemplate="%{x|%b %Y}<br>Attainment %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=bl["month"], y=bl["outlet_coverage"], name="Outlet coverage",
            mode="lines", line=dict(color=T.S2, width=2),
            hovertemplate="%{x|%b %Y}<br>Coverage %{y:.0%}<extra></extra>"))
        # One axis only: both are ratios, so they are comparable. Never a
        # second y-axis.
        T.apply_layout(fig, 320, ylab="ratio")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Turnover by region")
    by_region = (employees.assign(left=~employees["is_active"])
                 .groupby("region").agg(reps=("rep_id", "size"), left=("left", "sum")))
    by_region["rate"] = by_region["left"] / by_region["reps"]
    by_region = by_region.sort_values("rate").reset_index()
    fig = go.Figure(go.Bar(
        x=by_region["rate"], y=by_region["region"], orientation="h",
        marker_color=T.S1, marker_line_width=0,
        text=[f"{v:.0%}" for v in by_region["rate"]], textposition="outside",
        hovertemplate="%{y}<br>%{x:.1%} resigned<extra></extra>"))
    T.apply_layout(fig, 300, showlegend=False, xlab="share of reps who resigned")
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=T.GRIDLINE)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------- ATTRITION RISK
with tab2:
    st.subheader("Is joining HR and sales data worth anything?")
    st.markdown(
        "The same model is trained three times on different feature sets and "
        "evaluated **rolling-origin** - trained on the past, tested on the "
        "following three months, repeated at four points in time. The figures "
        "below are fold averages, not one flattering split.")

    summary = (folds.groupby("model")[["auc", "pr_auc", "lift"]]
               .agg(["mean", "std"]).round(3))
    summary.columns = ["AUC", "AUC ±", "PR-AUC", "PR-AUC ±", "Top-decile lift", "Lift ±"]
    order = ["HR only", "Sales only", "Combined"]
    st.dataframe(summary.reindex([o for o in order if o in summary.index]),
                 use_container_width=True)

    fig = go.Figure()
    for i, mdl in enumerate(order):
        sub = folds[folds["model"] == mdl]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["fold"], y=sub["auc"], name=mdl,
            marker_color=T.SERIES[i], marker_line_width=0,
            hovertemplate=f"{mdl}<br>%{{x}}<br>AUC %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dot", line_color=T.INK_MUTED,
                  annotation_text="random guessing", annotation_font_size=11)
    T.apply_layout(fig, 340, ylab="AUC")
    fig.update_layout(bargap=0.25, bargroupgap=0.04)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Finding:** HRIS data alone is close to useless here - AUC around "
        "0.58, barely better than guessing. The signal lives in commercial "
        "performance. HR data earns its place only as a **complement**, and "
        "that gain is consistent across all four folds. A people analytics "
        "team looking only at the HRIS would conclude this problem cannot be "
        "predicted.", icon="💡")

    with st.expander("Calibration - can the probabilities be trusted?"):
        st.markdown(
            "The model is used to **sum** expected leaver counts, so ranking "
            "alone is not enough; the numbers have to be close to reality.")
        k = calibration.copy()
        k.columns = ["Fold", "Mean predicted", "Actual", "Brier"]
        st.dataframe(k.round(4), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Watchlist")
    base = float(scores.attrs.get("base_rate", 0.08))
    st.markdown(
        f"Risk tiers are multiples of the baseline resignation rate "
        f"(**{base:.1%}** per three months), not hand-picked round numbers: "
        f"Moderate ≥1×, High ≥2×, Critical ≥3×.")

    watch = scores[scores["risk_tier"].astype(str) != "Low"].copy()
    c = st.columns(4)
    for i, tier in enumerate(["Low", "Moderate", "High", "Critical"]):
        n = int((scores["risk_tier"].astype(str) == tier).sum())
        colour, icon = T.RISK_STYLE[tier]
        c[i].markdown(
            f"<div style='border-left:4px solid {colour};padding:2px 0 2px 10px'>"
            f"<div style='font-size:12px;color:{T.INK_MUTED}'>{icon} {tier}</div>"
            f"<div style='font-size:26px;color:{T.INK_PRIMARY}'>{n}</div></div>",
            unsafe_allow_html=True)

    st.caption(
        f"Of {len(scores)} active reps, {len(watch)} need attention. A short "
        "list is the point - it is the number a manager can actually act on "
        "this month.")

    if watch.empty:
        st.success("No rep above the threshold this month.")
    else:
        view = watch[["rep_id", "region", "level", "employment_status",
                      "tenure_months", "attain_3m", "risk", "risk_tier"]].copy()
        view["risk"] = (view["risk"] * 100).round(1)
        view["attain_3m"] = (view["attain_3m"] * 100).round(0)
        view.columns = ["Rep", "Region", "Level", "Status", "Tenure (mo)",
                        "Attainment 3mo (%)", "Risk (%)", "Tier"]
        st.dataframe(view, use_container_width=True, hide_index=True)

        pick = st.selectbox("Inspect one rep's risk drivers", watch["rep_id"].tolist())
        row = watch[watch["rep_id"] == pick].iloc[0]
        reasons = top_reasons(row, ALL_FEATURES, n=5)
        if reasons:
            names = [r[0] for r in reasons][::-1]
            values = [r[1] for r in reasons][::-1]
            fig = go.Figure(go.Bar(
                x=values, y=names, orientation="h",
                marker_color=T.STATUS["serious"], marker_line_width=0,
                hovertemplate="%{y}<br>contribution %{x:.2f}<extra></extra>"))
            T.apply_layout(fig, 300, showlegend=False,
                           xlab="contribution to risk score (log-odds)")
            fig.update_xaxes(showgrid=True, gridcolor=T.GRIDLINE)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Contributions are the logistic regression coefficient times "
                "the standardised value. This is why the app uses logistic "
                "regression rather than gradient boosting: the reasoning is "
                "legible and can be discussed with a manager, and the AUC gap "
                "is negligible.")
        else:
            st.caption("No factor pushes this rep above average risk.")


# ---------------------------------------------------- COST & SIMULATION
with tab3:
    st.subheader("What does one resignation cost?")
    st.markdown(
        "Every assumption below is adjustable and the charts move with it. "
        "The source behind each reference figure is in `ASSUMPTIONS.md`.")

    s = st.columns(3)
    multiple = s[0].slider("Recruitment cost (× annual OTE)", 0.5, 2.0, 1.0, 0.1,
                           help="Industry reference 1.0-2.0× OTE.")
    vacancy = s[1].slider("Vacant territory (months)", 0.5, 4.0, 2.0, 0.5,
                          help="Reference time-to-fill 45-70 days.")
    margin = s[2].slider("Contribution margin", 0.05, 0.30, 0.12, 0.01,
                         help="Converts lost sales into lost profit.")

    s2 = st.columns(2)
    success = s2[0].slider("Intervention success rate", 0.10, 0.70, 0.35, 0.05,
                           help="Share of at-risk reps actually retained.")
    int_cost = s2[1].slider("Intervention cost per person (IDR million)",
                            1.0, 20.0, 3.5, 0.5)

    a = Assumptions(recruitment_multiple=multiple, vacancy_months=vacancy,
                    contribution_margin=margin, intervention_success=success,
                    intervention_cost_idr=int_cost * 1_000_000)
    costs = cost_table(scores, a)

    components = ["Recruitment & onboarding", "Vacant territory", "Replacement ramp-up"]
    means = [float(costs[c].mean()) for c in components]
    fig = go.Figure()
    for i, (c_name, v) in enumerate(zip(components, means)):
        fig.add_trace(go.Bar(
            x=[v], y=["Cost per resignation"], orientation="h", name=c_name,
            marker_color=T.SERIES[i], marker_line=dict(color=T.SURFACE, width=2),
            hovertemplate=c_name + "<br>Rp%{x:,.0f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    T.apply_layout(fig, 220, xlab="IDR")
    fig.update_xaxes(showgrid=True, gridcolor=T.GRIDLINE)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Mean total cost per resignation", T.idr(sum(means)))

    st.divider()
    st.subheader("If we intervene on the N highest-risk reps")
    n_target = st.slider("Reps to intervene on", 5, min(80, len(costs)), 20, 5)

    rows = []
    for n in range(5, min(80, len(costs)) + 1, 5):
        r = simulate_intervention(costs, n, a)
        rows.append(dict(n=n, net=r["net"], roi=r["roi"]))
    sim = pd.DataFrame(rows)
    chosen = simulate_intervention(costs, n_target, a)

    m = st.columns(3)
    m[0].metric("Loss avoided", T.idr(chosen["loss_avoided"]))
    m[1].metric("Programme cost", T.idr(chosen["programme_cost"]))
    m[2].metric("Net result", T.idr(chosen["net"]),
                delta=f"ROI {chosen['roi']:.1f}×")

    fig = go.Figure(go.Scatter(
        x=sim["n"], y=sim["net"], mode="lines+markers", name="Net result",
        line=dict(color=T.S1, width=2), marker=dict(size=8, color=T.S1),
        hovertemplate="%{x} reps treated<br>net Rp%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color=T.INK_MUTED)
    T.apply_layout(fig, 340, showlegend=False,
                   xlab="reps treated", ylab="net result (IDR)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This simulation does **not** assume everyone treated stays. The "
        "expected value is risk × intervention success × cost, which leaves a "
        "negative net result genuinely possible - and it does turn negative "
        "once intervention cost rises or the success rate falls.")


# ------------------------------------- QUOTA, INCENTIVES & RAMP
with tab4:
    mdf = monthly.copy()
    mdf["month"] = pd.to_datetime(mdf["month"])

    st.subheader("Are quotas set fairly?")
    st.markdown(
        "Quota unfairness = quota divided by the territory's fair capacity. "
        "Above 1 means a rep is targeted beyond what their territory can yield.")

    per_rep = (mdf.groupby("rep_id")
               .agg(fairness=("quota_fairness", "mean"),
                    attainment=("attainment", "mean")).reset_index()
               .merge(employees[["rep_id", "region", "is_active"]], on="rep_id"))
    fig = go.Figure()
    for i, (name, sub) in enumerate([("Still active", per_rep[per_rep["is_active"]]),
                                     ("Resigned", per_rep[~per_rep["is_active"]])]):
        fig.add_trace(go.Scatter(
            x=sub["fairness"], y=sub["attainment"], mode="markers", name=name,
            marker=dict(size=9, color=T.SERIES[i], opacity=0.75,
                        line=dict(color=T.SURFACE, width=1)),
            hovertemplate=f"{name}<br>unfairness %{{x:.2f}}<br>"
                          "attainment %{y:.0%}<extra></extra>"))
    fig.add_vline(x=1.0, line_dash="dot", line_color=T.INK_MUTED,
                  annotation_text="fair quota", annotation_font_size=11)
    T.apply_layout(fig, 400, xlab="quota unfairness ratio",
                   ylab="mean quota attainment")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    bands = pd.cut(per_rep["fairness"], [0, 0.95, 1.05, 1.15, 9],
                   labels=["Loose", "Fair", "Tight", "Unreasonable"])
    band = (per_rep.assign(band=bands).groupby("band", observed=True)
            .agg(reps=("rep_id", "size"),
                 left=("is_active", lambda s: int((~s).sum()))))
    band["rate"] = band["left"] / band["reps"]
    fig = go.Figure(go.Bar(
        x=band.index.astype(str), y=band["rate"],
        marker_color=T.S2, marker_line_width=0,
        text=[f"{v:.0%}" for v in band["rate"]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.1%} resigned<extra></extra>"))
    T.apply_layout(fig, 320, showlegend=False, ylab="share of reps who resigned")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note the relationship is **not** gradual. Loose, fair and tight "
        "quotas are nearly identical; only genuinely unreasonable quotas move "
        "the number. What drives people out is not a demanding target but an "
        "impossible one.")

    st.divider()
    st.subheader("The incentive cliff")
    st.markdown(
        "The incentive scheme jumps sharply the moment attainment touches "
        "100%. Reps who repeatedly land at 90-99% work hard without ever "
        "reaching the large tier - and they are the ones who leave most often.")
    bins = pd.cut(mdf["attainment"], [0, .8, .9, 1.0, 1.15, 9],
                  labels=["<80%", "80-89%", "90-99%", "100-114%", "≥115%"])
    inc = (mdf.assign(band=bins)
           .groupby("band", observed=True)["incentive_idr"].mean().reset_index())
    fig = go.Figure(go.Bar(
        x=inc["band"].astype(str), y=inc["incentive_idr"],
        marker_color=T.S1, marker_line_width=0,
        hovertemplate="%{x}<br>mean incentive Rp%{y:,.0f}<extra></extra>"))
    T.apply_layout(fig, 320, showlegend=False,
                   xlab="attainment band", ylab="mean incentive (IDR)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("New rep ramp-up")
    ramp = (mdf[mdf["tenure_months"] <= 18]
            .groupby("tenure_months")["attainment"].mean().reset_index())
    fig = go.Figure(go.Scatter(
        x=ramp["tenure_months"], y=ramp["attainment"], mode="lines+markers",
        line=dict(color=T.S1, width=2), marker=dict(size=7, color=T.S1),
        hovertemplate="month %{x}<br>attainment %{y:.0%}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dot", line_color=T.INK_MUTED,
                  annotation_text="full quota", annotation_font_size=11)
    T.apply_layout(fig, 340, showlegend=False,
                   xlab="tenure (months)", ylab="mean quota attainment")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This curve is what makes attrition expensive: a replacement needs "
        "months before covering quota, and that gap is counted as cost in the "
        "Cost & Simulation tab.")

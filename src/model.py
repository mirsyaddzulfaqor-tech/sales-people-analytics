"""Attrition risk model: resignation within the next 3 months.

Built three times with different feature sets - HR only, sales only, and
combined - because the question this project answers is not "how accurate is
the model" but "is joining the two systems worth anything".

The split is TEMPORAL (train on earlier months, test on later ones), not
random. A random split leaks the future into the past through rows belonging
to the same person, which makes the score meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, FRIENDLY_LABEL, HR_FEATURES, SALES_FEATURES

SPLIT_MONTH = pd.Timestamp("2026-03-01")

# Origins for rolling-origin evaluation. A single split is far too noisy on
# data this size: the number moves a long way purely on who happened to resign
# inside the test window. Reporting the mean of several folds is more honest
# than picking one flattering number.
FOLDS = [pd.Timestamp(t) for t in
         ["2025-09-01", "2025-12-01", "2026-03-01", "2026-06-01"]]

FEATURE_SETS = {
    "HR only": HR_FEATURES,
    "Sales only": SALES_FEATURES,
    "Combined": ALL_FEATURES,
}


@dataclass
class ModelResult:
    name: str
    features: list[str]
    auc: float
    pr_auc: float
    base_rate: float
    lift: float
    pipeline: Pipeline | None = None
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)


def _logistic() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        # NO class_weight="balanced". Re-weighting buys about 0.005 AUC but
        # destroys calibration outright (mean predicted probability jumps to
        # 0.35 when the real event rate is 0.06). The whole economics module
        # sums probabilities to estimate leaver counts, so calibration is
        # worth more here than AUC.
        ("clf", LogisticRegression(max_iter=2000, C=0.6)),
    ])


def _boosting() -> Pipeline:
    return Pipeline([("clf", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=300,
        min_samples_leaf=40, l2_regularization=1.0, random_state=7))])


def split(df: pd.DataFrame, split_month: pd.Timestamp = SPLIT_MONTH):
    return df[df["month"] < split_month], df[df["month"] >= split_month]


def _top_decile_lift(y: np.ndarray, p: np.ndarray, base: float) -> float:
    """How many times denser are leavers in the top-scoring 10%?

    This is the number an HR manager actually cares about: if I can only call
    twenty people this month, how much better than random is my list?
    """
    n_top = max(1, int(len(p) * 0.10))
    idx = np.argsort(p)[::-1][:n_top]
    return float(y[idx].mean() / base) if base > 0 else float("nan")


def evaluate_folds(df: pd.DataFrame, kind: str = "logistic") -> pd.DataFrame:
    """Rolling origin: train on the past, test on the following 3 months."""
    rows = []
    for origin in FOLDS:
        train = df[df["month"] < origin]
        test = df[(df["month"] >= origin) & (df["month"] < origin + pd.DateOffset(months=3))]
        if test["label_attrition"].sum() < 10 or train["label_attrition"].sum() < 30:
            continue
        for name, feats in FEATURE_SETS.items():
            pipe = _logistic() if kind == "logistic" else _boosting()
            pipe.fit(train[feats], train["label_attrition"])
            p = pipe.predict_proba(test[feats])[:, 1]
            y = test["label_attrition"].to_numpy()
            base = float(y.mean())
            rows.append(dict(
                fold=origin.strftime("%b %Y"), model=name,
                auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                lift=_top_decile_lift(y, p, base),
                n_test=len(test), n_positive=int(y.sum()), base_rate=base))
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, kind: str = "logistic") -> dict[str, ModelResult]:
    """Single train/test split - kept for inspecting coefficients."""
    train, test = split(df)
    out: dict[str, ModelResult] = {}
    base = float(test["label_attrition"].mean())

    for name, feats in FEATURE_SETS.items():
        pipe = _logistic() if kind == "logistic" else _boosting()
        pipe.fit(train[feats], train["label_attrition"])
        p = pipe.predict_proba(test[feats])[:, 1]
        y = test["label_attrition"].to_numpy()

        coef = pd.DataFrame()
        if kind == "logistic":
            coef = (pd.DataFrame({
                "feature": feats,
                "label": [FRIENDLY_LABEL.get(f, f) for f in feats],
                "coefficient": pipe.named_steps["clf"].coef_[0],
            }).sort_values("coefficient", key=abs, ascending=False)
              .reset_index(drop=True))

        out[name] = ModelResult(
            name, feats, float(roc_auc_score(y, p)),
            float(average_precision_score(y, p)), base,
            _top_decile_lift(y, p, base), pipe, coef)
    return out


def score_current(labelled: pd.DataFrame, scoring_rows: pd.DataFrame,
                  features: list[str] = ALL_FEATURES) -> pd.DataFrame:
    """Fit on ALL labelled data, then score each active rep's latest month."""
    pipe = _logistic()
    pipe.fit(labelled[features], labelled["label_attrition"])

    current = (scoring_rows[scoring_rows["exit_date"].isna()]
               .sort_values("month").groupby("rep_id", as_index=False).last())
    if current.empty:
        return current

    X = current[features]
    current = current.copy()
    current["risk"] = pipe.predict_proba(X)[:, 1]

    # Per-person contributions: coefficient x standardised value. This is why
    # the app uses logistic regression rather than gradient boosting - each
    # score can be read and argued about by a manager.
    Xi = pipe.named_steps["imp"].transform(X)
    Xs = pipe.named_steps["sc"].transform(Xi)
    contrib = Xs * pipe.named_steps["clf"].coef_[0]
    for j, f in enumerate(features):
        current[f"contrib__{f}"] = contrib[:, j]

    # Risk tiers are expressed as MULTIPLES of the base rate rather than
    # hand-picked round numbers. They move with the company's attrition rate,
    # and cannot be accused of being chosen to make the chart look good.
    base = float(labelled["label_attrition"].mean())
    current.attrs["base_rate"] = base
    current["risk_tier"] = pd.cut(
        current["risk"], bins=[-0.01, base, base * 2, base * 3, 1.01],
        labels=["Low", "Moderate", "High", "Critical"])
    return current.sort_values("risk", ascending=False)


def top_reasons(row: pd.Series, features: list[str],
                n: int = 4) -> list[tuple[str, float]]:
    """The largest factors pushing this person's risk UP."""
    pairs = [(FRIENDLY_LABEL.get(f, f), float(row.get(f"contrib__{f}", 0.0)))
             for f in features]
    rising = [p for p in pairs if p[1] > 0]
    return sorted(rising, key=lambda x: -x[1])[:n]


def check_calibration(df: pd.DataFrame,
                      features: list[str] = ALL_FEATURES) -> pd.DataFrame:
    """Can the output probabilities be trusted as probabilities?

    The model is used to SUM expected leaver counts, so ranking alone is not
    enough - the numbers have to be close to reality.
    """
    rows = []
    for origin in FOLDS:
        train = df[df["month"] < origin]
        test = df[(df["month"] >= origin) & (df["month"] < origin + pd.DateOffset(months=3))]
        if test["label_attrition"].sum() < 10:
            continue
        pipe = _logistic()
        pipe.fit(train[features], train["label_attrition"])
        p = pipe.predict_proba(test[features])[:, 1]
        rows.append(dict(fold=origin.strftime("%b %Y"),
                         mean_predicted=float(p.mean()),
                         actual=float(test["label_attrition"].mean()),
                         brier=float(brier_score_loss(test["label_attrition"], p))))
    return pd.DataFrame(rows)

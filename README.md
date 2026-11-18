# Sales Force People Analytics

Predicting which sales reps will resign within three months, explaining why,
and converting that into money — by joining HRIS records to sales performance
data.

> **All data in this repository is computer generated.** No row originates
> from a real company, distributor or employee.

---

## The question

People analytics teams usually start with HRIS data: attendance, leave,
tenure, employment status. The question here is simple — **is that enough?**

The same model is trained three times on different feature sets and evaluated
*rolling-origin* (trained on the past, tested on the following three months,
repeated at four points in time):

| Features | AUC | Top-decile lift |
|---|---|---|
| HR only (attendance, leave, tenure, status) | **0.58** | 2.0× |
| Sales only (attainment, incentives, quota, coverage) | **0.74** | 3.8× |
| Combined | **0.75** | 3.9× |

**The answer is no.** HRIS data on its own is close to useless for this
problem — AUC 0.58, barely better than guessing. The signal lives in
commercial performance. HR data earns its place only as a complement, and that
gain is consistent across all four folds.

This is an uncomfortable finding but a useful one: a team looking only at the
HR dashboard would conclude attrition cannot be predicted, when what is
actually missing is the join to a system another team owns.

## Other findings

- **Unreasonable quotas are the strongest driver.** Reps targeted above 1.15×
  their territory's fair capacity resign at **84%**, against 53–59% in the
  other three bands. The relationship is notably **not gradual** — "loose",
  "fair" and "tight" quotas are almost identical. What drives people out is
  not a demanding target but an impossible one.
- **The incentive cliff.** The incentive scheme jumps from 0.35× to 0.85× of
  base salary exactly at 100% attainment. Reps who repeatedly land at 90–99%
  work hard without ever reaching the large tier.
- **One resignation costs far more than retaining someone.** Under the default
  assumptions a single resignation costs roughly IDR 250 million, while an
  intervention programme covering the 20 highest-risk reps costs IDR 70
  million and returns about 1.5×. That result is not robust, deliberately: drop
  the intervention success rate to 10% and it falls to −0.3×; raise the cost per
  person to IDR 10 million and it falls to −0.1×. The sliders in the app are
  there so the reader can find the point where the programme stops paying for
  itself, rather than being handed one flattering number.

## Running it

```bash
pip install -r requirements.txt
python src/generate_data.py      # writes synthetic data to data/
streamlit run app.py
```

Smoke test, which needs neither Streamlit nor Plotly installed:

```bash
python tests/smoke_test.py
```

## What is in here

```
app.py                  Streamlit app, four tabs
src/generate_data.py    synthetic data generator (420 reps x 36 months)
src/features.py         feature engineering and labelling, HR vs sales split
src/model.py            risk model, rolling-origin evaluation, calibration check
src/economics.py        attrition cost and intervention simulation
src/theme.py            chart palette and styling
tests/smoke_test.py     runs app.py against Streamlit/Plotly stubs
ASSUMPTIONS.md          every reference figure with its source
```

## Technical decisions, and why

**Temporal split, not random.** A random split would put the same person's
March row in training and their February row in test — the future leaks into
the past and the score becomes meaningless. Here the model only ever trains on
months before the split point.

**Four folds, not one.** On data this size a single split moves a long way
purely on who happened to resign inside the test window — in early runs AUC
swung between 0.48 and 0.88 with identical structure. What is reported is the
mean across four folds, with the spread.

**Logistic regression, not gradient boosting.** Gradient boosting was tested
and was not better at this data size. Logistic regression was chosen because
each factor's contribution is legible: the watchlist shows per-person reasons
rather than a black-box score nobody can discuss with a manager.

**No `class_weight="balanced"`.** Re-weighting raises AUC by about 0.005 but
destroys calibration outright — mean predicted probability jumps to 0.35 when
the real event rate is 0.06. The entire economics module sums probabilities to
estimate leaver counts, so calibration is worth more than AUC here. After the
fix, mean predicted probability is 5.8% against an actual rate of 6.0%.

**Right censoring on the labels.** The final three months are dropped from the
labelled set because nobody's future is observed there yet. Dropping only the
still-active reps would leave those months filled almost entirely with leavers
and bias the evaluation.

**Risk tiers as multiples of the base rate**, not hand-picked round numbers,
so they move with the company's attrition rate and cannot be accused of being
chosen to make the chart look good.

## Limitations

The data is synthetic: the relationships were planted by the generator, so the
model succeeds at recovering something deliberately put there. What this
repository demonstrates is the **method** — how to frame the question, avoid
leakage, test calibration, and turn a model into a money decision — not
empirical evidence about any company.

The cost references come from US and B2B sources because public Indonesian
benchmarks for sales force turnover cost are effectively non-existent. All of
them are listed in `ASSUMPTIONS.md` and adjustable in the app.

And most importantly: this is correlation, not causation. Unreasonable quotas
correlate with resignation in this data; proving causation would require an
experiment, not this model.

---

Muhammad Irsyad Dzulfaqor — Data Analyst, Commercial Operations

# Assumptions and reference figures

Every number used to convert attrition risk into money is listed here with its
source, so the assumptions can be argued with. A cost analysis whose
assumptions are hidden cannot be used to make a decision.

## Cost of replacing one rep

| Assumption | Value used | Reference |
|---|---|---|
| Recruitment + onboarding | 1.0 × annual OTE | Industry reference 1.0–2.0 × OTE. The lower bound is used to stay conservative. |
| Vacant territory | 2.0 months | Time-to-fill for account executives is 45–70 days. |
| Ramp-up duration | 6 months | See the ramp-up section below. |
| Contribution margin | 12% | No public reference; a placeholder for FMCG distribution that **must be replaced** with the company's real margin. |

OTE is on-target earnings: base salary plus incentive at full quota
attainment. Every figure here is adjustable via sliders in the app.

## Ramp-up

The available public references are based on B2B software sales, not FMCG:

| Segment | 50% productivity | Full productivity |
|---|---|---|
| Inside / SMB | 2–3 months | 3–4 months |
| Mid-market | 3–4 months | 6–9 months |
| Enterprise / field | 5–6 months | 9–12+ months |

This project uses **6 months to full productivity**, faster than the B2B field
figure because FMCG distribution sales cycles are far shorter — a new rep
starts taking orders within weeks, not quarters. This is a judgement call, not
a measured figure, and is flagged as such.

## Turnover rates

| Measure | Reference |
|---|---|
| Total sales force turnover | 20–35% a year; best-in-class 15–20% |
| Voluntary turnover | 12–22% a year |
| First-year attrition | 25–40% in weaker programmes |

The synthetic data here is calibrated to roughly **29% a year**, inside the
reference range. No public benchmark exists specifically for FMCG field sales
or distribution; the figures above come from B2B account executive
populations.

## FMCG field KPIs

Used to sanity-check that the synthetic data is plausible, not to build the
cost model. These figures are specific to the Indian market and their
publisher does not cite a primary source, so they are treated as indicative
only.

| KPI | Reference | In the synthetic data |
|---|---|---|
| Strike rate (general trade) | 60–75% | mean 75% |
| Active outlet coverage | > 85% | mean 82% |
| Lines per call | 4–6 | mean 4.9 |
| Quota attainment | > 90% | median 99% |

## Limitations the reader should know

1. **The data is synthetic.** The relationships between variables were
   deliberately planted by the generator, so the model succeeds at recovering
   something that was put there on purpose. What this repository demonstrates
   is the **method**, not evidence that attendance predicts resignation at any
   particular company.
2. **The cost references are US and B2B**, not Indonesian. They are used
   because public Indonesian benchmarks for sales force turnover cost are
   effectively non-existent. For real use, replace them with internal figures.
3. **Correlation is not causation.** Unreasonable quotas correlate with
   resignation in this data; proving causation would require an experiment or
   a quasi-experimental design, not this model.

## Sources

- [Ramp-up time for new reps — Umbrex](https://umbrex.com/resources/company-analysis/sales/ramp-up-time-for-new-reps/)
- [Salesforce turnover rate — Umbrex](https://umbrex.com/resources/company-analysis/sales/salesforce-turnover-rate/)
- [Average sales turnover rates — Map My Customers](https://mapmycustomers.com/blog/average-sales-turnover-rates/)
- [FMCG sales KPIs with benchmarks — MAssist CRM](https://blog.massistcrm.com/10-kpis-every-fmcg-sales-manager-must-track-with-benchmarks)

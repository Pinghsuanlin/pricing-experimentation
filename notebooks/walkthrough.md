# Walkthrough

A step-by-step tour of the pipeline in plain Python. Paste the blocks into a
notebook or a REPL started with `uv run python`. No outputs are shown here on
purpose: run it on the data to see your own numbers.

Set `DEMO = True` to use synthetic data if you have not downloaded the CSV.

```python
DEMO = False

from pricing_lab import cuped, did, elasticity, power, simulate
from pricing_lab.data import add_derived_columns, aggregate_demand, clean_bookings, load_bookings
from pricing_lab.synthetic import make_synthetic_bookings

raw = make_synthetic_bookings() if DEMO else load_bookings("data/raw/hotel_bookings.csv")
clean, report = clean_bookings(add_derived_columns(raw))
report
```

## 1. Demand and price panel

One row per hotel x market segment x arrival week. `bookings` counts
non-cancelled stays (`demand="net"`); `demand="gross"` counts every booking.
`adr` is the mean nightly rate of the counted bookings.

```python
panel = aggregate_demand(clean, demand="net")
panel.head()
panel.groupby("market_segment")[["bookings", "adr", "cancel_rate"]].mean()
```

Things to look at: how much of the price variation is seasonal (plot `adr` by
`week_of_year`), and how cancellation rates differ by segment.

## 2. Hold out the last 26 weeks

The elasticity is estimated on earlier weeks only. The held-out weeks are kept
for the simulated test, so the effect we inject is never fitted on the same data.

```python
cutoff = simulate.holdout_cutoff(panel, n_weeks=26)
train = panel.loc[panel["period"] < cutoff]
frame = elasticity.build_model_frame(train)
```

## 3. Elasticity

```python
main = elasticity.fit_log_log(frame)                 # HC1 robust SEs
print(main.summary())

by_week = elasticity.fit_log_log(frame, cov_type="cluster", cluster_col="period")
monthly = elasticity.fit_log_log(frame, fixed_effects=("month", "year", "hotel", "market_segment"))
segments = elasticity.per_segment_elasticity(frame)
segments
```

Reading it: a negative coefficient means bookings fall when price rises. A
coefficient near zero or positive is the expected symptom of price endogeneity
(price follows demand), not evidence that guests ignore price.

The instrumental-variable check:

```python
iv_frame = elasticity.add_hausman_instrument(frame, name="z")
iv = elasticity.fit_2sls(iv_frame, "z")
iv.elasticity, iv.first_stage_f
```

Treat a first-stage F below 10 as a weak instrument. Even with a strong first
stage, the exclusion restriction (the other hotel's price affects this hotel's
bookings only through this hotel's price) is an assumption you cannot test.

## 4. Experiment design

```python
power.n_per_arm_two_proportions(baseline=0.10, mde=0.02)        # textbook: 3,839
mean_rpv, sd_rpv = power.rpv_moments(conversion=0.05, mean_value=300, sd_value=250)
power.n_per_arm_continuous(sd=sd_rpv, mde=0.05 * mean_rpv)

power.mde_duration_table(
    daily_units=5000, weeks=range(1, 9), metric="continuous",
    baseline=mean_rpv, sd=sd_rpv,
)
```

The visitor numbers above are inputs you choose, not facts about these hotels.
If you randomise clusters instead of visitors, pass `deff=power.design_effect(m, icc)`.

## 5. SIMULATED price test

```python
units = aggregate_demand(clean, group_cols=simulate.DEFAULT_UNIT_COLS, demand="net")
holdout = units.loc[units["period"] >= cutoff]
config = simulate.SimulationConfig(
    price_change=0.05, elasticity=-1.0, pre_weeks=13, post_weeks=13, seed=42
)
sim = simulate.simulate_price_test(holdout, config)
simulate.true_effects(config.price_change, config.elasticity)
```

`elasticity=-1.0` is an assumption. Swap in `main.elasticity` if it is negative
and you believe it.

## 6. Readout

```python
did.fit_did(sim)                                   # TWFE, unit-clustered SEs
did.fit_did(sim, twfe=False)                       # textbook 2x2
did.fit_did(sim, outcome="log_revenue_sim")        # revenue, not just volume
did.pre_trend_test(sim)                            # differential pre-trend slope
did.event_study(sim)                               # week-by-week gaps
did.permutation_test(sim)                          # randomisation inference

means = did.unit_pre_post_means(sim)
cuped.cuped_ate(means["post"], means["pre"], means["treated"])
```

Questions to answer from the output:

1. Does the DiD interval cover the true (injected) effect?
2. Are the pre-launch event-study points near zero?
3. How many units passed the volume filter, and do the cluster-robust and
   permutation p-values agree?
4. How much variance did CUPED remove, and how does it compare with DiD?
5. Rerun with 10 seeds. How often does the interval cover the truth?


"""The elasticity estimators recover a known elasticity from synthetic panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing_lab.elasticity import (
    add_hausman_instrument,
    build_model_frame,
    drop_singletons,
    fit_2sls,
    fit_log_log,
    per_segment_elasticity,
)

TRUE_ELASTICITY = -1.5
SEGMENTS = ("A", "B", "C")
HOTELS = ("City", "Resort")


def _panel(
    elasticities: dict[str, float] | None = None,
    n_weeks: int = 104,
    endogenous: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Weekly hotel x segment panel with log(bookings) linear in log(price).

    Exogenous case: price moves with seasonality (absorbed by week-of-year
    fixed effects) plus a random shock unrelated to demand.
    Endogenous case: a demand shock ``u`` raises both price and bookings, and a
    cost shifter ``z`` moves price only (a valid instrument).
    """
    elasticities = elasticities or dict.fromkeys(SEGMENTS, TRUE_ELASTICITY)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-01-05")
    rows = []
    for week in range(n_weeks):
        period = start + pd.Timedelta(weeks=week)
        woy = int(period.isocalendar().week)
        season = np.sin(2 * np.pi * woy / 52)
        for h, hotel in enumerate(HOTELS):
            for s, segment in enumerate(SEGMENTS):
                u = rng.normal(0, 0.2) if endogenous else 0.0
                z = rng.normal(0, 1.0)
                log_price = 4.5 + 0.3 * h + 0.2 * s + 0.15 * season + 0.1 * rng.normal()
                log_price += (0.8 * u + 0.3 * z) if endogenous else 0.1 * rng.normal()
                lead_time = rng.uniform(10, 120)
                log_q = (
                    11.0
                    + 0.5 * s
                    + 0.4 * season
                    + elasticities[segment] * log_price
                    + 0.1 * np.log1p(lead_time)
                    + u
                    + rng.normal(0, 0.1)
                )
                rows.append(
                    {
                        "hotel": hotel,
                        "market_segment": segment,
                        "period": period,
                        "year": int(period.isocalendar().year),
                        "week_of_year": woy,
                        "month": period.month,
                        "bookings": float(np.exp(log_q)),
                        "adr": float(np.exp(log_price)),
                        "lead_time": lead_time,
                        "z_cost": z,
                    }
                )
    return pd.DataFrame(rows)


def test_log_log_recovers_known_elasticity() -> None:
    frame = build_model_frame(_panel())
    res = fit_log_log(frame)
    assert res.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.1)
    assert res.ci_low < TRUE_ELASTICITY < res.ci_high
    assert res.n_obs == len(frame)


def test_clustered_and_month_fixed_effects_variants_run() -> None:
    frame = build_model_frame(_panel())
    clustered = fit_log_log(frame, cov_type="cluster", cluster_col="period")
    assert clustered.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.1)
    monthly = fit_log_log(frame, fixed_effects=("month", "year", "hotel", "market_segment"))
    assert monthly.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.2)
    with pytest.raises(ValueError):
        fit_log_log(frame, cov_type="cluster")


def test_per_segment_recovers_each_elasticity() -> None:
    truth = {"A": -0.8, "B": -1.5, "C": -2.2}
    table = per_segment_elasticity(build_model_frame(_panel(elasticities=truth)), min_obs=40)
    assert set(table["market_segment"]) == set(truth)
    for _, row in table.iterrows():
        assert row["elasticity"] == pytest.approx(truth[row["market_segment"]], abs=0.2)


def test_drop_singletons_removes_lone_fixed_effect_levels() -> None:
    frame = pd.DataFrame({"week": [1, 1, 2, 3, 3], "seg": ["a", "b", "a", "a", "b"]})
    out = drop_singletons(frame, ["week", "seg"])
    assert list(out.index) == [0, 1, 3, 4]
    # Dropping week 2 leaves no singletons in "seg", so one pass is enough here.
    lone = pd.DataFrame({"week": [1, 1, 2, 2], "seg": ["a", "a", "a", "b"]})
    assert list(drop_singletons(lone, ["week", "seg"]).index) == [0, 1]


def test_build_model_frame_rejects_zero_floor() -> None:
    with pytest.raises(ValueError):
        build_model_frame(_panel(n_weeks=4), min_bookings=0)


def test_2sls_fixes_endogeneity_bias_that_ols_has() -> None:
    frame = build_model_frame(_panel(endogenous=True, seed=1))
    ols = fit_log_log(frame)
    iv = fit_2sls(frame, instrument="z_cost")
    assert iv.first_stage_f > 10
    assert iv.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.15)
    # Demand shocks push price and bookings up together, so OLS is biased upward.
    assert ols.elasticity - TRUE_ELASTICITY > 0.1
    assert abs(iv.elasticity - TRUE_ELASTICITY) < abs(ols.elasticity - TRUE_ELASTICITY)


def test_hausman_instrument_is_the_other_hotels_price() -> None:
    frame = pd.DataFrame(
        {
            "hotel": ["City", "Resort", "City"],
            "market_segment": ["A", "A", "B"],
            "period": pd.to_datetime(["2016-01-04"] * 3),
            "log_price": [4.0, 5.0, 4.5],
        }
    )
    out = add_hausman_instrument(frame, name="z")
    assert out["z"].iloc[0] == pytest.approx(5.0)
    assert out["z"].iloc[1] == pytest.approx(4.0)
    assert np.isnan(out["z"].iloc[2])

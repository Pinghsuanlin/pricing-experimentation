"""Simulation mechanics and an end-to-end check that DiD recovers the injected effect."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pricing_lab.did import fit_did
from pricing_lab.simulate import (
    SimulationConfig,
    apply_price_change,
    assign_matched_pairs,
    holdout_cutoff,
    simulate_price_test,
    true_effects,
)


def _weekly_units(n_units: int = 30, n_weeks: int = 26, seed: int = 0) -> pd.DataFrame:
    """Untreated weekly panel shaped like ``aggregate_demand`` output."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-01-02")
    unit_scale = rng.uniform(80, 300, n_units)
    week_shock = np.exp(rng.normal(0, 0.1, n_weeks))
    rows = []
    for i in range(n_units):
        for w in range(n_weeks):
            bookings = int(rng.poisson(unit_scale[i] * week_shock[w]))
            adr = 100 + 5 * i
            rows.append(
                {
                    "hotel": "City Hotel" if i % 2 else "Resort Hotel",
                    "market_segment": f"seg{i // 2}",
                    "distribution_channel": "TA/TO",
                    "period": start + pd.Timedelta(weeks=w),
                    "bookings": bookings,
                    "adr": float(adr),
                    "revenue": float(bookings * adr * 3),
                }
            )
    return pd.DataFrame(rows)


def test_true_effects_formulas() -> None:
    unit = true_effects(price_change=0.10, elasticity=-1.0)
    assert unit.bookings_log == pytest.approx(-math.log(1.1))
    assert unit.revenue_log == pytest.approx(0.0)
    elastic = true_effects(price_change=0.10, elasticity=-2.0)
    assert elastic.revenue_pct < 0


def test_apply_price_change_scales_counts_in_expectation() -> None:
    rng = np.random.default_rng(0)
    counts = np.full(2000, 500)
    assert np.array_equal(apply_price_change(counts, 1.0, rng), counts)
    assert apply_price_change(counts, 0.9, rng).mean() / 500 == pytest.approx(0.9, abs=0.005)
    assert apply_price_change(counts, 1.1, rng).mean() / 500 == pytest.approx(1.1, abs=0.005)
    with pytest.raises(ValueError):
        apply_price_change(counts, 0.0, rng)


def test_matched_pairs_treat_one_unit_per_pair() -> None:
    volume = pd.Series(np.arange(10, dtype=float), index=[f"u{i}" for i in range(10)])
    assigned = assign_matched_pairs(volume, np.random.default_rng(0))
    assert assigned["treated"].sum() == 5
    assert (assigned.groupby("pair_id")["treated"].sum() == 1).all()


def test_holdout_cutoff() -> None:
    panel = _weekly_units(n_units=2, n_weeks=10)
    cutoff = holdout_cutoff(panel, 4)
    assert (panel["period"] >= cutoff).sum() == 2 * 4
    with pytest.raises(ValueError):
        holdout_cutoff(panel, 10)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(price_change=-1.0)
    with pytest.raises(ValueError):
        SimulationConfig(min_weekly_bookings=0)


def test_simulation_leaves_controls_and_pre_period_untouched() -> None:
    config = SimulationConfig(price_change=0.10, elasticity=-1.5, pre_weeks=13, post_weeks=13)
    sim = simulate_price_test(_weekly_units(), config)
    untouched = (sim["treated"] == 0) | (sim["post"] == 0)
    assert (sim.loc[untouched, "bookings_sim"] == sim.loc[untouched, "bookings"]).all()
    assert sim["is_simulated"].all()
    assert sim["rel_week"].min() == -13 and sim["rel_week"].max() == 12
    exposed = ~untouched
    assert np.allclose(sim.loc[exposed, "adr_sim"], sim.loc[exposed, "adr"] * 1.10)


def test_did_recovers_simulated_effect() -> None:
    config = SimulationConfig(price_change=0.10, elasticity=-1.5, pre_weeks=13, post_weeks=13)
    truth = true_effects(config.price_change, config.elasticity)
    sim = simulate_price_test(_weekly_units(), config)
    bookings = fit_did(sim, outcome="log_bookings_sim")
    revenue = fit_did(sim, outcome="log_revenue_sim")
    assert bookings.effect == pytest.approx(truth.bookings_log, abs=0.04)
    assert revenue.effect == pytest.approx(truth.revenue_log, abs=0.04)


def test_simulation_rejects_wrong_window() -> None:
    config = SimulationConfig(pre_weeks=10, post_weeks=10)
    with pytest.raises(ValueError, match="expected 20"):
        simulate_price_test(_weekly_units(), config)

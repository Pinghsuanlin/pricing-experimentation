"""DiD, pre-trend, event-study and permutation checks on panels with a known effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing_lab.did import (
    event_study,
    fit_did,
    permutation_test,
    pre_trend_test,
    unit_pre_post_means,
)

TRUE_EFFECT = -0.08


def _panel(
    effect: float = TRUE_EFFECT,
    pre_trend: float = 0.0,
    n_units: int = 40,
    pre: int = 8,
    post: int = 8,
    seed: int = 1,
) -> pd.DataFrame:
    """y = unit effect + common week shock + effect * treated * post + noise.

    ``pre_trend`` adds a treated-only linear drift, which breaks parallel trends.
    """
    rng = np.random.default_rng(seed)
    unit_fe = rng.normal(3.0, 0.5, n_units)
    week_fe = rng.normal(0.0, 0.1, pre + post)
    # Matched pairs (0, 1), (2, 3), ...: one unit in each pair is treated.
    treated = np.zeros(n_units, dtype=int)
    treated[np.arange(0, n_units, 2) + rng.integers(0, 2, n_units // 2)] = 1
    rows = []
    for i in range(n_units):
        for t, rel in enumerate(range(-pre, post)):
            is_post = int(rel >= 0)
            y = (
                unit_fe[i]
                + week_fe[t]
                + effect * treated[i] * is_post
                + pre_trend * treated[i] * rel
                + rng.normal(0, 0.05)
            )
            rows.append(
                {
                    "unit": f"u{i}",
                    "rel_week": rel,
                    "post": is_post,
                    "treated": treated[i],
                    "pair_id": i // 2,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def test_twfe_recovers_known_effect() -> None:
    res = fit_did(_panel(), outcome="y")
    assert res.effect == pytest.approx(TRUE_EFFECT, abs=0.03)
    assert res.ci_low < TRUE_EFFECT < res.ci_high
    assert res.p_value < 0.01
    assert res.n_units == 40
    assert res.pct_effect == pytest.approx(np.expm1(res.effect))


def test_twfe_matches_2x2_on_balanced_panel() -> None:
    panel = _panel()
    twfe = fit_did(panel, outcome="y", twfe=True)
    simple = fit_did(panel, outcome="y", twfe=False)
    assert twfe.effect == pytest.approx(simple.effect, abs=1e-8)


def test_no_effect_is_not_significant() -> None:
    res = fit_did(_panel(effect=0.0, seed=3), outcome="y")
    assert res.ci_low < 0.0 < res.ci_high


def test_pre_trend_test_flags_diverging_groups() -> None:
    assert pre_trend_test(_panel(), outcome="y").p_value > 0.05
    drifting = pre_trend_test(_panel(pre_trend=0.02), outcome="y")
    assert drifting.p_value < 0.01
    assert drifting.slope == pytest.approx(0.02, abs=0.01)


def test_event_study_shape_and_post_coefficients() -> None:
    table = event_study(_panel(), outcome="y")
    assert len(table) == 16
    assert table.loc[table["rel_week"] == -1, "coef"].item() == 0.0
    post_mean = table.loc[table["rel_week"] >= 0, "coef"].mean()
    assert post_mean == pytest.approx(TRUE_EFFECT, abs=0.03)


def test_permutation_test_detects_effect_and_matches_did() -> None:
    panel = _panel()
    res = permutation_test(panel, outcome="y", n_permutations=500, seed=0)
    assert res.p_value < 0.05
    assert res.observed == pytest.approx(fit_did(panel, outcome="y").effect, abs=1e-8)
    null = permutation_test(_panel(effect=0.0, seed=3), outcome="y", pair_col=None, seed=0)
    assert null.p_value > 0.05


def test_unit_pre_post_means() -> None:
    means = unit_pre_post_means(_panel(n_units=6), outcome="y")
    assert list(means.columns) == ["pre", "post", "treated"]
    assert len(means) == 6

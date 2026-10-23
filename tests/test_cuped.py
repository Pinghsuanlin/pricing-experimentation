"""CUPED: correct theta, variance reduction close to rho squared, no bias."""

from __future__ import annotations

import numpy as np
import pytest

from pricing_lab.cuped import cuped_adjust, cuped_ate, cuped_theta, variance_reduction


def _data(
    n: int = 20000, rho: float = 0.7, effect: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcome correlated with a pre-period covariate, plus a random treatment."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y_base = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    treated = rng.integers(0, 2, n)
    return y_base + effect * treated, x, treated


def test_theta_is_the_regression_slope() -> None:
    y, x, _ = _data(effect=0.0)
    slope = np.polyfit(x, y, 1)[0]
    assert cuped_theta(y, x) == pytest.approx(slope, rel=1e-6)


def test_variance_reduction_close_to_rho_squared() -> None:
    y, x, _ = _data(effect=0.0)
    assert variance_reduction(y, x) == pytest.approx(0.49, abs=0.02)


def test_adjustment_keeps_the_mean() -> None:
    y, x, _ = _data()
    assert cuped_adjust(y, x).mean() == pytest.approx(y.mean())


def test_ate_is_unbiased_and_tighter() -> None:
    y, x, treated = _data()
    res = cuped_ate(y, x, treated)
    assert res.adjusted_effect == pytest.approx(0.5, abs=0.03)
    assert res.raw_effect == pytest.approx(0.5, abs=0.05)
    assert res.adjusted_std_error < res.raw_std_error
    assert res.adjusted_ci_low < 0.5 < res.adjusted_ci_high
    assert res.n_treated + res.n_control == len(y)


def test_constant_covariate_raises() -> None:
    y, _, _ = _data(n=100)
    with pytest.raises(ValueError):
        cuped_theta(y, np.ones_like(y))

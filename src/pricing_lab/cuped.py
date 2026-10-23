"""CUPED variance reduction (Deng, Xu, Kohavi & Walker, 2013).

Adjust the outcome with a pre-experiment covariate ``x``:

    y_adj = y - theta * (x - mean(x)),   theta = cov(y, x) / var(x)

``theta`` is estimated on both arms pooled, so it does not depend on the
treatment. The adjustment removes the part of ``y`` that ``x`` predicts,
cutting the variance by ``corr(y, x) ** 2`` without biasing the effect,
because ``x`` was measured before assignment.

Link to DiD: comparing ``post - pre`` is CUPED with ``theta`` fixed at 1.
CUPED picks the ``theta`` that minimises variance instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class CupedResult:
    """Difference in means before and after the CUPED adjustment."""

    raw_effect: float
    raw_std_error: float
    raw_ci_low: float
    raw_ci_high: float
    adjusted_effect: float
    adjusted_std_error: float
    adjusted_ci_low: float
    adjusted_ci_high: float
    theta: float
    variance_reduction: float
    n_treated: int
    n_control: int


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Regression slope of ``y`` on ``x``: cov(y, x) / var(x)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        raise ValueError("covariate has zero variance; CUPED cannot help")
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """Return the CUPED-adjusted outcome."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = cuped_theta(y, x) if theta is None else theta
    return y - theta * (x - x.mean())


def variance_reduction(y: np.ndarray, x: np.ndarray) -> float:
    """Share of outcome variance removed: 1 - var(y_adj) / var(y)."""
    y = np.asarray(y, dtype=float)
    return float(1 - np.var(cuped_adjust(y, x), ddof=1) / np.var(y, ddof=1))


def _diff_in_means(
    y: np.ndarray, treated: np.ndarray, alpha: float
) -> tuple[float, float, float, float]:
    """Welch difference in means with a normal-approximation interval."""
    yt, yc = y[treated == 1], y[treated == 0]
    if len(yt) < 2 or len(yc) < 2:
        raise ValueError("each arm needs at least 2 observations")
    effect = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    crit = NormalDist().inv_cdf(1 - alpha / 2)
    return effect, se, effect - crit * se, effect + crit * se


def cuped_ate(
    y: np.ndarray, x: np.ndarray, treated: np.ndarray, alpha: float = 0.05
) -> CupedResult:
    """Average treatment effect with and without the CUPED adjustment.

    Args:
        y: Post-period outcome per unit.
        x: Pre-period covariate per unit (same metric before launch works best).
        treated: 1 for treatment, 0 for control.
        alpha: Significance level for the intervals.

    The intervals use the normal approximation. With fewer than about 30
    units per arm, read them as indicative and prefer the permutation p-value.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    treated = np.asarray(treated, dtype=int)
    theta = cuped_theta(y, x)
    raw = _diff_in_means(y, treated, alpha)
    adjusted = _diff_in_means(cuped_adjust(y, x, theta), treated, alpha)
    return CupedResult(
        raw_effect=raw[0],
        raw_std_error=raw[1],
        raw_ci_low=raw[2],
        raw_ci_high=raw[3],
        adjusted_effect=adjusted[0],
        adjusted_std_error=adjusted[1],
        adjusted_ci_low=adjusted[2],
        adjusted_ci_high=adjusted[3],
        theta=theta,
        variance_reduction=variance_reduction(y, x),
        n_treated=int((treated == 1).sum()),
        n_control=int((treated == 0).sum()),
    )

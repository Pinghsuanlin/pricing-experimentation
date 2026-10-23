"""Sample size, power and minimum detectable effect (MDE) for a price test.

Uses the normal approximation and only the standard library (``statistics``),
so the formulas are easy to check by hand.

Conventions:
* ``mde`` is an absolute difference (treatment minus control).
* ``ratio`` is treatment size / control size. Functions return the control
  arm size; the treatment arm is ``ratio`` times that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

import pandas as pd

Metric = Literal["proportion", "continuous"]
_NORMAL = NormalDist()


def _check_rates(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")


def z_alpha(alpha: float, two_sided: bool = True) -> float:
    """Critical value of the standard normal for the chosen test."""
    return _NORMAL.inv_cdf(1 - alpha / 2) if two_sided else _NORMAL.inv_cdf(1 - alpha)


def z_power(power: float) -> float:
    """Standard normal quantile for the target power."""
    return _NORMAL.inv_cdf(power)


def _ceil(x: float) -> int:
    """Ceiling that ignores floating-point noise just above an integer."""
    return math.ceil(x - 1e-9)


def n_per_arm_two_proportions(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
    pooled: bool = False,
) -> int:
    """Control-arm size to detect ``baseline -> baseline + mde`` in a proportion.

    Unpooled (default): n = (z_a + z_b)^2 * (p1 q1 + p2 q2 / r) / mde^2.
    Pooled uses the pooled variance under the null for the z_a term, which is
    slightly more conservative. For p=0.10, mde=0.02, alpha=0.05, power=0.80
    the unpooled answer is 3,839 per arm.
    """
    _check_rates(alpha, power)
    p1, p2 = baseline, baseline + mde
    if mde == 0:
        raise ValueError("mde must be non-zero")
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("baseline and baseline + mde must both be in (0, 1)")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    za, zb = z_alpha(alpha, two_sided), z_power(power)
    var_alt = p1 * (1 - p1) + p2 * (1 - p2) / ratio
    if pooled:
        p_bar = (p1 + ratio * p2) / (1 + ratio)
        var_null = p_bar * (1 - p_bar) * (1 + 1 / ratio)
        numerator = (za * math.sqrt(var_null) + zb * math.sqrt(var_alt)) ** 2
    else:
        numerator = (za + zb) ** 2 * var_alt
    return _ceil(numerator / mde**2)


def n_per_arm_continuous(
    sd: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> int:
    """Control-arm size to detect a mean difference ``mde`` with common ``sd``.

    n = (z_a + z_b)^2 * sd^2 * (1 + 1/r) / mde^2. For sd=1, mde=0.2 (a 0.2
    standard-deviation effect) this gives 393 per arm.
    """
    _check_rates(alpha, power)
    if sd <= 0 or mde == 0 or ratio <= 0:
        raise ValueError("sd and ratio must be positive and mde non-zero")
    za, zb = z_alpha(alpha, two_sided), z_power(power)
    return _ceil((za + zb) ** 2 * sd**2 * (1 + 1 / ratio) / mde**2)


def mde_continuous(
    sd: float,
    n_per_arm: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Smallest absolute mean difference detectable with ``n_per_arm`` controls."""
    _check_rates(alpha, power)
    if sd <= 0 or n_per_arm <= 0 or ratio <= 0:
        raise ValueError("sd, n_per_arm and ratio must be positive")
    za, zb = z_alpha(alpha, two_sided), z_power(power)
    return (za + zb) * sd * math.sqrt((1 + 1 / ratio) / n_per_arm)


def mde_two_proportions(
    baseline: float,
    n_per_arm: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Smallest absolute uplift detectable from ``baseline`` (unpooled formula).

    The variance depends on the treatment rate, so this solves
    ``(z_a + z_b) * sqrt((p1 q1 + p2 q2 / r) / n) = mde`` by bisection.
    """
    _check_rates(alpha, power)
    if not 0 < baseline < 1 or n_per_arm <= 0 or ratio <= 0:
        raise ValueError("baseline must be in (0, 1); n_per_arm and ratio positive")
    k = z_alpha(alpha, two_sided) + z_power(power)

    def gap(d: float) -> float:
        p2 = baseline + d
        var = baseline * (1 - baseline) + p2 * (1 - p2) / ratio
        return k * math.sqrt(var / n_per_arm) - d

    low, high = 0.0, 1 - baseline - 1e-12
    if gap(high) > 0:
        raise ValueError("n_per_arm is too small to detect any uplift from this baseline")
    for _ in range(200):
        mid = (low + high) / 2
        if gap(mid) > 0:
            low = mid
        else:
            high = mid
    return high


def power_two_proportions(
    baseline: float,
    treatment: float,
    n_per_arm: float,
    alpha: float = 0.05,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Achieved power for a two-proportion z-test (unpooled variance)."""
    se = math.sqrt(
        baseline * (1 - baseline) / n_per_arm
        + treatment * (1 - treatment) / (ratio * n_per_arm)
    )
    return _power_from_z(abs(treatment - baseline) / se, alpha, two_sided)


def power_continuous(
    sd: float,
    mde: float,
    n_per_arm: float,
    alpha: float = 0.05,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Achieved power for a two-sample z-test on means."""
    se = sd * math.sqrt((1 + 1 / ratio) / n_per_arm)
    return _power_from_z(abs(mde) / se, alpha, two_sided)


def _power_from_z(z_effect: float, alpha: float, two_sided: bool) -> float:
    za = z_alpha(alpha, two_sided)
    result = _NORMAL.cdf(z_effect - za)
    if two_sided:
        result += _NORMAL.cdf(-z_effect - za)
    return result


def rpv_moments(conversion: float, mean_value: float, sd_value: float) -> tuple[float, float]:
    """Mean and sd of revenue per visitor (RPV).

    RPV is zero for non-converters and the booking value for converters, so
    E[R] = p * mu and Var[R] = p * (sigma^2 + mu^2) - (p * mu)^2.
    """
    if not 0 < conversion <= 1 or sd_value < 0:
        raise ValueError("conversion must be in (0, 1] and sd_value non-negative")
    mean = conversion * mean_value
    var = conversion * (sd_value**2 + mean_value**2) - mean**2
    return mean, math.sqrt(max(var, 0.0))


def design_effect(cluster_size: float, icc: float) -> float:
    """Variance inflation from randomising clusters: 1 + (m - 1) * ICC.

    Use it when the randomisation unit is bigger than the analysis unit (for
    example randomising hotels or cities but measuring bookings). Divide the
    nominal sample size by this to get the effective sample size.
    """
    if cluster_size < 1 or not 0 <= icc <= 1:
        raise ValueError("cluster_size must be >= 1 and icc in [0, 1]")
    return 1 + (cluster_size - 1) * icc


def mde_duration_table(
    daily_units: float,
    weeks: Sequence[int],
    metric: Metric,
    baseline: float,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    treat_share: float = 0.5,
    deff: float = 1.0,
) -> pd.DataFrame:
    """MDE for each test length, given daily eligible traffic.

    Args:
        daily_units: Eligible units per day across both arms.
        weeks: Test lengths to evaluate.
        metric: ``"proportion"`` (``baseline`` is the rate) or ``"continuous"``
            (``baseline`` is the control mean and ``sd`` is required).
        treat_share: Share of traffic in treatment.
        deff: Design effect; effective n = nominal n / deff.

    Returns:
        Columns ``weeks``, ``control_n``, ``treatment_n``, ``effective_control_n``,
        ``mde_abs`` and ``mde_rel`` (MDE as a share of ``baseline``).
    """
    if not 0 < treat_share < 1:
        raise ValueError("treat_share must be in (0, 1)")
    if metric == "continuous" and (sd is None or sd <= 0):
        raise ValueError("sd is required and must be positive for a continuous metric")
    ratio = treat_share / (1 - treat_share)
    sd_value = float(sd) if sd is not None else 0.0
    rows = []
    for w in weeks:
        control_n = daily_units * 7 * w * (1 - treat_share)
        effective_n = control_n / deff
        if metric == "proportion":
            mde = mde_two_proportions(baseline, effective_n, alpha, power, ratio=ratio)
        else:
            mde = mde_continuous(sd_value, effective_n, alpha, power, ratio=ratio)
        rows.append(
            {
                "weeks": int(w),
                "control_n": int(control_n),
                "treatment_n": int(control_n * ratio),
                "effective_control_n": int(effective_n),
                "mde_abs": mde,
                "mde_rel": mde / baseline if baseline else float("nan"),
            }
        )
    return pd.DataFrame(rows)

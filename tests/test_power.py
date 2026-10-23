"""Power and sample-size formulas against hand-computed textbook values."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from pricing_lab import power

Z_975 = NormalDist().inv_cdf(0.975)  # 1.95996...
Z_80 = NormalDist().inv_cdf(0.80)  # 0.84162...


def test_two_proportion_textbook_value() -> None:
    # p1 = 0.10, p2 = 0.12, alpha = 0.05 two-sided, power = 0.80.
    by_hand = math.ceil((Z_975 + Z_80) ** 2 * (0.10 * 0.90 + 0.12 * 0.88) / 0.02**2)
    n = power.n_per_arm_two_proportions(0.10, 0.02)
    assert n == by_hand
    assert abs(n - 3839) <= 2


def test_pooled_variance_is_slightly_more_conservative() -> None:
    unpooled = power.n_per_arm_two_proportions(0.10, 0.02)
    pooled = power.n_per_arm_two_proportions(0.10, 0.02, pooled=True)
    assert unpooled < pooled <= unpooled + 10


def test_continuous_textbook_value() -> None:
    # 0.2 standard-deviation effect: 2 * (1.96 + 0.84)^2 / 0.04 = 392.4 -> 393.
    assert power.n_per_arm_continuous(sd=1.0, mde=0.2) == 393


def test_unequal_allocation_needs_more_controls_when_treatment_is_small() -> None:
    equal = power.n_per_arm_continuous(1.0, 0.2, ratio=1.0)
    small_treatment = power.n_per_arm_continuous(1.0, 0.2, ratio=0.5)
    assert small_treatment > equal


def test_mde_round_trips_with_sample_size() -> None:
    n = power.n_per_arm_two_proportions(0.10, 0.02)
    assert power.mde_two_proportions(0.10, n) == pytest.approx(0.02, abs=1e-4)
    assert power.mde_continuous(1.0, 393) == pytest.approx(0.2, abs=1e-3)


def test_power_at_required_n_is_at_least_target() -> None:
    n = power.n_per_arm_two_proportions(0.10, 0.02)
    achieved = power.power_two_proportions(0.10, 0.12, n)
    assert 0.80 <= achieved < 0.81
    assert power.power_continuous(1.0, 0.2, 393) == pytest.approx(0.80, abs=0.01)


def test_one_sided_needs_fewer_units() -> None:
    assert power.n_per_arm_continuous(1.0, 0.2, two_sided=False) < 393


def test_duration_table_shrinks_with_root_n() -> None:
    table = power.mde_duration_table(
        daily_units=200, weeks=[1, 4], metric="continuous", baseline=100.0, sd=50.0
    )
    assert table["mde_abs"].is_monotonic_decreasing
    assert table["mde_abs"].iloc[0] / table["mde_abs"].iloc[1] == pytest.approx(2.0)
    assert table.loc[0, "control_n"] == 700
    prop = power.mde_duration_table(1000, [1, 2, 4], "proportion", baseline=0.3)
    assert prop["mde_abs"].is_monotonic_decreasing


def test_design_effect_inflates_mde() -> None:
    assert power.design_effect(10, 0.0) == 1.0
    assert power.design_effect(10, 0.1) == pytest.approx(1.9)
    plain = power.mde_duration_table(100, [2], "proportion", 0.3)
    clustered = power.mde_duration_table(100, [2], "proportion", 0.3, deff=1.9)
    assert clustered.loc[0, "mde_abs"] > plain.loc[0, "mde_abs"]


def test_rpv_moments() -> None:
    assert power.rpv_moments(1.0, 100.0, 20.0) == pytest.approx((100.0, 20.0))
    # Bernoulli(0.1) x 100: mean 10, sd 100 * sqrt(0.09) = 30.
    assert power.rpv_moments(0.1, 100.0, 0.0) == pytest.approx((10.0, 30.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline": 0.10, "mde": 0.0},
        {"baseline": 0.95, "mde": 0.10},
        {"baseline": 0.10, "mde": 0.02, "alpha": 0.0},
        {"baseline": 0.10, "mde": 0.02, "power": 1.0},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        power.n_per_arm_two_proportions(**kwargs)

"""Simulated price test on held-out weeks.

SIMULATED. Nothing in this module observes a real price change. It takes the
real bookings in held-out weeks as the untreated outcome, picks a random half
of the units (hotel x segment x channel) as treated, and from the first post
week applies a price change whose demand response is set by an assumed
elasticity. Control units and all pre-period data stay exactly as observed.

Because the true effect is known by construction, the analysis code in
``did.py`` and ``cuped.py`` can be checked against it.

Demand response for a price multiplier ``1 + c`` and elasticity ``e``:
expected bookings scale by ``m = (1 + c) ** e``. Counts are kept as integers:
for ``m <= 1`` each booking survives with probability ``m`` (binomial
thinning); for ``m > 1`` extra bookings are drawn from Poisson(bookings * (m - 1)).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_UNIT_COLS: tuple[str, ...] = ("hotel", "market_segment", "distribution_channel")


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for one simulated price test."""

    price_change: float = 0.05
    elasticity: float = -1.0
    pre_weeks: int = 13
    post_weeks: int = 13
    min_weekly_bookings: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.price_change <= -1:
            raise ValueError("price_change must be above -100%")
        if self.pre_weeks < 2 or self.post_weeks < 1:
            raise ValueError("need at least 2 pre weeks and 1 post week")
        if self.min_weekly_bookings < 1:
            raise ValueError("min_weekly_bookings must be at least 1 (outcomes are logged)")


@dataclass(frozen=True)
class TrueEffects:
    """Effects implied by the simulation settings (the answer key)."""

    bookings_log: float
    bookings_pct: float
    revenue_log: float
    revenue_pct: float


def true_effects(price_change: float, elasticity: float) -> TrueEffects:
    """Log and % effects on bookings and on revenue for a constant-elasticity demand.

    Bookings: e * log(1 + c). Revenue (price x bookings): (1 + e) * log(1 + c).
    """
    log_multiplier = math.log1p(price_change)
    bookings_log = elasticity * log_multiplier
    revenue_log = (1 + elasticity) * log_multiplier
    return TrueEffects(
        bookings_log=bookings_log,
        bookings_pct=math.expm1(bookings_log),
        revenue_log=revenue_log,
        revenue_pct=math.expm1(revenue_log),
    )


def holdout_cutoff(panel: pd.DataFrame, n_weeks: int, period_col: str = "period") -> pd.Timestamp:
    """First period of the last ``n_weeks`` periods.

    Use ``period < cutoff`` to estimate elasticity and ``period >= cutoff`` for
    the simulated test, so the two never share data.
    """
    periods = np.sort(panel[period_col].unique())
    if not 2 <= n_weeks < len(periods):
        raise ValueError(f"n_weeks must be between 2 and {len(periods) - 1}")
    return pd.Timestamp(periods[-n_weeks])


def build_unit_panel(
    holdout: pd.DataFrame,
    unit_cols: Sequence[str] = DEFAULT_UNIT_COLS,
    min_weekly_bookings: int = 5,
) -> pd.DataFrame:
    """Keep units observed in every period with at least ``min_weekly_bookings``.

    A balanced panel keeps the difference-in-differences estimate equal to a
    simple comparison of unit means, and a floor on volume keeps logs defined.
    """
    df = holdout.copy()
    df["unit"] = df[list(unit_cols)].astype(str).agg(" | ".join, axis=1)
    n_periods = df["period"].nunique()
    stats = df.groupby("unit").agg(n=("period", "nunique"), low=("bookings", "min"))
    keep = stats.index[(stats["n"] == n_periods) & (stats["low"] >= min_weekly_bookings)]
    return df.loc[df["unit"].isin(keep)].sort_values(["unit", "period"]).reset_index(drop=True)


def assign_matched_pairs(volume: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Matched-pair randomisation on pre-period volume.

    Units are sorted by volume and paired with their neighbour; one unit in
    each pair is treated at random. With few units this keeps the arms
    balanced on size. An odd unit out is assigned by coin flip.

    Args:
        volume: Pre-period mean bookings, indexed by unit.
        rng: Random generator.

    Returns:
        Columns ``unit``, ``pair_id``, ``treated``.
    """
    order = volume.sort_values(ascending=False).index.to_numpy()
    pair_id = np.arange(len(order)) // 2
    treated = np.zeros(len(order), dtype=int)
    for pair in np.unique(pair_id):
        idx = np.flatnonzero(pair_id == pair)
        if len(idx) == 2:
            treated[idx[rng.integers(2)]] = 1
        else:
            treated[idx[0]] = int(rng.integers(2))
    return pd.DataFrame({"unit": order, "pair_id": pair_id, "treated": treated})


def apply_price_change(
    bookings: np.ndarray, multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    """Scale integer booking counts by ``multiplier`` in expectation."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    counts = np.asarray(bookings, dtype=np.int64)
    if multiplier <= 1:
        return rng.binomial(counts, multiplier)
    return counts + rng.poisson(counts * (multiplier - 1))


def simulate_price_test(
    holdout: pd.DataFrame,
    config: SimulationConfig,
    unit_cols: Sequence[str] = DEFAULT_UNIT_COLS,
) -> pd.DataFrame:
    """Run one SIMULATED price test on the held-out panel.

    Args:
        holdout: Weekly panel (``aggregate_demand`` grouped by ``unit_cols``)
            covering exactly ``pre_weeks + post_weeks`` periods.
        config: Price change, assumed elasticity, window and seed.
        unit_cols: Columns that define a randomisation unit.

    Returns:
        Unit x week panel with ``treated``, ``post``, ``rel_week`` (0 = first
        post week), observed ``bookings`` and simulated ``bookings_sim``,
        ``adr_sim``, ``revenue_sim``, their logs and ``is_simulated = True``.
    """
    rng = np.random.default_rng(config.seed)
    panel = build_unit_panel(holdout, unit_cols, config.min_weekly_bookings)
    n_periods = panel["period"].nunique()
    if n_periods != config.pre_weeks + config.post_weeks:
        raise ValueError(
            f"holdout has {n_periods} usable periods; expected "
            f"{config.pre_weeks + config.post_weeks}"
        )
    if panel["unit"].nunique() < 4:
        raise ValueError("fewer than 4 units pass the volume filter; lower min_weekly_bookings")

    panel["rel_week"] = panel["period"].rank(method="dense").astype(int) - 1 - config.pre_weeks
    panel["post"] = (panel["rel_week"] >= 0).astype(int)

    pre_volume = panel.loc[panel["post"] == 0].groupby("unit")["bookings"].mean()
    panel = panel.merge(assign_matched_pairs(pre_volume, rng), on="unit", how="left")

    exposed = ((panel["treated"] == 1) & (panel["post"] == 1)).to_numpy()
    multiplier = (1 + config.price_change) ** config.elasticity
    observed = panel["bookings"].to_numpy(dtype=np.int64)
    simulated = observed.copy()
    simulated[exposed] = apply_price_change(observed[exposed], multiplier, rng)

    price_factor = np.where(exposed, 1 + config.price_change, 1.0)
    volume_ratio = np.divide(
        simulated, observed, out=np.zeros(len(observed), dtype=float), where=observed > 0
    )
    panel["bookings_sim"] = simulated
    panel["adr_sim"] = panel["adr"] * price_factor
    panel["revenue_sim"] = panel["revenue"] * volume_ratio * price_factor
    panel["log_bookings_sim"] = np.log(panel["bookings_sim"].where(panel["bookings_sim"] > 0))
    panel["log_revenue_sim"] = np.log(panel["revenue_sim"].where(panel["revenue_sim"] > 0))
    panel["is_simulated"] = True
    return panel

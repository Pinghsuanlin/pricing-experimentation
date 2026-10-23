"""SYNTHETIC booking data in the same schema as ``hotel_bookings.csv``.

Used only for the demo mode (``make analyse-demo`` and the app's demo toggle)
so the project runs before the real data is downloaded. Every number produced
from it is synthetic and says nothing about real hotels.

Price is exogenous here by construction (random shocks, not set in response
to demand), so a log-log regression should recover ``elasticity``. That is the
opposite of the real data, where price responds to demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_lab.data import MONTHS

# (segment, channel, weekly net bookings at the base price, base log price)
UNITS: tuple[tuple[str, str, float, float], ...] = (
    ("Online TA", "TA/TO", 60.0, 4.6),
    ("Offline TA/TO", "TA/TO", 25.0, 4.4),
    ("Groups", "TA/TO", 15.0, 4.3),
    ("Groups", "Direct", 10.0, 4.3),
    ("Direct", "Direct", 20.0, 4.7),
    ("Corporate", "Corporate", 12.0, 4.4),
)
# (hotel, log price offset, volume multiplier, cancellation rate)
HOTELS: tuple[tuple[str, float, float, float], ...] = (
    ("City Hotel", 0.0, 1.0, 0.35),
    ("Resort Hotel", 0.1, 0.7, 0.25),
)
MONTH_NAMES: dict[int, str] = {number: name for name, number in MONTHS.items()}


def _cells(
    n_weeks: int, elasticity: float, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """Expected weekly demand per hotel x segment x channel x week."""
    rows = []
    for w in range(n_weeks):
        week_start = start + pd.Timedelta(weeks=w)
        season = np.sin(2 * np.pi * week_start.isocalendar().week / 52)
        for hotel, offset, vol_mult, cancel_rate in HOTELS:
            segment_prices: dict[str, float] = {}
            for segment, channel, base_volume, base_price in UNITS:
                if segment not in segment_prices:
                    shock = rng.normal(0, 0.12)
                    segment_prices[segment] = base_price + offset + 0.15 * season + shock
                log_price = segment_prices[segment]
                log_rate = (
                    np.log(base_volume * vol_mult)
                    + elasticity * (log_price - base_price - offset)
                    + 0.35 * season
                    + rng.normal(0, 0.05)
                )
                rows.append(
                    (week_start, hotel, segment, channel, log_price, np.exp(log_rate), cancel_rate)
                )
    return pd.DataFrame(
        rows,
        columns=["week_start", "hotel", "segment", "channel", "log_price", "rate", "cancel"],
    )


def make_synthetic_bookings(
    n_weeks: int = 110,
    elasticity: float = -1.2,
    seed: int = 0,
    start: str = "2015-07-06",
) -> pd.DataFrame:
    """Generate booking-level synthetic data with a known price elasticity.

    Args:
        n_weeks: Number of arrival weeks (weeks start on Monday).
        elasticity: True constant elasticity of net bookings to price.
        seed: Random seed.
        start: First Monday.

    Returns:
        A frame with the columns ``data.REQUIRED_COLUMNS`` expects.
    """
    rng = np.random.default_rng(seed)
    cells = _cells(n_weeks, elasticity, pd.Timestamp(start), rng)
    net = rng.poisson(cells["rate"].to_numpy())
    cancelled = rng.poisson((cells["rate"] * cells["cancel"] / (1 - cells["cancel"])).to_numpy())

    idx = np.concatenate([np.repeat(cells.index, net), np.repeat(cells.index, cancelled)])
    is_canceled = np.concatenate([np.zeros(net.sum(), int), np.ones(cancelled.sum(), int)])
    rows = cells.loc[idx].reset_index(drop=True)
    n = len(rows)

    arrival = rows["week_start"] + pd.to_timedelta(rng.integers(0, 7, n), unit="D")
    return pd.DataFrame(
        {
            "hotel": rows["hotel"],
            "is_canceled": is_canceled,
            "lead_time": rng.geometric(1 / 60, n) - 1,
            "arrival_date_year": arrival.dt.year,
            "arrival_date_month": arrival.dt.month.map(MONTH_NAMES),
            "arrival_date_week_number": arrival.dt.isocalendar().week.astype(int).to_numpy(),
            "arrival_date_day_of_month": arrival.dt.day,
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(1, 5, n),
            "adults": rng.choice([1, 2, 3], n, p=[0.2, 0.7, 0.1]),
            "children": 0.0,
            "babies": 0,
            "market_segment": rows["segment"],
            "distribution_channel": rows["channel"],
            "customer_type": "Transient",
            "adr": np.round(np.exp(rows["log_price"] + rng.normal(0, 0.08, n)), 2),
        }
    )

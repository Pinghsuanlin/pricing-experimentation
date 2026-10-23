"""Cleaning and aggregation on a tiny hand-built bookings frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pricing_lab.data import (
    add_derived_columns,
    aggregate_demand,
    clean_bookings,
    load_bookings,
)


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hotel": "City Hotel",
        "is_canceled": 0,
        "lead_time": 10,
        "arrival_date_year": 2016,
        "arrival_date_month": "March",
        "arrival_date_week_number": 10,
        "arrival_date_day_of_month": 7,  # Monday 7 March 2016
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "customer_type": "Transient",
        "adr": 100.0,
    }
    base.update(overrides)
    return base


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(adr=100.0, lead_time=10),
            _row(adr=120.0, lead_time=30, arrival_date_day_of_month=9),
            _row(adr=80.0, is_canceled=1),
            _row(adr=0.0),  # dropped: non-positive adr
            _row(adr=6000.0),  # dropped: above cap
            _row(adults=0, children=float("nan")),  # dropped: no guests
            _row(stays_in_weekend_nights=0, stays_in_week_nights=0),  # dropped: zero nights
            _row(market_segment="Complementary"),  # dropped: excluded segment
            _row(arrival_date_day_of_month=13),  # Sunday, same week; makes it a full week
            _row(hotel="Resort Hotel", adr=200.0),
        ]
    )


def test_derived_columns() -> None:
    out = add_derived_columns(_toy())
    assert out.loc[0, "arrival_date"] == pd.Timestamp("2016-03-07")
    assert (out["week_start"] == pd.Timestamp("2016-03-07")).all()
    assert out.loc[0, "nights"] == 3
    assert out.loc[5, "guests"] == 0


def test_cleaning_report_counts_each_rule() -> None:
    clean, report = clean_bookings(add_derived_columns(_toy()))
    assert report.rows_in == 10
    assert report.excluded_segments == 1
    assert report.non_positive_adr == 1
    assert report.adr_above_cap == 1
    assert report.no_guests == 1
    assert report.zero_nights == 1
    assert report.rows_out == len(clean) == 5


def test_net_demand_excludes_cancellations() -> None:
    clean, _ = clean_bookings(add_derived_columns(_toy()))
    weekly = aggregate_demand(clean, demand="net")
    city = weekly.loc[weekly["hotel"] == "City Hotel"].iloc[0]
    assert city["gross_bookings"] == 4
    assert city["cancellations"] == 1
    assert city["bookings"] == 3
    assert city["adr"] == pytest.approx((100 + 120 + 100) / 3)
    assert city["cancel_rate"] == pytest.approx(0.25)
    assert city["revenue"] == pytest.approx((100 + 120 + 100) * 3)
    assert city["week_of_year"] == 10


def test_gross_demand_counts_all_bookings() -> None:
    clean, _ = clean_bookings(add_derived_columns(_toy()))
    weekly = aggregate_demand(clean, demand="gross")
    city = weekly.loc[weekly["hotel"] == "City Hotel"].iloc[0]
    assert city["bookings"] == 4
    assert city["adr"] == pytest.approx((100 + 120 + 80 + 100) / 4)


def test_partial_weeks_are_dropped() -> None:
    toy = _toy()
    toy = toy.loc[toy["arrival_date_day_of_month"] != 13]  # data now ends on Wednesday
    clean, _ = clean_bookings(add_derived_columns(toy))
    assert aggregate_demand(clean).empty
    assert not aggregate_demand(clean, drop_partial_weeks=False).empty


def test_daily_frequency() -> None:
    clean, _ = clean_bookings(add_derived_columns(_toy()))
    daily = aggregate_demand(clean, freq="D", group_cols=("hotel",))
    assert set(daily["period"].dt.day) == {7, 9, 13}


def test_bad_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bookings(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="Missing required columns"):
        add_derived_columns(_toy().drop(columns=["adr"]))
    with pytest.raises(ValueError, match="Unrecognised month"):
        add_derived_columns(_toy().assign(arrival_date_month="Marchh"))
    with pytest.raises(ValueError):
        aggregate_demand(_toy(), demand="booked")  # type: ignore[arg-type]

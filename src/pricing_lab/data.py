"""Load, clean and aggregate the hotel booking demand data.

The raw file has one row per booking. This module turns it into a panel of
demand (bookings) and price (average daily rate, ``adr``) per group and period.

Cancellations are handled explicitly through the ``demand`` argument:

* ``"net"``: only bookings that were not cancelled (realised stays).
* ``"gross"``: every booking made, cancelled or not (booking intent).

Periods are keyed on the arrival date, not the booking date. ``adr`` is the
average nightly rate of the stay, so it is the price the guest agreed to pay.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Demand = Literal["net", "gross"]
Frequency = Literal["D", "W"]

REQUIRED_COLUMNS: tuple[str, ...] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "adr",
)

MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}

# "Complementary" stays are free (adr = 0) and "Undefined" has almost no rows.
DEFAULT_EXCLUDED_SEGMENTS: tuple[str, ...] = ("Complementary", "Undefined")
DEFAULT_MAX_ADR: float = 1000.0


@dataclass(frozen=True)
class CleaningReport:
    """Rows removed at each cleaning step, in the order the steps are applied."""

    rows_in: int
    excluded_segments: int
    non_positive_adr: int
    adr_above_cap: int
    no_guests: int
    zero_nights: int
    rows_out: int


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any column the pipeline needs is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_bookings(path: str | Path) -> pd.DataFrame:
    """Read the raw bookings CSV and check its columns.

    Args:
        path: Location of ``hotel_bookings.csv``.

    Returns:
        The raw booking-level frame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download steps.")
    df = pd.read_csv(path)
    validate_columns(df)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``arrival_date``, ``week_start`` (Monday), ``nights`` and ``guests``."""
    validate_columns(df)
    month = df["arrival_date_month"].map(MONTHS)
    if month.isna().any():
        bad = sorted(df.loc[month.isna(), "arrival_date_month"].astype(str).unique())
        raise ValueError(f"Unrecognised month names: {bad}")

    out = df.copy()
    out["arrival_date"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": out["arrival_date_year"],
                "month": month.astype(int),
                "day": out["arrival_date_day_of_month"],
            }
        )
    )
    weekday = pd.to_timedelta(out["arrival_date"].dt.weekday, unit="D")
    out["week_start"] = out["arrival_date"] - weekday
    out["nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["guests"] = out["adults"] + out["children"].fillna(0) + out["babies"]
    return out


def clean_bookings(
    df: pd.DataFrame,
    max_adr: float = DEFAULT_MAX_ADR,
    excluded_segments: Sequence[str] = DEFAULT_EXCLUDED_SEGMENTS,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove rows that are not valid priced stays.

    Steps, applied in order: excluded segments, ``adr <= 0``, ``adr > max_adr``,
    zero guests, zero nights. Cancelled bookings are kept; ``aggregate_demand``
    decides how to treat them.

    Returns:
        The cleaned frame and a ``CleaningReport`` with rows dropped per step.
    """
    work = df if "nights" in df.columns else add_derived_columns(df)
    rows_in = len(work)
    rules: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "excluded_segments": lambda f: f["market_segment"].isin(list(excluded_segments)),
        "non_positive_adr": lambda f: f["adr"] <= 0,
        "adr_above_cap": lambda f: f["adr"] > max_adr,
        "no_guests": lambda f: f["guests"] <= 0,
        "zero_nights": lambda f: f["nights"] <= 0,
    }
    dropped: dict[str, int] = {}
    for name, rule in rules.items():
        mask = rule(work)
        dropped[name] = int(mask.sum())
        work = work.loc[~mask]
    report = CleaningReport(rows_in=rows_in, rows_out=len(work), **dropped)
    return work.reset_index(drop=True), report


def prepare_bookings(path: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
    """Load, derive and clean in one call."""
    return clean_bookings(add_derived_columns(load_bookings(path)))


def add_calendar_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ISO ``year``, ISO ``week_of_year`` and ``month`` from ``period``."""
    out = panel.copy()
    iso = out["period"].dt.isocalendar()
    out["year"] = iso["year"].astype(int)
    out["week_of_year"] = iso["week"].astype(int)
    out["month"] = out["period"].dt.month.astype(int)
    return out


def aggregate_demand(
    df: pd.DataFrame,
    freq: Frequency = "W",
    group_cols: Sequence[str] = ("hotel", "market_segment"),
    demand: Demand = "net",
    drop_partial_weeks: bool = True,
) -> pd.DataFrame:
    """Aggregate bookings into a demand and price panel.

    Args:
        df: Cleaned booking-level frame (output of ``clean_bookings``).
        freq: ``"W"`` for weeks starting Monday, ``"D"`` for days (arrival date).
        group_cols: Columns that define one demand series.
        demand: ``"net"`` counts non-cancelled bookings, ``"gross"`` counts all.
        drop_partial_weeks: Drop the first and last week if the data does not
            cover all seven days. Partial weeks look like demand drops.

    Returns:
        One row per group and period with ``bookings`` (per ``demand``),
        ``gross_bookings``, ``cancellations``, ``cancel_rate``, ``adr`` (mean
        nightly rate of the counted bookings), ``lead_time`` (mean days),
        ``room_nights``, ``revenue`` (adr x nights) and calendar columns.
    """
    if demand not in ("net", "gross"):
        raise ValueError("demand must be 'net' or 'gross'")
    if freq not in ("D", "W"):
        raise ValueError("freq must be 'D' or 'W'")

    work = df if "nights" in df.columns else add_derived_columns(df)
    period_source = "week_start" if freq == "W" else "arrival_date"
    work = work.assign(period=work[period_source], revenue=work["adr"] * work["nights"])
    keys = [*group_cols, "period"]

    totals = work.groupby(keys).agg(
        gross_bookings=("is_canceled", "size"),
        cancellations=("is_canceled", "sum"),
    )
    counted = work if demand == "gross" else work.loc[work["is_canceled"] == 0]
    stats = counted.groupby(keys).agg(
        bookings=("adr", "size"),
        adr=("adr", "mean"),
        lead_time=("lead_time", "mean"),
        room_nights=("nights", "sum"),
        revenue=("revenue", "sum"),
    )
    out = totals.join(stats, how="left").reset_index()
    out["bookings"] = out["bookings"].fillna(0).astype(int)
    out["room_nights"] = out["room_nights"].fillna(0).astype(int)
    out["revenue"] = out["revenue"].fillna(0.0)
    out["cancel_rate"] = out["cancellations"] / out["gross_bookings"]

    if freq == "W" and drop_partial_weeks:
        first, last = work["arrival_date"].min(), work["arrival_date"].max()
        full_week = (out["period"] >= first) & (out["period"] + pd.Timedelta(days=6) <= last)
        out = out.loc[full_week]

    return add_calendar_columns(out).sort_values(keys).reset_index(drop=True)

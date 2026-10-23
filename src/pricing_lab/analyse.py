"""Run the full analysis and write ``results.json``, ``results.md`` and the simulated panel.

    python -m pricing_lab.analyse --data data/raw/hotel_bookings.csv --out reports
    python -m pricing_lab.analyse --demo --out reports/demo    # SYNTHETIC data

Pipeline: clean -> weekly panel -> split training weeks / held-out weeks ->
elasticity on training weeks -> power tables -> SIMULATED price test on the
held-out weeks -> DiD, pre-trend, permutation and CUPED readouts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pricing_lab import cuped, did, elasticity, power, simulate
from pricing_lab.data import (
    CleaningReport,
    Demand,
    add_derived_columns,
    aggregate_demand,
    clean_bookings,
    load_bookings,
)
from pricing_lab.synthetic import make_synthetic_bookings

FALLBACK_ELASTICITY = -1.0
MONTH_FIXED_EFFECTS: tuple[str, ...] = ("month", "year", "hotel", "market_segment")
DURATION_WEEKS: tuple[int, ...] = (1, 2, 4, 6, 8)
INSTRUMENT = "z_other_hotel_price"


def load_clean(data_path: Path | None, demo: bool) -> tuple[pd.DataFrame, CleaningReport, str]:
    """Load the real CSV (or synthetic data in demo mode) and clean it."""
    if demo:
        raw, source = make_synthetic_bookings(), "SYNTHETIC demo data (not real hotels)"
    else:
        if data_path is None:
            raise ValueError("--data is required unless --demo is set")
        raw, source = load_bookings(data_path), str(data_path)
    clean, report = clean_bookings(add_derived_columns(raw))
    return clean, report, source


def _training_frame(clean: pd.DataFrame, cutoff: pd.Timestamp, demand: Demand) -> pd.DataFrame:
    """Model frame built from training weeks only."""
    panel = aggregate_demand(clean, demand=demand)
    return elasticity.build_model_frame(panel.loc[panel["period"] < cutoff])


def elasticity_section(clean: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Main estimate, robustness variants, per-segment fits and the 2SLS check."""
    net = _training_frame(clean, cutoff, "net")
    gross = _training_frame(clean, cutoff, "gross")
    fit = elasticity.fit_log_log
    out: dict[str, Any] = {
        "main_robust": asdict(fit(net)),
        "cluster_by_week": asdict(fit(net, cov_type="cluster", cluster_col="period")),
        "month_fixed_effects": asdict(fit(net, fixed_effects=MONTH_FIXED_EFFECTS)),
        "no_lead_time_control": asdict(fit(net, include_lead_time=False)),
        "gross_demand": asdict(fit(gross)),
        "per_segment": elasticity.per_segment_elasticity(net).to_dict(orient="records"),
    }
    try:
        iv_frame = elasticity.add_hausman_instrument(net, name=INSTRUMENT)
        out["iv_2sls"] = asdict(elasticity.fit_2sls(iv_frame, INSTRUMENT))
    except ValueError as err:
        out["iv_2sls"] = {"error": str(err)}
    return out


def power_section(clean: pd.DataFrame) -> dict[str, Any]:
    """MDE vs duration tables using the dataset's own volumes and variances.

    Illustrative: the data has no visitor or quote traffic, so the unit is a
    booking at the observed daily volume.
    """
    days = (clean["arrival_date"].max() - clean["arrival_date"].min()).days + 1
    stays = clean.loc[clean["is_canceled"] == 0]
    revenue_per_stay = stays["adr"] * stays["nights"]
    inputs = {
        "daily_bookings": len(clean) / days,
        "daily_stays": len(stays) / days,
        "cancellation_rate": float(clean["is_canceled"].mean()),
        "revenue_per_stay_mean": float(revenue_per_stay.mean()),
        "revenue_per_stay_sd": float(revenue_per_stay.std()),
    }
    cancel_table = power.mde_duration_table(
        inputs["daily_bookings"], DURATION_WEEKS, "proportion", inputs["cancellation_rate"]
    )
    revenue_table = power.mde_duration_table(
        inputs["daily_stays"],
        DURATION_WEEKS,
        "continuous",
        inputs["revenue_per_stay_mean"],
        sd=inputs["revenue_per_stay_sd"],
    )
    return {
        "inputs": inputs,
        "cancellation_rate_mde": cancel_table.to_dict(orient="records"),
        "revenue_per_stay_mde": revenue_table.to_dict(orient="records"),
        "textbook_n_per_arm_p10_mde2": power.n_per_arm_two_proportions(0.10, 0.02),
    }


def choose_elasticity(ols_estimate: float, override: float | None) -> tuple[float, str]:
    """Elasticity to simulate with, and where it came from."""
    if override is not None:
        return override, "set by the user (--elasticity)"
    if ols_estimate < 0:
        return ols_estimate, "OLS estimate on training weeks"
    return (
        FALLBACK_ELASTICITY,
        f"assumed {FALLBACK_ELASTICITY}: the OLS estimate was non-negative, "
        "which points to price endogeneity",
    )


def simulation_section(
    clean: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: simulate.SimulationConfig,
    source: str,
    unit_cols: Sequence[str] = simulate.DEFAULT_UNIT_COLS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """SIMULATED price test on held-out weeks, with every readout."""
    units = aggregate_demand(clean, group_cols=unit_cols, demand="net")
    sim = simulate.simulate_price_test(units.loc[units["period"] >= cutoff], config, unit_cols)
    means = did.unit_pre_post_means(sim)
    result = {
        "label": "SIMULATED: real held-out bookings plus an injected price effect",
        "elasticity_source": source,
        "config": asdict(config),
        "unit_cols": list(unit_cols),
        "true_effects": asdict(simulate.true_effects(config.price_change, config.elasticity)),
        "n_units": int(sim["unit"].nunique()),
        "n_treated_units": int(sim.groupby("unit")["treated"].first().sum()),
        "did_twfe_bookings": asdict(did.fit_did(sim)),
        "did_2x2_bookings": asdict(did.fit_did(sim, twfe=False)),
        "did_twfe_revenue": asdict(did.fit_did(sim, outcome="log_revenue_sim")),
        "pre_trend": asdict(did.pre_trend_test(sim)),
        "permutation": asdict(did.permutation_test(sim)),
        "cuped": asdict(cuped.cuped_ate(means["post"], means["pre"], means["treated"])),
        "event_study": did.event_study(sim).to_dict(orient="records"),
    }
    return result, sim


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _ci(row: dict[str, Any], low: str = "ci_low", high: str = "ci_high") -> str:
    return f"{_fmt(row[low])} to {_fmt(row[high])}"


def render_markdown(results: dict[str, Any]) -> str:
    """Human-readable summary of ``results`` (no numbers beyond what was computed)."""
    e, p, s = results["elasticity"], results["power"], results["simulation"]
    lines = [
        "# Results",
        "",
        f"Data: {results['source']}. Rows after cleaning: {results['cleaning']['rows_out']}.",
        f"Training weeks end before {results['holdout_cutoff']}; later weeks are held out.",
        "",
        "## Price elasticity (observational, training weeks)",
        "",
        "| Specification | Elasticity | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for key in (
        "main_robust",
        "cluster_by_week",
        "month_fixed_effects",
        "no_lead_time_control",
        "gross_demand",
    ):
        row = e[key]
        lines.append(f"| {key} | {_fmt(row['elasticity'])} | {_ci(row)} | {row['n_obs']} |")
    iv = e["iv_2sls"]
    if "error" in iv:
        lines.append(f"| iv_2sls | not estimated: {iv['error']} | | |")
    else:
        lines.append(
            f"| iv_2sls (first-stage F {_fmt(iv['first_stage_f'], 1)}) | "
            f"{_fmt(iv['elasticity'])} | {_ci(iv)} | {iv['n_obs']} |"
        )
    lines += ["", "| Segment | Elasticity | 95% CI | n |", "|---|---|---|---|"]
    for row in e["per_segment"]:
        lines.append(
            f"| {row['market_segment']} | {_fmt(row['elasticity'])} | {_ci(row)} | {row['n_obs']} |"
        )

    lines += [
        "",
        "## Power (illustrative, booking-level randomisation at observed volume)",
        "",
        f"Textbook check, p=0.10, MDE=0.02: {p['textbook_n_per_arm_p10_mde2']} per arm.",
        "",
        "| Weeks | Cancellation-rate MDE (abs) | Revenue-per-stay MDE (rel) |",
        "|---|---|---|",
    ]
    for c_row, r_row in zip(p["cancellation_rate_mde"], p["revenue_per_stay_mde"], strict=True):
        lines.append(
            f"| {c_row['weeks']} | {_fmt(c_row['mde_abs'], 4)} | {_fmt(r_row['mde_rel'], 4)} |"
        )

    t, twfe = s["true_effects"], s["did_twfe_bookings"]
    lines += [
        "",
        "## SIMULATED price test (held-out weeks)",
        "",
        f"Elasticity used: {_fmt(s['config']['elasticity'])} ({s['elasticity_source']}). "
        f"Price change: {s['config']['price_change']:+.1%}. "
        f"Units: {s['n_units']} ({s['n_treated_units']} treated), "
        f"defined by {', '.join(s['unit_cols'])}.",
        "",
        "| Quantity | Value | 95% CI |",
        "|---|---|---|",
        f"| True bookings effect (log points) | {_fmt(t['bookings_log'])} | known by design |",
        f"| DiD TWFE bookings | {_fmt(twfe['effect'])} | {_ci(twfe)} |",
        f"| DiD 2x2 bookings | {_fmt(s['did_2x2_bookings']['effect'])} | "
        f"{_ci(s['did_2x2_bookings'])} |",
        f"| True revenue effect (log points) | {_fmt(t['revenue_log'])} | known by design |",
        f"| DiD TWFE revenue | {_fmt(s['did_twfe_revenue']['effect'])} | "
        f"{_ci(s['did_twfe_revenue'])} |",
        f"| Pre-trend slope (p-value) | {_fmt(s['pre_trend']['slope'], 4)} "
        f"(p={_fmt(s['pre_trend']['p_value'])}) | |",
        f"| Permutation p-value | {_fmt(s['permutation']['p_value'])} | |",
        f"| CUPED raw effect (se) | {_fmt(s['cuped']['raw_effect'])} "
        f"({_fmt(s['cuped']['raw_std_error'])}) | {_ci(s['cuped'], 'raw_ci_low', 'raw_ci_high')} |",
        f"| CUPED adjusted effect (se) | {_fmt(s['cuped']['adjusted_effect'])} "
        f"({_fmt(s['cuped']['adjusted_std_error'])}) | "
        f"{_ci(s['cuped'], 'adjusted_ci_low', 'adjusted_ci_high')} |",
        f"| CUPED variance reduction | {s['cuped']['variance_reduction']:.1%} | |",
        "",
    ]
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def run_analysis(
    data_path: Path | None,
    out_dir: Path,
    demo: bool = False,
    holdout_weeks: int = 26,
    price_change: float = 0.05,
    elasticity_override: float | None = None,
    min_weekly_bookings: int = 5,
    seed: int = 42,
    unit_cols: Sequence[str] = simulate.DEFAULT_UNIT_COLS,
) -> dict[str, Any]:
    """Run every step and write the outputs to ``out_dir``."""
    clean, report, source = load_clean(data_path, demo)
    hotel_segment = aggregate_demand(clean)
    cutoff = simulate.holdout_cutoff(hotel_segment, holdout_weeks)

    elasticity_results = elasticity_section(clean, cutoff)
    value, origin = choose_elasticity(
        elasticity_results["main_robust"]["elasticity"], elasticity_override
    )
    pre_weeks = holdout_weeks // 2
    config = simulate.SimulationConfig(
        price_change=price_change,
        elasticity=value,
        pre_weeks=pre_weeks,
        post_weeks=holdout_weeks - pre_weeks,
        min_weekly_bookings=min_weekly_bookings,
        seed=seed,
    )
    simulation_results, sim_panel = simulation_section(clean, cutoff, config, origin, unit_cols)

    results = {
        "source": source,
        "cleaning": asdict(report),
        "holdout_cutoff": str(cutoff.date()),
        "elasticity": elasticity_results,
        "power": power_section(clean),
        "simulation": simulation_results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    (out_dir / "results.md").write_text(render_markdown(results))
    sim_panel.to_csv(out_dir / "simulated_panel.csv", index=False)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None, help="path to hotel_bookings.csv")
    parser.add_argument("--demo", action="store_true", help="use SYNTHETIC data instead")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--holdout-weeks", type=int, default=26)
    parser.add_argument("--price-change", type=float, default=0.05)
    parser.add_argument("--elasticity", type=float, default=None, help="override for simulation")
    parser.add_argument("--min-weekly-bookings", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit-cols",
        nargs="+",
        default=list(simulate.DEFAULT_UNIT_COLS),
        help="columns that define a randomisation unit in the simulated test",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m pricing_lab.analyse``."""
    args = parse_args(argv)
    results = run_analysis(
        data_path=args.data,
        out_dir=args.out,
        demo=args.demo,
        holdout_weeks=args.holdout_weeks,
        price_change=args.price_change,
        elasticity_override=args.elasticity,
        min_weekly_bookings=args.min_weekly_bookings,
        seed=args.seed,
        unit_cols=args.unit_cols,
    )
    print(render_markdown(results))
    print(f"Wrote results to {args.out}/")


if __name__ == "__main__":
    main()

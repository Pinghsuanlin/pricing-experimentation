"""End to end on SYNTHETIC Kaggle-schema data: clean -> aggregate -> estimate -> report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pricing_lab.analyse import run_analysis
from pricing_lab.data import add_derived_columns, aggregate_demand, clean_bookings
from pricing_lab.elasticity import build_model_frame, fit_log_log
from pricing_lab.synthetic import make_synthetic_bookings


def test_pipeline_recovers_synthetic_elasticity() -> None:
    raw = make_synthetic_bookings(n_weeks=80, elasticity=-1.2, seed=3)
    clean, report = clean_bookings(add_derived_columns(raw))
    assert report.rows_out == len(clean) > 0
    frame = build_model_frame(aggregate_demand(clean, demand="net"))
    res = fit_log_log(frame)
    assert res.elasticity == pytest.approx(-1.2, abs=0.15)


def test_run_analysis_writes_outputs(tmp_path: Path) -> None:
    results = run_analysis(data_path=None, out_dir=tmp_path, demo=True, holdout_weeks=20)
    for name in ("results.json", "results.md", "simulated_panel.csv"):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["source"].startswith("SYNTHETIC")
    assert saved["simulation"]["label"].startswith("SIMULATED")
    assert results["simulation"]["n_units"] >= 4
    assert "SIMULATED price test" in (tmp_path / "results.md").read_text()

.PHONY: help setup setup-pip test lint app analyse analyse-demo clean

DATA ?= data/raw/hotel_bookings.csv
OUT ?= reports

help:  ## List targets
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed "s/:.*##/:/"

setup:  ## Create .venv and install with uv (runtime + dev deps)
	uv sync --extra dev

setup-pip:  ## Alternative without uv (run inside an activated virtualenv)
	pip install -e ".[dev]"

test:  ## Run the offline test suite
	uv run --extra dev pytest

lint:  ## Ruff lint
	uv run --extra dev ruff check .

app:  ## Launch the Streamlit app
	uv run streamlit run app.py

analyse:  ## Full analysis on the real CSV, writes $(OUT)/results.md and results.json
	uv run python -m pricing_lab.analyse --data $(DATA) --out $(OUT)

analyse-demo:  ## Same pipeline on SYNTHETIC data (no download needed)
	uv run python -m pricing_lab.analyse --demo --out $(OUT)/demo

clean:  ## Remove caches and generated reports
	rm -rf $(OUT) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Convenience targets (Linux/macOS/CI). On Windows without make, run the same commands with
# `uv run ...` directly, e.g. `uv run cocoasense all`.
.PHONY: install lint format typecheck test test-fast reproduce dashboard clean

install:
	uv sync --all-groups

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

test-fast:
	uv run pytest -m "not slow"

reproduce:
	uv run cocoasense all

dashboard:
	uv run cocoasense dashboard

clean:
	rm -rf data/interim data/processed artifacts/forecasts artifacts/metrics \
	       artifacts/simulation artifacts/figures

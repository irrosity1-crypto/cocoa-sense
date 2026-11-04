"""Artifact loaders and shared UI pieces for the dashboard.

The dashboard reads files written by the pipeline and never fits a model. Set the environment
variable ``COCOASENSE_ROOT`` to point it at another project directory (used by tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from cocoasense.config import PROJECT_ROOT, ProjectConfig, load_config

DISCLAIMER = (
    "**Simulated results on public prices.** These are not Ghana's or COCOBOD's actual gains "
    "or losses, and the forecasts are not trading advice. Models were trained only on data up "
    "to December 2019; everything from January 2020 is out-of-sample."
)

MODEL_LABELS = {
    "naive": "Naive (last price)",
    "arima": "ARIMA",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def root() -> Path:
    """Project root used by the dashboard."""
    return Path(os.environ.get("COCOASENSE_ROOT", PROJECT_ROOT))


@st.cache_resource
def config(root_dir: str) -> ProjectConfig:
    """Project configuration."""
    return load_config(Path(root_dir))


def cfg() -> ProjectConfig:
    """Configuration for the current root."""
    return config(str(root()))


@st.cache_data
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=[
            c for c in ("decision_date", "target_date") if c in pd.read_csv(path, nrows=0).columns
        ],
    )


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    reader = _parquet if path.suffix == ".parquet" else _csv
    return reader(str(path), path.stat().st_mtime)


def monthly() -> pd.DataFrame | None:
    """Validated monthly prices and exchange rate."""
    return _load(cfg().paths.interim / "monthly.parquet")


def test_predictions() -> pd.DataFrame | None:
    """Frozen-model forecasts for the test period (main models only)."""
    df = _load(cfg().paths.forecasts / "test_predictions.parquet")
    return None if df is None else df[df["experiment"] == "main"]


def validation_predictions() -> pd.DataFrame | None:
    """Development-stage forecasts for the validation period."""
    return _load(cfg().paths.forecasts / "validation_predictions.parquet")


def metrics(name: str) -> pd.DataFrame | None:
    """A CSV table from ``artifacts/metrics``."""
    return _load(cfg().paths.metrics / name)


def simulation_table(name: str) -> pd.DataFrame | None:
    """A table from ``artifacts/simulation``."""
    return _load(cfg().paths.simulation / name)


def frozen() -> dict[str, Any] | None:
    """The frozen configuration, if written."""
    path = cfg().paths.frozen
    if not path.exists():
        return None
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def manifest() -> pd.DataFrame | None:
    """Raw-file provenance as a table."""
    path = cfg().paths.manifest
    if not path.exists():
        return None
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    return pd.DataFrame(files.values())


def text(path: Path) -> str | None:
    """Contents of a text file, or None."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def page_header(title: str) -> None:
    """Page title plus the permanent disclaimer banner."""
    st.set_page_config(page_title=f"CocoaSense AI · {title}", layout="wide")
    st.title(title)
    st.warning(DISCLAIMER, icon="⚠️")


def missing(what: str, command: str) -> None:
    """Friendly message instead of an error when artifacts are missing."""
    st.info(
        f"{what} not found. Run the pipeline first:\n\n```\nuv run cocoasense {command}\n```\n\n"
        "or `uv run cocoasense all` to reproduce everything from the raw data."
    )


def label(model: str) -> str:
    """Human-readable model name."""
    return MODEL_LABELS.get(model, model)

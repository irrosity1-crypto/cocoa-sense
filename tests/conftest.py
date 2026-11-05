"""Shared fixtures.

All data created here is SYNTHETIC (seeded random walks) and exists only to test code paths.
No synthetic number is ever used in a result, report or the dashboard.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cocoasense.config import PROJECT_ROOT, ProjectConfig, load_config

REPO_CONFIGS = PROJECT_ROOT / "configs"


def synthetic_history(start: str = "1975-01", end: str = "2019-12", seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC monthly price (USD/t) and exchange rate (GHS/USD, from 1990)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    n = len(idx)
    r = 0.004 + 0.06 * rng.standard_normal(n)
    r[1:] += 0.2 * r[:-1]  # mild autocorrelation so models have something to learn
    price = 1500 * np.exp(np.cumsum(r))
    fx = 0.03 * np.exp(np.cumsum(0.012 + 0.02 * rng.standard_normal(n)))
    df = pd.DataFrame({"price_usd_t": price, "ghs_per_usd": fx}, index=idx)
    df.loc[df.index < "1990-01-01", "ghs_per_usd"] = np.nan
    df.index.name = "date"
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df


@pytest.fixture
def history() -> pd.DataFrame:
    """SYNTHETIC history used by unit and leakage tests."""
    return synthetic_history()


def write_pinksheet(path: Path, price: pd.Series) -> None:
    """Write a SYNTHETIC workbook with the Pink Sheet layout (cocoa in $/kg)."""
    rows: list[list[object]] = [
        ["World Bank Commodity Price Data (The Pink Sheet)", None, None],
        ["monthly prices in nominal US dollars, 1960 to present", None, None],
        ["(synthetic test fixture)", None, None],
        ["Updated on January 01, 2000", None, None],
        [None, "Crude oil, average", "Cocoa"],
        [None, "($/bbl)", "($/kg)"],
    ]
    for d, v in price.items():
        rows.append([f"{d:%Y}M{d:%m}", 1.0, round(float(v) / 1000, 6)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Monthly Prices", header=False, index=False)


def write_fred(path: Path, price: pd.Series) -> None:
    """Write a SYNTHETIC FRED-style CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"observation_date": price.index.strftime("%Y-%m-%d"), "PCOCOUSDM": price.to_numpy()}
    )
    df.to_csv(path, index=False)


def write_imf(path: Path, fx: pd.Series, country: str = "GHA") -> None:
    """Write a SYNTHETIC IMF SDMX structure-specific XML file."""
    obs = "".join(
        f'<Obs TIME_PERIOD="{d:%Y}-M{d:%m}" OBS_VALUE="{v}"/>' for d, v in fx.dropna().items()
    )
    xml = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        '<message:StructureSpecificData xmlns:message="urn:test:message">'
        "<message:DataSet>"
        f'<Series COUNTRY="{country}" INDICATOR="XDC_USD" FREQUENCY="M">{obs}</Series>'
        "</message:DataSet></message:StructureSpecificData>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


SMALL_OVERRIDES = {
    "models.yaml": {
        "cv_splits": 3,
        "models": {
            "naive": {},
            "arima": {"max_p": 1, "max_q": 1, "d": 1, "trend": "n"},
            "random_forest": {
                "grid": {"n_estimators": [20], "max_depth": [3], "min_samples_leaf": [5]}
            },
            "xgboost": {"grid": {"n_estimators": [20], "max_depth": [2], "learning_rate": [0.1]}},
        },
    },
    "evaluation.yaml": {
        "train_start": "1985-01",
        "train_end": "2009-12",
        "validation_start": "2010-01",
        "validation_end": "2013-12",
        "final_train_end": "2013-12",
        "test_start": "2014-01",
        "test_end": None,
        "periods": {
            "pre_surge": ["2014-01", "2016-12"],
            "surge": ["2017-01", "2018-06"],
            "post_peak": ["2018-07", None],
        },
        "surge_start": "2017-01",
        "train_start_candidates": ["1982-01", "1991-01"],
    },
    "simulation.yaml": {"a_grid": [0.0, 0.5, 1.0]},
    "insights.yaml": {
        "min_past_errors": 6,
        "learning": {"first_target": "2013-01", "models": ["naive", "arima", "xgboost"]},
    },
    "lstm.yaml": {
        "input_sets": [["ret_1m"], ["ret_1m", "log_price"]],
        "lookback": [6],
        "hidden_size": [4],
        "epochs": 15,
        "n_seeds": 1,
    },
}


@pytest.fixture
def project(tmp_path: Path) -> ProjectConfig:
    """A temporary project root with repo configs, small grids and a SYNTHETIC timeline."""
    cfg_dir = tmp_path / "configs"
    shutil.copytree(REPO_CONFIGS, cfg_dir, ignore=shutil.ignore_patterns("frozen*.yaml"))
    for name, override in SMALL_OVERRIDES.items():
        path = cfg_dir / name
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        content.update(override)
        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def raw_fixture_files(project: ProjectConfig) -> ProjectConfig:
    """Put SYNTHETIC raw files where `ingest` expects them (as manual drop-ins)."""
    h = synthetic_history(end="2019-12")
    srcs = project.data.sources
    write_pinksheet(
        project.paths.manual / Path(srcs["worldbank_pinksheet"].file).name, h["price_usd_t"]
    )
    write_fred(
        project.paths.manual / Path(srcs["fred_pcocousdm"].file).name, h["price_usd_t"].loc["1992":]
    )
    write_imf(project.paths.manual / Path(srcs["imf_er_ghs_usd"].file).name, h["ghs_per_usd"])
    return project

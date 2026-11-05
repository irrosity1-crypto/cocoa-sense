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
    "lstm": "LSTM",
}


# One colour per model / strategy, used by every chart (labels are the displayed names).
MODEL_COLOURS = {
    "Naive (last price)": "#9aa4b2",
    "ARIMA": "#4c8dff",
    "Random Forest": "#2dd4bf",
    "XGBoost": "#a78bfa",
    "LSTM": "#fbbf24",
}
STRATEGY_COLOURS = {
    "Sell forward (w = 1)": "#4c8dff",
    "Wait (w = 0)": "#fbbf24",
    "Fixed 50/50": "#9aa4b2",
    "Forecast-driven": "#a78bfa",
}
PERIOD_LABELS = {
    "full": "Full test period",
    "pre_surge": "Pre-surge (2020–2022)",
    "surge": "Surge (2023 – Jan 2025)",
    "post_peak": "After the peak (Feb 2025 on)",
}


def dashes(df: pd.DataFrame, columns: list[str], fmt: str) -> pd.DataFrame:
    """Format numeric columns as text, showing missing values as an en dash.

    ``st.dataframe`` renders NaN as "None", which reads like an error in a screenshot.
    """
    out = df.copy()
    for c in columns:
        if c in out:
            out[c] = out[c].map(lambda v: "–" if pd.isna(v) else format(v, fmt))
    return out


STRATEGY_LABELS = {
    "sell_forward": "Sell forward (w = 1)",
    "wait": "Wait (w = 0)",
    "fixed_50_50": "Fixed 50/50",
    "forecast_driven": "Forecast-driven",
}


def strategy_label(name: str) -> str:
    """Human-readable strategy name."""
    return STRATEGY_LABELS.get(name, name)


def strategy_display(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Readable strategy names; blank the 50/50 strategy's win rate against itself."""
    out = metrics_df.copy()
    if "win_rate_vs_5050" in out:
        rate = out["win_rate_vs_5050"].map(lambda v: f"{v:.0%}")
        rate[out["strategy"] == "fixed_50_50"] = "–"
        out["win_rate_vs_5050"] = rate  # text, because st.dataframe shows NaN as "None"
    out["strategy"] = out["strategy"].map(strategy_label)
    return out


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
    """Frozen-model forecasts for the test period (main models, plus the LSTM if run)."""
    df = _load(cfg().paths.forecasts / "test_predictions.parquet")
    if df is None:
        return None
    df = df[df["experiment"] == "main"]
    lstm = _load(cfg().paths.forecasts / "lstm_test_predictions.parquet")
    return df if lstm is None else pd.concat([df, lstm], ignore_index=True)


def test_metrics() -> pd.DataFrame | None:
    """Test-period metrics, including the LSTM when its final run exists."""
    with_lstm = metrics("test_metrics_with_lstm.csv")
    return with_lstm if with_lstm is not None else metrics("test_metrics.csv")


TRACK_LABELS = {
    "frozen": "Frozen (trained to 2019, never updated)",
    "learning": "Learning (refitted every month)",
}
CONFIDENCE_COLOURS = {
    "High": "#16c784",
    "Medium": "#fbbf24",
    "Low": "#ea3943",
    "Not enough history": "#8a93a3",
}


def fx_daily() -> pd.DataFrame:
    """Stored daily US$ to cedi rates from the live API (empty if never fetched)."""
    from cocoasense.ingestion.fx_live import read_daily

    path = cfg().paths.artifacts / "fx" / "usd_ghs_daily.csv"
    return read_daily(path)


def fx_latest() -> tuple[pd.Timestamp, float, str] | None:
    """Latest known rate: live API if fetched, else the last IMF monthly average."""
    from cocoasense.ingestion.fx_live import latest

    live = latest(fx_daily())
    if live is not None:
        return live
    m = monthly()
    if m is None or m["ghs_per_usd"].dropna().empty:
        return None
    s = m["ghs_per_usd"].dropna()
    return pd.Timestamp(s.index[-1]), float(s.iloc[-1]), "IMF monthly average"


def fx_monthly() -> pd.Series:
    """One GHS-per-US$ rate per month: IMF where available, then live-API monthly averages."""
    from cocoasense.ingestion.fx_live import combined_monthly

    m = monthly()
    imf = m["ghs_per_usd"] if m is not None else pd.Series(dtype=float)
    out: pd.Series = combined_monthly(imf, fx_daily())["ghs_per_usd"]
    return out


def cedis(usd: float, rate: float | None) -> str:
    """'GH₵12,345' for a US$ amount, or an empty string without a rate."""
    return "" if rate is None or pd.isna(rate) else f"GH₵{usd * rate:,.0f}"


def fx_sidebar() -> None:
    """Show the current US$ to cedi rate in the sidebar on every page."""
    fx = fx_latest()
    if fx is None:
        return
    when, rate, source = fx
    st.sidebar.markdown(f"**1 US$ = GH₵{rate:,.2f}**")
    st.sidebar.caption(f"{when:%d %b %Y} · {source}")


def insights_rows() -> pd.DataFrame | None:
    """Every test-period and live forecast with its ex-ante confidence (both tracks)."""
    return _load(cfg().paths.forecasts / "insights_predictions.parquet")


def scored_rows() -> pd.DataFrame | None:
    """Forecasts whose actual price is known, with error and success columns."""
    from cocoasense.evaluation.insights import outcomes

    rows = insights_rows()
    if rows is None or cfg().insights is None:
        return None
    return outcomes(rows, cfg().insights.success_thresholds_pct)  # type: ignore[union-attr]


def update_log() -> list[dict[str, Any]]:
    """Records of `cocoasense update` runs, newest last."""
    path = cfg().paths.artifacts / "update_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def explainer(text: str) -> None:
    """A "what this means" box in plain language."""
    st.info(text, icon="💡")


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


def ticker_items() -> list[tuple[str, str, str]]:
    """Market strip: price and its monthly move, cedi value, rate, next forecast, confidence."""
    m = monthly()
    if m is None:
        return []
    price = m["price_usd_t"].dropna()
    items: list[tuple[str, str, str]] = []
    last, prev = float(price.iloc[-1]), float(price.iloc[-2])
    move = (last / prev - 1) * 100
    items.append(("COCOA", f"US${last:,.0f}/t", "cs-acc"))
    items.append(
        ("1M", f"{'+' if move >= 0 else ''}{move:.1f}%", "cs-up" if move >= 0 else "cs-down")
    )
    fx = fx_latest()
    if fx:
        items.append(("IN CEDIS", f"GH₵{last * fx[1]:,.0f}/t", ""))
        items.append(("USD/GHS", f"{fx[1]:.2f}", ""))
    year = price.loc[price.index >= price.index[-1] - pd.DateOffset(months=11)]
    items.append(("12M RANGE", f"{year.min():,.0f} – {year.max():,.0f}", ""))
    rows = insights_rows()
    if rows is not None:
        live = rows[(rows["track"] == "learning") & rows["y_true"].isna() & (rows["horizon"] == 1)]
        live = live[live["model"] != "naive"]
        if not live.empty:
            cons = float(live["y_pred"].mean())
            chg = (cons / last - 1) * 100
            when = pd.Timestamp(live["target_date"].iloc[0])
            items.append(
                (
                    f"{when:%b %y} FORECAST".upper(),
                    f"US${cons:,.0f} ({'+' if chg >= 0 else ''}{chg:.1f}%)",
                    "cs-up" if chg >= 0 else "cs-down",
                )
            )
            conf = live["confidence"].mode().iloc[0]
            cls = {"High": "cs-up", "Low": "cs-down"}.get(conf, "cs-acc")
            items.append(("CONFIDENCE", str(conf).upper(), cls))
    items.append(("DATA TO", f"{price.index[-1]:%b %Y}", ""))
    return items


def page_header(title: str) -> None:
    """Page title, terminal styling, market ticker and the permanent disclaimer banner."""
    from cocoasense.dashboard import ui

    st.set_page_config(page_title=f"CocoaSense AI · {title}", layout="wide")
    ui.apply_style()
    st.title(title)
    ui.ticker(ticker_items())
    st.warning(DISCLAIMER, icon="⚠️")
    fx_sidebar()


def missing(what: str, command: str) -> None:
    """Friendly message instead of an error when artifacts are missing."""
    st.info(
        f"{what} not found. Run the pipeline first:\n\n```\nuv run cocoasense {command}\n```\n\n"
        "or `uv run cocoasense all` to reproduce everything from the raw data."
    )


def label(model: str) -> str:
    """Human-readable model name."""
    return MODEL_LABELS.get(model, model)

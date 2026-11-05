"""Static figures for the dissertation (matplotlib, PNG). Every axis is labelled with units."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cocoasense.config import EvaluationConfig

MODEL_COLOURS = {
    "naive": "#7f7f7f",
    "arima": "#1f77b4",
    "random_forest": "#2ca02c",
    "xgboost": "#d62728",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shade_splits(ax: plt.Axes, ev: EvaluationConfig, last: pd.Timestamp) -> None:
    ax.axvspan(
        ev.ts("validation_start"),
        ev.ts("validation_end"),
        color="#ffd27f",
        alpha=0.35,
        label="validation (tuning)",
    )
    ax.axvspan(
        ev.ts("test_start"),
        last,
        color="#9ecae1",
        alpha=0.35,
        label="test (held out, never trained on)",
    )


def price_history(monthly: pd.DataFrame, ev: EvaluationConfig, path: Path) -> None:
    """Monthly cocoa price with the validation and test periods shaded."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    s = monthly["price_usd_t"].dropna()
    ax.plot(s.index, s.values, color="black", lw=1)
    _shade_splits(ax, ev, s.index.max())
    ax.set_yscale("log")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cocoa price, US$ per tonne (log scale)")
    ax.set_title("World Bank monthly cocoa price, 1960–present")
    ax.legend(loc="upper left")
    _save(fig, path)


def returns_and_volatility(monthly: pd.DataFrame, path: Path) -> None:
    """Monthly log returns and 12-month rolling volatility."""
    r = np.log(monthly["price_usd_t"].dropna()).diff().dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(r.index, r.values * 100, width=25, color="#555555")
    axes[0].set_ylabel("Monthly log return, %")
    axes[0].set_title("Monthly log returns and rolling volatility of cocoa prices")
    vol = r.rolling(12).std() * 100
    axes[1].plot(vol.index, vol.values, color="#d62728")
    axes[1].set_ylabel("12-month rolling s.d. of\nmonthly log returns, %")
    axes[1].set_xlabel("Month")
    _save(fig, path)


def exchange_rate(monthly: pd.DataFrame, path: Path) -> None:
    """Ghana cedis per US dollar."""
    s = monthly["ghs_per_usd"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ghana cedis per US$ (log scale)")
    ax.set_title("Exchange rate (IMF, monthly average, new cedis)")
    _save(fig, path)


def actual_vs_predicted(preds: pd.DataFrame, horizon: int, title: str, path: Path) -> None:
    """Actual price against each model's forecast, by target month."""
    p = preds[preds["horizon"] == horizon]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    actual = p.drop_duplicates("target_date").set_index("target_date")["y_true"].sort_index()
    ax.plot(actual.index, actual.values, color="black", lw=2, label="actual")
    for model, g in p.groupby("model"):
        g = g.sort_values("target_date")
        base = str(model).split("@")[0]
        ax.plot(
            g["target_date"], g["y_pred"], lw=1.2, label=str(model), color=MODEL_COLOURS.get(base)
        )
    ax.set_xlabel("Target month")
    ax.set_ylabel("Cocoa price, US$ per tonne")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)


def rolling_error(preds: pd.DataFrame, horizon: int, window: int, path: Path) -> None:
    """Rolling mean absolute percentage error per model."""
    p = preds[preds["horizon"] == horizon].copy()
    p["ape"] = (p["y_pred"] - p["y_true"]).abs() / p["y_true"] * 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for model, g in p.groupby("model"):
        s = g.set_index("target_date")["ape"].sort_index().rolling(window).mean()
        ax.plot(s.index, s.values, label=str(model), color=MODEL_COLOURS.get(str(model)))
    ax.set_xlabel("Target month")
    ax.set_ylabel(f"{window}-month rolling MAPE, %")
    ax.set_title(f"Rolling forecast error, {horizon}-month horizon (test period)")
    ax.legend()
    _save(fig, path)


def error_by_period(table: pd.DataFrame, path: Path) -> None:
    """MAPE by period, horizon and model."""
    t = table[table["period"] != "full"]
    horizons = sorted(t["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons, strict=True):
        pivot = t[t["horizon"] == h].pivot_table(index="period", columns="model", values="mape")
        pivot = pivot.reindex([p for p in ("pre_surge", "surge", "post_peak") if p in pivot.index])
        pivot.plot.bar(ax=ax, color=[MODEL_COLOURS.get(c, "#999999") for c in pivot.columns])
        ax.set_title(f"{h}-month horizon")
        ax.set_xlabel("Test sub-period (by target month)")
        ax.set_ylabel("MAPE, %")
        ax.tick_params(axis="x", rotation=0)
    _save(fig, path)


def strategy_outcomes(results: pd.DataFrame, path: Path) -> None:
    """Revenue per tonne by strategy over decision dates, and the forecast-driven share w."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for name, g in results.groupby("strategy"):
        g = g.sort_values("decision_date")
        axes[0].plot(g["decision_date"], g["revenue_per_t"], label=str(name), lw=1.2)
    axes[0].set_ylabel("Simulated revenue, US$ per tonne")
    axes[0].set_title("Simulated revenue by strategy (spot-proxy forward price)")
    axes[0].legend(fontsize=8)
    fd = results[results["strategy"] == "forecast_driven"].sort_values("decision_date")
    axes[1].step(fd["decision_date"], fd["w"], where="post", color="#9467bd")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Forward share w\n(forecast-driven)")
    axes[1].set_xlabel("Decision month")
    _save(fig, path)


def cedi_revenue(cedi: pd.DataFrame, path: Path) -> None:
    """Simulated revenue in cedis per tonne by delivery month, and the exchange rate used."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for name, g in cedi.groupby("strategy"):
        g = g.sort_values("target_date")
        axes[0].plot(g["target_date"], g["revenue_ghs_per_t"], label=str(name), lw=1.2)
    axes[0].set_ylabel("Simulated revenue, GHS per tonne")
    axes[0].set_title(r"Simulated revenue in cedis (US\$ revenue × delivery-month GHS/US\$)")
    axes[0].legend(fontsize=8)
    fx = cedi.drop_duplicates("target_date").sort_values("target_date")
    axes[1].plot(fx["target_date"], fx["ghs_per_usd"], color="#1f77b4")
    axes[1].set_ylabel("Ghana cedis per US$\n(IMF, monthly average)")
    axes[1].set_xlabel("Delivery month")
    _save(fig, path)

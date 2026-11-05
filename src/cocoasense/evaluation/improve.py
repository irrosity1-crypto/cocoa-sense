"""Improvement candidates built from stored frozen forecasts (decision D23, configs/improve.yaml).

Nothing here refits a model. Each candidate transforms forecasts that were already made with
information up to their decision month, and uses only prices up to that month itself:

* ``ensemble``: equal-weight geometric mean of several models' point forecasts, with the
  members' average log half-widths as its interval.
* ``scale_intervals``: widens each log half-width by max(1, current volatility / reference),
  where the reference is the median 12-month volatility up to the training cut-off.
* ``guard_dates``: decision months whose volatility exceeds anything seen up to the cut-off;
  the forecast-driven selling rule falls back to a fixed 50/50 split there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cocoasense.evaluation.insights import volatility

KEYS = ["decision_date", "target_date", "horizon"]


def ensemble(preds: pd.DataFrame, members: list[str], name: str = "ensemble") -> pd.DataFrame:
    """Equal-weight combination of ``members`` on the forecasts they all made."""
    sub = preds[preds["model"].isin(members)].copy()
    sub["log_pred"] = np.log(sub["y_pred"])
    sub["lo_w"] = np.log(sub["y_pred"] / sub["lower"])
    sub["hi_w"] = np.log(sub["upper"] / sub["y_pred"])
    out = (
        sub.groupby(KEYS)
        .agg(
            n=("model", "nunique"),
            last_obs=("last_obs", "first"),
            y_true=("y_true", "first"),
            log_pred=("log_pred", "mean"),
            lo_w=("lo_w", "mean"),
            hi_w=("hi_w", "mean"),
        )
        .reset_index()
    )
    out = out[out["n"] == len(members)]
    point = np.exp(out["log_pred"])
    return pd.DataFrame(
        {
            "model": name,
            **{k: out[k] for k in KEYS},
            "last_obs": out["last_obs"],
            "y_true": out["y_true"],
            "y_pred": point,
            "lower": point * np.exp(-out["lo_w"]),
            "upper": point * np.exp(out["hi_w"]),
        }
    ).reset_index(drop=True)


def volatility_reference(price: pd.Series, window: int, cutoff: pd.Timestamp) -> float:
    """Median 12-month volatility over all months up to the training cut-off."""
    return float(volatility(price, window).loc[:cutoff].median())


def scale_intervals(
    preds: pd.DataFrame, price: pd.Series, window: int, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Widen intervals when volatility at the decision month exceeds the training median."""
    vol = volatility(price, window)
    ref = volatility_reference(price, window, cutoff)
    factor = (preds["decision_date"].map(vol) / ref).clip(lower=1.0).fillna(1.0)
    out = preds.copy()
    lo_w = np.log(out["y_pred"] / out["lower"]) * factor
    hi_w = np.log(out["upper"] / out["y_pred"]) * factor
    out["lower"] = out["y_pred"] * np.exp(-lo_w)
    out["upper"] = out["y_pred"] * np.exp(hi_w)
    out["vol_factor"] = factor
    return out


def guard_dates(price: pd.Series, window: int, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    """Decision months after the cut-off whose volatility exceeds the highest up to it."""
    vol = volatility(price, window)
    ceiling = vol.loc[:cutoff].max()
    return pd.DatetimeIndex(vol.index[(vol > ceiling) & (vol.index > cutoff)])


def apply_guard(results: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast-driven rows take the fixed 50/50 outcome on guarded decision months."""
    out = results.copy()
    fd = out["strategy"] == "forecast_driven"
    half = out[out["strategy"] == "fixed_50_50"].set_index("decision_date")
    hit = fd & out["decision_date"].isin(dates)
    for col in ("w", "revenue", "revenue_per_t"):
        out.loc[hit, col] = out.loc[hit, "decision_date"].map(half[col]).to_numpy()
    return out


def scale_intervals_expanding(
    preds: pd.DataFrame, price: pd.Series, window: int, models: list[str]
) -> pd.DataFrame:
    """Volatility-aware ranges for continually refitted forecasts (learning track, live).

    Like ``scale_intervals``, but the reference is the median volatility of all months up to each
    row's own decision month, so no row uses information from after its decision. Only rows of
    ``models`` are changed; ``range_factor`` records the widening (1 = unchanged).
    """
    vol = volatility(price, window)
    ref = vol.expanding().median()
    out = preds.copy()
    sel = out["model"].isin(models)
    factor = (out["decision_date"].map(vol) / out["decision_date"].map(ref)).clip(lower=1.0)
    factor = factor.fillna(1.0).where(sel, 1.0)
    lo_w = np.log(out["y_pred"] / out["lower"]) * factor
    hi_w = np.log(out["upper"] / out["y_pred"]) * factor
    out["lower"] = out["y_pred"] * np.exp(-lo_w)
    out["upper"] = out["y_pred"] * np.exp(hi_w)
    out["range_factor"] = factor
    return out

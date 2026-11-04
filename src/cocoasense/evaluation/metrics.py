"""Forecast accuracy metrics and the Diebold–Mariano test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Accuracy of one group of predictions (columns ``y_true, y_pred, last_obs, lower, upper``).

    * MAE, RMSE in USD/tonne; MAPE in percent.
    * ``directional_acc``: share of forecasts that called the direction of change from the last
      observed price correctly (forecasts of exactly no change are excluded, so it is NaN for
      the naive model).
    * ``within_5pct`` / ``within_10pct``: share of forecasts within 5% / 10% of the actual price.
    * ``coverage``: share of actual prices inside the prediction interval.
    """
    y, p = df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)
    err = p - y
    ape = np.abs(err) / np.abs(y)
    pred_move = np.sign(p - df["last_obs"].to_numpy(float))
    true_move = np.sign(y - df["last_obs"].to_numpy(float))
    called = pred_move != 0
    lower, upper = df["lower"].to_numpy(float), df["upper"].to_numpy(float)
    has_int = ~np.isnan(lower) & ~np.isnan(upper)
    return {
        "n": float(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(ape) * 100),
        "bias": float(np.mean(err)),
        "directional_acc": float(np.mean(pred_move[called] == true_move[called]))
        if called.any()
        else float("nan"),
        "within_5pct": float(np.mean(ape <= 0.05)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "coverage": float(np.mean((y[has_int] >= lower[has_int]) & (y[has_int] <= upper[has_int])))
        if has_int.any()
        else float("nan"),
        "interval_width_pct": float(np.mean((upper[has_int] - lower[has_int]) / y[has_int]) * 100)
        if has_int.any()
        else float("nan"),
    }


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, horizon: int) -> tuple[float, float]:
    """Diebold–Mariano test on squared errors with the Harvey–Leybourne–Newbold correction.

    Returns ``(statistic, two-sided p-value)``. Negative statistics favour the model. With the
    small samples used here (tens of months) the test has low power and is only indicative.
    """
    d = e_model**2 - e_bench**2
    n = d.size
    if n < 8:
        return float("nan"), float("nan")
    d_bar = d.mean()
    lags = max(horizon - 1, 0)
    gamma = [np.mean((d[k:] - d_bar) * (d[: n - k] - d_bar)) for k in range(lags + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = d_bar / math.sqrt(var)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def assign_periods(
    preds: pd.DataFrame, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]
) -> pd.DataFrame:
    """Long table with one copy of each row per period it belongs to (plus ``full``)."""
    parts = [preds.assign(period="full")]
    for name, (start, end) in periods.items():
        mask = preds["target_date"] >= start
        if end is not None:
            mask &= preds["target_date"] <= end
        parts.append(preds.loc[mask].assign(period=name))
    return pd.concat(parts, ignore_index=True)


def score(preds: pd.DataFrame, baseline: str = "naive") -> pd.DataFrame:
    """Metrics by stage, period, horizon and model, with error relative to the baseline.

    Expects a ``period`` column (see :func:`assign_periods`). Relative errors and the
    Diebold–Mariano test compare each model with the baseline on the identical set of months.
    """
    keys = ["stage", "period", "horizon"]
    rows = []
    for key, group in preds.groupby([*keys, "model"], sort=True):
        rows.append({**dict(zip([*keys, "model"], key, strict=True)), **point_metrics(group)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    rel_cols = {"rel_mae": "mae", "rel_rmse": "rmse", "rel_mape": "mape"}
    for col in [*rel_cols, "dm_stat", "dm_pvalue"]:
        table[col] = np.nan
    for key, group in preds.groupby(keys):
        base = group[group["model"] == baseline].set_index("decision_date")
        if base.empty:
            continue
        for model, mg in group.groupby("model"):
            joined = mg.set_index("decision_date").join(base, rsuffix="_b", how="inner")
            if joined.empty:
                continue
            sel = (table[keys] == pd.Series(key, index=keys)).all(axis=1) & (
                table["model"] == model
            )
            m = point_metrics(joined[["y_true", "y_pred", "last_obs", "lower", "upper"]])
            b_df = joined[["y_true_b", "y_pred_b", "last_obs_b", "lower_b", "upper_b"]]
            b = point_metrics(
                b_df.set_axis(["y_true", "y_pred", "last_obs", "lower", "upper"], axis=1)
            )
            for col, src in rel_cols.items():
                table.loc[sel, col] = m[src] / b[src] if b[src] else np.nan
            if model != baseline:
                e_m = (joined["y_pred"] - joined["y_true"]).to_numpy()
                e_b = (joined["y_pred_b"] - joined["y_true_b"]).to_numpy()
                stat, p = diebold_mariano(e_m, e_b, int(key[2]))
                table.loc[sel, "dm_stat"] = stat
                table.loc[sel, "dm_pvalue"] = p
    return table

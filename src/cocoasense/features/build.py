"""Leakage-safe features and targets.

Timing convention: a row dated month ``t`` describes the decision made at the *end* of month t,
when that month's average price is known. Features for row t use prices up to and including t.
Targets look forward and are therefore built by a separate function that is never used as input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cocoasense.config import FeatureConfig

PRICE = "price_usd_t"
FX = "ghs_per_usd"


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """Ordered list of feature names produced by :func:`build_features`."""
    cols: list[str] = []
    if cfg.include_level:
        cols.append("log_price")
    cols += [f"ret_{k}m" for k in cfg.return_windows]
    cols += [f"vol_{w}m" for w in cfg.rolling_windows]
    cols += [f"gap_ma_{w}m" for w in cfg.rolling_windows]
    if cfg.include_seasonality:
        cols += ["month_sin", "month_cos"]
    if cfg.use_fx:
        cols += [f"fx_ret_{k}m" for k in cfg.fx_return_windows]
    return cols


def build_features(
    history: pd.DataFrame, as_of: pd.Timestamp | None, cfg: FeatureConfig
) -> pd.DataFrame:
    """Return one feature row per month up to ``as_of``, using only rows up to ``as_of``.

    ``history`` must be indexed by month-start dates and contain ``price_usd_t`` (and
    ``ghs_per_usd`` when ``cfg.use_fx`` is true). Warm-up rows whose look-back window is
    incomplete contain NaN and are left for the caller to drop explicitly.
    """
    if as_of is not None:
        history = history.loc[:as_of]
    if PRICE not in history:
        raise KeyError(f"history needs a '{PRICE}' column")
    lp = np.log(history[PRICE].astype(float))
    r1 = lp.diff()
    out = pd.DataFrame(index=history.index)
    if cfg.include_level:
        out["log_price"] = lp
    for k in cfg.return_windows:
        out[f"ret_{k}m"] = lp - lp.shift(k)
    for w in cfg.rolling_windows:
        out[f"vol_{w}m"] = r1.rolling(w, min_periods=w).std()
    for w in cfg.rolling_windows:
        out[f"gap_ma_{w}m"] = lp - lp.rolling(w, min_periods=w).mean()
    if cfg.include_seasonality:
        month = pd.DatetimeIndex(history.index).month.to_numpy()
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)
    if cfg.use_fx:
        if FX not in history:
            raise KeyError(f"use_fx is true but history has no '{FX}' column")
        lfx = np.log(history[FX].astype(float))
        for k in cfg.fx_return_windows:
            out[f"fx_ret_{k}m"] = lfx - lfx.shift(k)
    return out[feature_columns(cfg)]


def build_targets(history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Forward log returns ``log(P[t+h] / P[t])``; NaN where t+h is beyond the data.

    These look into the future by construction. They are used only as training labels for rows
    whose target month lies inside the training window, and as ``y_true`` in evaluation.
    """
    lp = np.log(history[PRICE].astype(float))
    return pd.DataFrame({f"target_{h}m": lp.shift(-h) - lp for h in horizons}, index=history.index)


def training_table(
    history: pd.DataFrame,
    cfg: FeatureConfig,
    horizon: int,
    train_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and label for every complete row with a known target inside ``history``.

    Because targets are computed from ``history`` itself, no label can reach past the last month
    of ``history``. Callers pass history truncated at the training cut-off.
    """
    X = build_features(history, None, cfg)
    y = build_targets(history, [horizon])[f"target_{horizon}m"]
    mask = (X.index >= train_start) & X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]

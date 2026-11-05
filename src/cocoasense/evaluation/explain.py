"""Why did a tree model forecast what it did? Exact per-input contributions for one forecast.

A tree model predicts the h-month log return as ``base + sum(contributions)``:

* XGBoost: Shapley values from the booster's built-in ``pred_contribs`` (exact for trees).
* Random Forest: the decision-path decomposition (each split's change in node value is
  credited to the feature it split on, averaged over trees).

Contributions are in log-return units; multiplied by 100 they read as approximate percentage
points of the forecast change. The two seasonal features are reported together.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from cocoasense.features.build import build_features
from cocoasense.models.tree import TreeForecaster

FEATURE_LABELS = {
    "log_price": "Price level",
    "ret_1m": "Change over the last month",
    "ret_2m": "Change over the last 2 months",
    "ret_3m": "Change over the last 3 months",
    "ret_6m": "Change over the last 6 months",
    "ret_12m": "Change over the last 12 months",
    "vol_3m": "Volatility, last 3 months",
    "vol_6m": "Volatility, last 6 months",
    "vol_12m": "Volatility, last 12 months",
    "gap_ma_3m": "Distance from 3-month average",
    "gap_ma_6m": "Distance from 6-month average",
    "gap_ma_12m": "Distance from 12-month average",
    "month_sin": "Time of year",
    "month_cos": "Time of year",
}


def _rf_contributions(rf: RandomForestRegressor, row: np.ndarray) -> tuple[float, np.ndarray]:
    """Decision-path decomposition averaged over the forest."""
    bias, contrib = 0.0, np.zeros(row.shape[1])
    for est in rf.estimators_:
        tree = est.tree_
        values = tree.value[:, 0, 0]
        path = est.decision_path(row).indices
        bias += values[path[0]]
        for parent, child in itertools.pairwise(path):
            contrib[tree.feature[parent]] += values[child] - values[parent]
    n = len(rf.estimators_)
    return bias / n, contrib / n


def contributions(model: TreeForecaster, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per input: its value, label and contribution to this forecast's log return."""
    est = model.model_for(horizon)
    tail = history.iloc[-(model.features.max_lookback + 2) :]
    row = build_features(tail, None, model.features).iloc[[-1]][model.columns]
    if row.isna().any(axis=None):
        raise ValueError("incomplete features for the latest month")
    if isinstance(est, xgb.XGBRegressor):
        out = est.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0]
        base, contrib = float(out[-1]), np.asarray(out[:-1], dtype=float)
    elif isinstance(est, RandomForestRegressor):
        base, contrib = _rf_contributions(est, row.to_numpy(dtype=float))
    else:
        raise TypeError(f"no explanation method for {type(est).__name__}")
    table = pd.DataFrame(
        {
            "feature": model.columns,
            "value": row.iloc[0].to_numpy(dtype=float),
            "contribution": contrib,
        }
    )
    table["label"] = table["feature"].map(FEATURE_LABELS).fillna(table["feature"])
    table["base"] = base
    table["prediction"] = base + float(contrib.sum())
    return table


def grouped(table: pd.DataFrame) -> pd.DataFrame:
    """Combine inputs that share a label (the two seasonal terms), largest effect first."""
    g = table.groupby("label", sort=False).agg(
        contribution=("contribution", "sum"), feature=("feature", "first"), value=("value", "first")
    )
    g = g.reset_index()
    return g.loc[g["contribution"].abs().sort_values(ascending=False).index].reset_index(drop=True)


def describe_value(feature: str, value: float) -> str:
    """Plain wording for an input's current value."""
    if feature == "log_price":
        return f"US${np.exp(value):,.0f}/t"
    if feature.startswith(("ret_", "gap_ma_")):
        return f"{(np.exp(value) - 1) * 100:+.1f}%"
    if feature.startswith("vol_"):
        return f"{value * 100:.1f}% a month"
    return ""

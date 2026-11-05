"""Learning track: refit every month on all prices known so far (a separately labelled comparison).

The frozen track (``walk_forward.py``) fits each model once on targets up to 2019-12 and never
refits; it is the project's held-out test. The learning track answers a different question: does
updating the models with each new month of prices help? At every decision month *t* each model is
refitted on ``history.loc[:t]`` (so every training label is dated at or before *t*) and then
forecasts *t + h*. Hyperparameters stay fixed at the frozen choice, which was tuned on pre-2020
data only; the fitted parameters and the conformal interval widths are re-estimated each month.

The same loop, run from the last data month, produces the live forecasts for months that have not
happened yet (``y_true`` is NaN for those rows).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from cocoasense.models.base import Forecaster

logger = logging.getLogger(__name__)

LEARNING_COLUMNS = [
    "track",
    "model",
    "decision_date",
    "target_date",
    "horizon",
    "last_obs",
    "y_true",
    "y_pred",
    "lower",
    "upper",
    "fit_end",
]


def fixed_grid(best_params: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    """One-point grid (per parameter, the values chosen for any horizon) from frozen best params."""
    grid: dict[str, list[Any]] = {}
    for params in best_params.values():
        for k, v in params.items():
            if v not in grid.setdefault(k, []):
                grid[k].append(v)
    return grid


def refit_walk_forward(
    make_model: Callable[[], Forecaster],
    history: pd.DataFrame,
    first_target: pd.Timestamp,
    horizons: Iterable[int],
    include_future: bool = True,
    skip: set[tuple[pd.Timestamp, int]] | None = None,
) -> pd.DataFrame:
    """Refit at each decision month and forecast every horizon from it.

    Decision months run from ``first_target - max(h)`` to the last month of ``history``. A row is
    kept when its target month is on/after ``first_target`` and either inside the data (scored)
    or, with ``include_future``, after the last data month (live forecast). ``skip`` lists
    (decision_date, horizon) pairs already computed, so updates only fit new months.
    """
    horizons = sorted(horizons)
    idx = pd.DatetimeIndex(history.index)
    last = idx[-1]
    price = history["price_usd_t"]
    skip = skip or set()
    rows: list[dict[str, Any]] = []
    for t in idx[idx >= first_target - pd.DateOffset(months=max(horizons))]:
        wanted = []
        for h in horizons:
            target = t + pd.DateOffset(months=h)
            if target < first_target or (pd.Timestamp(t), h) in skip:
                continue
            if target <= last or include_future:
                wanted.append((h, pd.Timestamp(target)))
        if not wanted:
            continue
        model = make_model()
        past = history.loc[:t]
        model.fit(past)
        info = model.info()
        _assert_fitted_on_past(info, pd.Timestamp(t))
        for h, target in wanted:
            fc = model.predict(past, h)
            rows.append(
                {
                    "track": "learning",
                    "model": model.name,
                    "decision_date": pd.Timestamp(t),
                    "target_date": target,
                    "horizon": h,
                    "last_obs": float(price.loc[t]),
                    "y_true": float(price.loc[target]) if target in idx else np.nan,
                    "y_pred": fc.point,
                    "lower": np.nan if fc.lower is None else fc.lower,
                    "upper": np.nan if fc.upper is None else fc.upper,
                    "fit_end": pd.Timestamp(t),
                }
            )
        logger.info("learning: %s refitted through %s", model.name, f"{t:%Y-%m}")
    return pd.DataFrame(rows, columns=LEARNING_COLUMNS)


def _assert_fitted_on_past(info: dict[str, Any], t: pd.Timestamp) -> None:
    """Guard: no training label may be dated after the decision month."""
    tmax = info.get("train_target_max")
    if tmax is not None and pd.Timestamp(tmax) > t:
        raise AssertionError(f"learning model trained on data up to {tmax}, after {t:%Y-%m}")
    for d in (info.get("last_label_date") or {}).values():
        if pd.Timestamp(d) > t:
            raise AssertionError(f"learning model label dated {d} is after {t:%Y-%m}")


def merge_predictions(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Combine stored and new learning rows.

    Rows are keyed by (model, decision_date, horizon); a stored live forecast keeps its original
    prediction, so later updates can score what was actually forecast at the time.
    """
    order = ["model", "horizon", "decision_date"]
    if old is None or old.empty:
        return new.sort_values(order).reset_index(drop=True)
    if new.empty:
        return old.sort_values(order).reset_index(drop=True)
    key = ["model", "decision_date", "horizon"]
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(key, keep="first")
    return both.sort_values(order).reset_index(drop=True)


def fill_actuals(preds: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Set ``y_true`` from ``price`` wherever the target month is now known."""
    out = preds.copy()
    known = out["target_date"].map(price)
    out["y_true"] = known.where(known.notna(), out["y_true"])
    return out

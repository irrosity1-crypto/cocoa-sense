"""Improvement candidates (SYNTHETIC data): ensemble, volatility-scaled ranges, guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.evaluation.improve import (
    apply_guard,
    ensemble,
    guard_dates,
    scale_intervals,
    scale_intervals_expanding,
    volatility_reference,
)


def _preds() -> pd.DataFrame:
    d = pd.Timestamp("2020-01-01")
    rows = []
    for model, pred, half in (("a", 100.0, 0.1), ("b", 121.0, 0.2), ("c", 110.0, 0.1)):
        rows.append(
            {
                "model": model,
                "decision_date": d,
                "target_date": d + pd.DateOffset(months=1),
                "horizon": 1,
                "last_obs": 105.0,
                "y_true": 112.0,
                "y_pred": pred,
                "lower": pred * np.exp(-half),
                "upper": pred * np.exp(half),
            }
        )
    return pd.DataFrame(rows)


def test_ensemble_is_geometric_mean_with_average_widths() -> None:
    out = ensemble(_preds(), ["a", "b"])
    assert len(out) == 1 and out.loc[0, "model"] == "ensemble"
    assert out.loc[0, "y_pred"] == pytest.approx(110.0)  # sqrt(100 * 121)
    assert np.log(out.loc[0, "upper"] / out.loc[0, "y_pred"]) == pytest.approx(0.15)
    assert out.loc[0, "y_true"] == 112.0
    # A forecast is combined only when every member made it.
    assert ensemble(_preds()[_preds()["model"] != "b"], ["a", "b"]).empty


def _price(n: int = 60, jump_at: int = 40) -> pd.Series:
    rng = np.random.default_rng(0)
    r = 0.02 * rng.standard_normal(n)
    r[jump_at:] *= 6  # a much more volatile regime later on
    return pd.Series(100 * np.exp(np.cumsum(r)), pd.date_range("2015-01-01", periods=n, freq="MS"))


def test_scale_intervals_only_widens_and_uses_the_past() -> None:
    price = _price()
    cutoff = price.index[30]
    rows = pd.concat(
        [_preds().assign(decision_date=d) for d in price.index[31:]], ignore_index=True
    )
    scaled = scale_intervals(rows, price, 12, cutoff)
    assert (scaled["vol_factor"] >= 1).all()
    assert (scaled["upper"] >= rows["upper"] - 1e-9).all()
    assert (scaled["lower"] <= rows["lower"] + 1e-9).all()
    assert scaled["vol_factor"].iloc[-1] > 1.5  # widened in the volatile regime
    # Changing prices after a decision month leaves that month's widening unchanged.
    t = price.index[45]
    wrecked = price.copy()
    wrecked[wrecked.index > t] *= 10
    again = scale_intervals(rows, wrecked, 12, cutoff)
    early = rows["decision_date"] <= t
    pd.testing.assert_series_equal(scaled.loc[early, "vol_factor"], again.loc[early, "vol_factor"])
    assert volatility_reference(price, 12, cutoff) > 0


def test_guard_dates_and_apply_guard() -> None:
    price = _price()
    cutoff = price.index[30]
    dates = guard_dates(price, 12, cutoff)
    assert len(dates) > 0 and (dates > cutoff).all()
    d1, d2 = dates[0], price.index[31]
    res = pd.DataFrame(
        {
            "decision_date": [d1, d1, d2, d2],
            "strategy": ["forecast_driven", "fixed_50_50"] * 2,
            "w": [0.9, 0.5, 0.2, 0.5],
            "revenue": [9.0, 5.0, 2.0, 5.0],
            "revenue_per_t": [9.0, 5.0, 2.0, 5.0],
        }
    )
    out = apply_guard(res, pd.DatetimeIndex([d1]))
    assert out.loc[0, "w"] == 0.5 and out.loc[0, "revenue_per_t"] == 5.0  # guarded month
    assert out.loc[2, "w"] == 0.2  # unguarded month unchanged


def test_expanding_scaling_touches_only_chosen_models_and_uses_the_past() -> None:
    price = _price()
    rows = pd.concat(
        [_preds().assign(decision_date=d) for d in price.index[20:]], ignore_index=True
    )
    out = scale_intervals_expanding(rows, price, 12, ["b"])
    others = out["model"] != "b"
    assert (out.loc[others, "range_factor"] == 1.0).all()
    pd.testing.assert_series_equal(out.loc[others, "upper"], rows.loc[others, "upper"])
    assert out.loc[~others, "range_factor"].max() > 1.5  # widened in the volatile regime
    t = price.index[45]
    wrecked = price.copy()
    wrecked[wrecked.index > t] *= 10
    again = scale_intervals_expanding(rows, wrecked, 12, ["b"])
    early = rows["decision_date"] <= t
    pd.testing.assert_series_equal(out.loc[early, "upper"], again.loc[early, "upper"])

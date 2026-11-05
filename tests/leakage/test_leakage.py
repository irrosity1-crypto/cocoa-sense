"""Mandatory leakage tests (build prompt section 6). Data is SYNTHETIC.

1. Truncation: features from full history == features from history cut at as_of.
2. Perturbation: multiplying every price after as_of by 10 changes no feature and no forecast.
3. Walk-forward: training rows and their targets end at or before the cut-off.
4. Fit on training only: tuning and calibration are unchanged by post-cut-off data.
5. Strategies: every decision uses only data up to its decision date.
6. Learning track: a monthly-refitted forecast made at t is unchanged by any data after t.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from cocoasense.config import ProjectConfig
from cocoasense.evaluation.learning import refit_walk_forward
from cocoasense.evaluation.walk_forward import TestPeriodLockedError, Window, forecast_window
from cocoasense.features.build import build_features
from cocoasense.models.base import Forecast
from cocoasense.models.registry import MODEL_NAMES, build_model
from cocoasense.simulation.engine import simulate
from cocoasense.simulation.futures_provider import SpotProxyProvider
from cocoasense.simulation.strategies import FIXED_STRATEGIES, ForecastDriven

AS_OF = [pd.Timestamp(d) for d in ("1995-06-01", "2005-01-01", "2012-12-01")]
CUT = pd.Timestamp("2009-12-01")
WINDOW = Window("validation", CUT, pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-01"))


def _perturb(history: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    h = history.copy()
    h.loc[h.index > after, "price_usd_t"] *= 10
    h.loc[h.index > after, "ghs_per_usd"] *= 10
    return h


@pytest.mark.parametrize("use_fx", [False, True])
@pytest.mark.parametrize("as_of", AS_OF)
def test_1_truncation(
    history: pd.DataFrame, project: ProjectConfig, as_of: pd.Timestamp, use_fx: bool
) -> None:
    cfg = project.features.model_copy(update={"use_fx": use_fx})
    full = build_features(history, as_of, cfg)
    cut = build_features(history.loc[:as_of], as_of, cfg)
    pd.testing.assert_frame_equal(full, cut)
    # also: rows up to as_of are identical when computed on the entire history
    whole = build_features(history, None, cfg).loc[:as_of]
    pd.testing.assert_frame_equal(whole, cut)


@pytest.mark.parametrize("as_of", AS_OF)
def test_2_perturbation_features(
    history: pd.DataFrame, project: ProjectConfig, as_of: pd.Timestamp
) -> None:
    cfg = project.features.model_copy(update={"use_fx": True})
    a = build_features(history, None, cfg).loc[:as_of]
    b = build_features(_perturb(history, as_of), None, cfg).loc[:as_of]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_2_perturbation_forecasts(history: pd.DataFrame, project: ProjectConfig, name: str) -> None:
    """Forecasts made at decision dates up to as_of are unchanged by later prices."""
    as_of = pd.Timestamp("2011-12-01")
    a, _ = forecast_window(
        build_model(name, project), history, WINDOW, [1, 3], pd.Timestamp("2017-01-01")
    )
    b, _ = forecast_window(
        build_model(name, project),
        _perturb(history, as_of),
        WINDOW,
        [1, 3],
        pd.Timestamp("2017-01-01"),
    )
    sel_a = a[a["decision_date"] <= as_of].reset_index(drop=True)
    sel_b = b[b["decision_date"] <= as_of].reset_index(drop=True)
    cols = ["decision_date", "horizon", "y_pred", "lower", "upper"]
    pd.testing.assert_frame_equal(sel_a[cols], sel_b[cols])


class _Cheater:
    """Test double that claims to have trained beyond the cut-off."""

    name = "cheater"

    def fit(self, history: pd.DataFrame) -> None:
        return None

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        t = history.index[-1]
        p = float(history["price_usd_t"].iloc[-1])
        return Forecast(t, horizon, p, p, p)

    def info(self) -> dict[str, Any]:
        return {"train_target_max": "2015-01-01"}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_3_walk_forward_training_ends_before_decisions(
    history: pd.DataFrame, project: ProjectConfig, name: str
) -> None:
    preds, info = forecast_window(
        build_model(name, project), history, WINDOW, [1, 3], pd.Timestamp("2017-01-01")
    )
    assert pd.Timestamp(info["train_target_max"]) <= CUT
    for d in (info.get("last_label_date") or {}).values():
        assert pd.Timestamp(d) <= CUT
    assert (preds["target_date"] > CUT).all()
    for row in preds.itertuples():
        assert row.target_date == row.decision_date + pd.DateOffset(months=int(row.horizon))


def test_3_guard_rejects_models_trained_past_cutoff(history: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="after cut-off"):
        forecast_window(_Cheater(), history, WINDOW, [1], pd.Timestamp("2017-01-01"))


def test_3_test_period_locked(history: pd.DataFrame, project: ProjectConfig) -> None:
    test = Window(
        "test", pd.Timestamp("2013-12-01"), pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-01")
    )
    with pytest.raises(TestPeriodLockedError):
        forecast_window(
            build_model("naive", project), history, test, [1], pd.Timestamp("2017-01-01")
        )


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "arima", "naive", "lstm"])
def test_4_fit_on_training_only(history: pd.DataFrame, project: ProjectConfig, name: str) -> None:
    """Hyperparameters, intervals and coefficients do not depend on post-cut-off data."""
    _, info_a = forecast_window(
        build_model(name, project), history, WINDOW, [1, 3], pd.Timestamp("2017-01-01")
    )
    _, info_b = forecast_window(
        build_model(name, project),
        _perturb(history, CUT),
        WINDOW,
        [1, 3],
        pd.Timestamp("2017-01-01"),
    )
    for d in (info_a, info_b):
        d.pop("fit_seconds")
        d.pop("total_seconds")
    assert info_a == info_b


def test_5_strategy_decisions_use_past_only(history: pd.DataFrame, project: ProjectConfig) -> None:
    as_of = pd.Timestamp("2011-06-01")
    strategies = [*FIXED_STRATEGIES, ForecastDriven(0.5)]

    def run(h: pd.DataFrame) -> pd.DataFrame:
        preds, _ = forecast_window(
            build_model("arima", project), h, WINDOW, [3], pd.Timestamp("2017-01-01")
        )
        res = simulate(preds, strategies, SpotProxyProvider(h["price_usd_t"]), 1.0, 0.1)
        return res[res["decision_date"] <= as_of].reset_index(drop=True)

    a, b = run(history), run(_perturb(history, as_of))
    pd.testing.assert_series_equal(a["w"], b["w"])
    pd.testing.assert_series_equal(a["forward_price"], b["forward_price"])


@pytest.mark.parametrize("name", ["naive", "arima", "xgboost"])
def test_6_learning_track_refit_uses_only_the_past(
    history: pd.DataFrame, project: ProjectConfig, name: str
) -> None:
    t = pd.Timestamp("2005-01-01")
    upto = history.loc[: t + pd.DateOffset(months=2)]

    def make() -> Any:
        return build_model(name, project, pd.Timestamp("1985-01"))

    first = t + pd.DateOffset(months=1)
    clean = refit_walk_forward(make, upto, first, [1, 3])
    dirty = refit_walk_forward(make, _perturb(upto, t), first, [1, 3])
    a = clean[clean["decision_date"] == t].reset_index(drop=True)
    b = dirty[dirty["decision_date"] == t].reset_index(drop=True)
    assert len(a) == 2
    pd.testing.assert_frame_equal(
        a[["y_pred", "lower", "upper", "fit_end"]], b[["y_pred", "lower", "upper", "fit_end"]]
    )

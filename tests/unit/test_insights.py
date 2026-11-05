"""Scorecards, ex-ante confidence, explanations and the learning loop (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.config import ConfidenceRule, InsightsConfig, LearningConfig, ProjectConfig
from cocoasense.evaluation.insights import (
    NOT_ENOUGH,
    brier,
    confidence_check,
    ex_ante_confidence,
    explain,
    label,
    model_spread,
    outcomes,
    probability_calibration,
    scorecard,
)
from cocoasense.evaluation.learning import (
    fill_actuals,
    fixed_grid,
    merge_predictions,
    refit_walk_forward,
)
from cocoasense.models.registry import build_model

CFG = InsightsConfig(
    success_thresholds_pct=[5, 10],
    min_past_errors=3,
    volatility_window=4,
    high=ConfidenceRule(p_within_10=0.7, model_spread_pct=5),
    low=ConfidenceRule(p_within_10=0.5, model_spread_pct=15),
    learning=LearningConfig(first_target="2010-01", models=["naive"]),
)


def _rows() -> pd.DataFrame:
    """Hand-made forecasts: last price 100; actuals 104, 92, 100."""
    return pd.DataFrame(
        {
            "track": "frozen",
            "model": ["m", "m", "m"],
            "decision_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "target_date": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"]),
            "horizon": 1,
            "last_obs": [100.0, 100.0, 100.0],
            "y_true": [104.0, 92.0, 100.0],
            "y_pred": [102.0, 103.0, 99.0],
            "lower": [95.0, 95.0, 101.0],
            "upper": [110.0, 110.0, 105.0],
        }
    )


def test_outcomes_and_scorecard() -> None:
    o = outcomes(_rows(), [5, 10])
    assert o["within_5pct"].tolist() == [1.0, 0.0, 1.0]
    assert o["within_10pct"].tolist() == [1.0, 0.0, 1.0]  # 11/92 = 12 % misses ±10 %
    # direction: up/up right, up/down wrong, actual unchanged -> undefined
    assert o["direction_correct"].iloc[:2].tolist() == [1.0, 0.0]
    assert np.isnan(o["direction_correct"].iloc[2])
    assert o["in_interval"].tolist() == [1.0, 0.0, 0.0]
    card = scorecard(o, ["model"], [5, 10]).iloc[0]
    assert card["n"] == 3
    assert card["within_10pct_pct"] == pytest.approx(200 / 3)
    assert card["direction_correct_pct"] == pytest.approx(50)
    assert card["mae"] == pytest.approx((2 + 11 + 1) / 3)


def test_outcomes_drop_rows_without_actuals() -> None:
    rows = _rows()
    rows.loc[2, "y_true"] = np.nan
    assert len(outcomes(rows, [10])) == 2


def test_label_rules() -> None:
    assert label(0.8, 2.0, CFG) == "High"
    assert label(0.8, 8.0, CFG) == "Medium"  # good history but models disagree a bit
    assert label(0.6, 1.0, CFG) == "Medium"
    assert label(0.4, 1.0, CFG) == "Low"
    assert label(0.9, 20.0, CFG) == "Low"  # strong disagreement overrides
    assert label(float("nan"), 1.0, CFG) == NOT_ENOUGH
    assert label(0.8, float("nan"), CFG) == "High"  # single model: no spread signal


def _history(n: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Past forecasts of model m (all 2 % too low) and a gently moving price."""
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    price = pd.Series(100 * np.exp(0.01 * np.arange(n) + 0.02 * np.sin(np.arange(n))), dates)
    hist = pd.DataFrame(
        {
            "track": "frozen",
            "model": "m",
            "decision_date": dates[:-1],
            "target_date": dates[1:],
            "horizon": 1,
            "last_obs": price.iloc[:-1].to_numpy(),
            "y_true": price.iloc[1:].to_numpy(),
            "y_pred": price.iloc[1:].to_numpy() / 1.02,
            "lower": np.nan,
            "upper": np.nan,
        }
    )
    return hist, price


def test_ex_ante_confidence_uses_only_known_errors() -> None:
    hist, price = _history()
    out = ex_ante_confidence(hist, hist, price, CFG)
    for r in out.itertuples():  # only errors whose target month is <= the decision month
        assert r.n_past_errors <= (hist["target_date"] <= r.decision_date).sum()
    early = out.iloc[0]
    assert early["confidence"] == NOT_ENOUGH and np.isnan(early["p_within_10"])
    late = out.iloc[-1]
    assert late["n_past_errors"] >= CFG.min_past_errors
    assert late["p_within_10"] > 0.5  # every past error is about 2 %
    assert 0 <= late["p_up"] <= 1


def test_ex_ante_confidence_ignores_future_outcomes() -> None:
    hist, price = _history()
    base = ex_ante_confidence(hist, hist, price, CFG)
    wrecked = hist.copy()
    wrecked.loc[wrecked.index >= 20, "y_true"] *= 3  # later actuals change drastically
    again = ex_ante_confidence(hist, wrecked, price, CFG)
    early = base["decision_date"] < hist.loc[20, "target_date"]
    pd.testing.assert_series_equal(base.loc[early, "p_within_10"], again.loc[early, "p_within_10"])


def test_model_spread_excludes_naive() -> None:
    rows = pd.concat(
        [
            _rows().iloc[[0]].assign(model="naive", y_pred=100.0),
            _rows().iloc[[0]].assign(model="a", y_pred=104.0),
            _rows().iloc[[0]].assign(model="b", y_pred=110.0),
        ]
    )
    assert model_spread(rows).iloc[0] == pytest.approx(6.0)


def test_confidence_check_and_calibration() -> None:
    hist, price = _history(40)
    scored = outcomes(ex_ante_confidence(hist, hist, price, CFG), [5, 10])
    check = confidence_check(scored)
    assert check["n"].sum() == len(scored)
    assert set(check["confidence"]) <= {"High", "Medium", "Low", NOT_ENOUGH}
    assert probability_calibration(scored)["n"].sum() <= len(scored)
    assert 0 <= brier(scored) <= 1
    assert np.isnan(brier(scored.iloc[:0]))


def test_explain_mentions_numbers_and_advice() -> None:
    row = _rows().iloc[0].copy()
    row["p_within_10"], row["p_within_5"], row["p_up"] = 0.72, 0.4, 0.66
    row["model_spread_pct"], row["confidence"] = 3.0, "High"
    text = explain(row, "XGBoost")
    assert "XGBoost" in text and "US$102/t" in text and "72 %" in text and "HIGH" in text
    assert "agree closely" in text and "coin flip" not in text
    assert "coin flip" in explain(row, "XGBoost", direction_reliable=False)
    row["confidence"], row["p_within_10"] = NOT_ENOUGH, np.nan
    assert "not yet enough" in explain(row, "XGBoost")
    flat = _rows().iloc[2].copy()
    flat["y_pred"] = 99.8  # within 0.5 % of 100: "stay close"
    flat["confidence"] = NOT_ENOUGH
    assert "stay close" in explain(flat, "ARIMA")


def test_fixed_grid() -> None:
    best = {"1": {"depth": 3, "lr": 0.1}, "3": {"depth": 2, "lr": 0.1}}
    assert fixed_grid(best) == {"depth": [3, 2], "lr": [0.1]}


def test_refit_walk_forward_learning_loop(history: pd.DataFrame, project: ProjectConfig) -> None:
    hist = history.loc[:"2012-12"]
    first = pd.Timestamp("2012-06-01")

    def make() -> object:
        return build_model("naive", project)

    out = refit_walk_forward(make, hist, first, [1, 3])  # type: ignore[arg-type]
    assert (out["target_date"] >= first).all()
    assert (out["fit_end"] == out["decision_date"]).all()
    live = out[out["y_true"].isna()]
    assert set(live["target_date"]) == set(
        pd.to_datetime(["2013-01-01", "2013-02-01", "2013-03-01"])
    )
    row = out[out["y_true"].notna()].iloc[0]  # naive forecast = price at the decision month
    assert row["y_pred"] == pytest.approx(hist.loc[row["decision_date"], "price_usd_t"])

    done = set(zip(out["decision_date"], out["horizon"], strict=True))
    assert refit_walk_forward(make, hist, first, [1, 3], skip=done).empty  # type: ignore[arg-type]

    # A month later the stored live forecast is kept and gains its actual price.
    longer = history.loc[:"2013-01"]
    new = refit_walk_forward(make, longer, first, [1, 3], skip=done)  # type: ignore[arg-type]
    merged = fill_actuals(merge_predictions(out, new), longer["price_usd_t"])
    jan = merged[(merged["target_date"] == "2013-01-01") & (merged["horizon"] == 1)].iloc[0]
    assert jan["y_pred"] == pytest.approx(live[live["horizon"] == 1]["y_pred"].iloc[0])
    assert jan["y_true"] == pytest.approx(longer.loc["2013-01-01", "price_usd_t"])
    assert merge_predictions(None, new).equals(
        new.sort_values(["model", "horizon", "decision_date"]).reset_index(drop=True)
    )

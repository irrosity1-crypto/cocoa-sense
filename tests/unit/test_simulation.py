"""Simulation tests. All prices here are SYNTHETIC hand-computed examples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cocoasense.simulation.engine import revenue, scenario_delivery, simulate
from cocoasense.simulation.futures_provider import CsvFuturesProvider, SpotProxyProvider
from cocoasense.simulation.metrics import strategy_metrics, worst_share_mean
from cocoasense.simulation.strategies import (
    FIXED_STRATEGIES,
    DecisionContext,
    Fixed5050,
    ForecastDriven,
    SellForward,
    Wait,
)

F = 5000.0  # SYNTHETIC forward price


@pytest.mark.parametrize(
    ("delivery", "w", "expected"),
    [
        (6000, 1.0, 5000),
        (5000, 1.0, 5000),
        (4000, 1.0, 5000),
        (6000, 0.0, 6000),
        (5000, 0.0, 5000),
        (4000, 0.0, 4000),
        (6000, 0.5, 5500),
        (5000, 0.5, 5000),
        (4000, 0.5, 4500),
    ],
)
def test_revenue_hand_computed_no_financing(delivery: float, w: float, expected: float) -> None:
    assert revenue(w, F, delivery, 1.0, 0.0, 3) == pytest.approx(expected)


def test_financing_cost() -> None:
    # 10% a year for 3 months on the unsold half of 1 t valued at 5,000: 0.1*0.25*0.5*5000=62.5
    assert revenue(0.5, F, 6000, 1.0, 0.10, 3) == pytest.approx(5500 - 62.5)
    assert revenue(1.0, F, 6000, 1.0, 0.10, 3) == pytest.approx(5000)  # nothing to finance


def test_revenue_linear_in_quantity() -> None:
    assert revenue(0.3, F, 4200, 250.0, 0.08, 3) == pytest.approx(
        250 * revenue(0.3, F, 4200, 1.0, 0.08, 3)
    )
    with pytest.raises(ValueError):
        revenue(1.5, F, 4000, 1, 0, 3)


def _ctx(forecast: float, lower: float, upper: float) -> DecisionContext:
    return DecisionContext(pd.Timestamp("2020-01-01"), 3, F, forecast, lower, upper)


def test_forecast_driven_rule() -> None:
    assert ForecastDriven(0.0).forward_share(_ctx(9000, 8000, 10000)) == 0.5  # a=0 -> 50/50
    # forecast below forward (expects fall): z = (5000-4000)/1000 = 1 -> w = 0.5 + 0.25
    assert ForecastDriven(0.25).forward_share(_ctx(4000, 3000, 5000)) == pytest.approx(0.75)
    # forecast above forward (expects rise) -> waits more
    assert ForecastDriven(0.25).forward_share(_ctx(6000, 5000, 7000)) == pytest.approx(0.25)
    assert ForecastDriven(10).forward_share(_ctx(1000, 900, 1100)) == 1.0  # clipped
    assert ForecastDriven(1).forward_share(_ctx(4000, 4000, 4000)) == 0.5  # zero width
    with pytest.raises(ValueError):
        ForecastDriven(-1)
    assert [s.forward_share(_ctx(1, 0, 2)) for s in (SellForward(), Wait(), Fixed5050())] == [
        1.0,
        0.0,
        0.5,
    ]


def _forecasts() -> pd.DataFrame:
    d = pd.date_range("2020-01", periods=3, freq="MS")
    return pd.DataFrame(
        {
            "decision_date": d,
            "target_date": d + pd.DateOffset(months=3),
            "horizon": 3,
            "y_true": [6000.0, 5000.0, 4000.0],
            "y_pred": [5500.0, 5000.0, 4500.0],
            "lower": [5000.0, 4500.0, 4000.0],
            "upper": [6000.0, 5500.0, 5000.0],
        }
    )


def test_engine_and_metrics() -> None:
    fc = _forecasts()
    provider = SpotProxyProvider(pd.Series(F, index=fc["decision_date"]))
    res = simulate(fc, [*FIXED_STRATEGIES, ForecastDriven(0.5)], provider, 2.0, 0.0)
    sf = res[res["strategy"] == "sell_forward"]["revenue_per_t"]
    assert (sf == F).all()
    wait = res[res["strategy"] == "wait"]["revenue_per_t"].tolist()
    assert wait == [6000, 5000, 4000]
    m = strategy_metrics(res).set_index("strategy")
    assert m.loc["sell_forward", "mean_revenue_per_t"] == pytest.approx(5000)
    # best fixed per date: 6000 (wait), 5000 (all tie), 5000 (sell forward)
    assert m.loc["fixed_50_50", "regret_vs_best_fixed"] == pytest.approx((500 + 0 + 500) / 3)
    assert m.loc["wait", "win_rate_vs_5050"] == pytest.approx(1 / 3)
    # Property: when the rule saturates at w=1 it equals SellForward; at w=0 it equals Wait.
    extreme = simulate(fc, [ForecastDriven(100.0)], provider, 2.0, 0.0)
    assert extreme.loc[2, "w"] == 1.0 and extreme.loc[2, "revenue_per_t"] == pytest.approx(F)
    assert extreme.loc[0, "w"] == 0.0 and extreme.loc[0, "revenue_per_t"] == pytest.approx(6000)


def test_stress_scenario_overrides_delivery() -> None:
    fc = _forecasts()
    provider = SpotProxyProvider(pd.Series(F, index=fc["decision_date"]))
    up = scenario_delivery(fc, provider, 1.2)
    res = simulate(fc, [Wait()], provider, 1.0, 0.0, delivery_override=up)
    assert res["revenue_per_t"].tolist() == pytest.approx([6000.0] * 3)


def test_worst_share() -> None:
    assert worst_share_mean(pd.Series(range(1, 21)).to_numpy(float), 0.10) == pytest.approx(1.5)


def test_csv_futures_provider(tmp_path: Path) -> None:
    """SYNTHETIC futures settlements."""
    path = tmp_path / "fut.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-15", "2020-01-31", "2020-01-31", "2020-02-28"],
            "contract_month": ["2020-03", "2020-03", "2020-05", "2020-05"],
            "settle": [2500, 2550, 2600, 2700],
        }
    ).to_csv(path, index=False)
    p = CsvFuturesProvider(path)
    assert p.forward_price(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")) == 2550
    assert p.forward_price(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")) == 2600
    with pytest.raises(LookupError):
        p.forward_price(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-01"))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "settle": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        CsvFuturesProvider(bad)

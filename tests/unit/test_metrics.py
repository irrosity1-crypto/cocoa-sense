from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.evaluation.metrics import assign_periods, diebold_mariano, point_metrics, score
from cocoasense.evaluation.reports import leaderboard, to_markdown, write_table


def _df(y_true: list[float], y_pred: list[float], last: list[float]) -> pd.DataFrame:
    n = len(y_true)
    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "last_obs": last,
            "lower": [p - 50 for p in y_pred],
            "upper": [p + 50 for p in y_pred],
        },
        index=range(n),
    )


def test_point_metrics_hand_computed() -> None:
    # SYNTHETIC numbers
    m = point_metrics(_df([100, 200], [110, 180], [100, 190]))
    assert m["mae"] == pytest.approx(15)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mape"] == pytest.approx((10 + 10) / 2)
    assert m["within_10pct"] == 1.0 and m["within_5pct"] == 0.0
    # row 1: predicted up from 100, actual flat (0) -> wrong; row 2: predicted down, actual up
    assert m["directional_acc"] == 0.0
    assert m["coverage"] == 1.0


def test_directional_excludes_no_change() -> None:
    m = point_metrics(_df([110, 90], [100, 100], [100, 100]))
    assert np.isnan(m["directional_acc"])


def test_diebold_mariano() -> None:
    rng = np.random.default_rng(1)
    e = rng.standard_normal(60)
    stat, p = diebold_mariano(e * 0.5, e, 1)
    assert stat < 0 and p < 0.05
    assert np.isnan(diebold_mariano(e[:5], e[:5], 1)[0])


def _preds() -> pd.DataFrame:
    dates = pd.date_range("2020-01", periods=24, freq="MS")
    rows = []
    for model, bias in (("naive", 0.0), ("good", 0.0)):
        for i, d in enumerate(dates):
            y = 1000 + 10 * i
            pred = y - 30 if model == "naive" else y - 10
            rows.append(
                {
                    "stage": "test",
                    "model": model,
                    "decision_date": d,
                    "target_date": d + pd.DateOffset(months=1),
                    "horizon": 1,
                    "last_obs": y - 30,
                    "y_true": y,
                    "y_pred": pred + bias,
                    "lower": pred - 40,
                    "upper": pred + 40,
                }
            )
    return pd.DataFrame(rows)


def test_score_relative_to_naive(tmp_path: object) -> None:
    periods = {
        "early": (pd.Timestamp("2020-01"), pd.Timestamp("2020-12")),
        "late": (pd.Timestamp("2021-01"), None),
    }
    table = score(assign_periods(_preds(), periods))
    full = table[table["period"] == "full"].set_index("model")
    assert full.loc["naive", "rel_mae"] == pytest.approx(1.0)
    assert full.loc["good", "rel_mae"] == pytest.approx(1 / 3)
    assert set(table["period"]) == {"full", "early", "late"}
    lb = leaderboard(table, "test", "full", 1)
    assert lb.iloc[0]["model"] == "good"
    assert "| model |" in to_markdown(lb)


def test_write_table(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "t.csv"
    write_table(pd.DataFrame({"a": [1.0, np.nan]}), p, "Title")
    assert p.exists() and "## Title" in p.with_suffix(".md").read_text(encoding="utf-8")

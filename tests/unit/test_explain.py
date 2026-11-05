"""Forecast explanations add up to the model's own prediction (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.config import ProjectConfig
from cocoasense.evaluation.explain import contributions, describe_value, grouped
from cocoasense.features.build import build_features
from cocoasense.models.registry import build_model


@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_contributions_sum_to_prediction(
    history: pd.DataFrame, project: ProjectConfig, name: str
) -> None:
    hist = history.loc[:"2009-12"]
    model = build_model(name, project, pd.Timestamp("1985-01"))
    model.fit(hist)
    for h in project.features.horizons:
        table = contributions(model, hist, h)  # type: ignore[arg-type]
        row = build_features(hist.iloc[-30:], None, project.features).iloc[[-1]]
        direct = float(model.model_for(h).predict(row[model.columns])[0])  # type: ignore[attr-defined]
        assert table["prediction"].iloc[0] == pytest.approx(direct, abs=1e-5)
        assert table["base"].iloc[0] + table["contribution"].sum() == pytest.approx(
            direct, abs=1e-5
        )
        fc = model.predict(hist, h)
        assert fc.point == pytest.approx(hist["price_usd_t"].iloc[-1] * np.exp(direct), rel=1e-6)


def test_grouped_and_wording() -> None:
    table = pd.DataFrame(
        {
            "feature": ["log_price", "month_sin", "month_cos", "ret_3m"],
            "label": [
                "Price level",
                "Time of year",
                "Time of year",
                "Change over the last 3 months",
            ],
            "value": [np.log(5000), 0.5, 0.2, np.log(1.1)],
            "contribution": [-0.05, 0.01, 0.005, 0.02],
        }
    )
    g = grouped(table)
    assert list(g["label"]) == ["Price level", "Change over the last 3 months", "Time of year"]
    assert g.loc[g["label"] == "Time of year", "contribution"].iloc[0] == pytest.approx(0.015)
    assert describe_value("log_price", np.log(5000)) == "US$5,000/t"
    assert describe_value("ret_3m", np.log(1.1)) == "+10.0%"
    assert describe_value("vol_6m", 0.065) == "6.5% a month"
    assert describe_value("month_sin", 0.3) == ""

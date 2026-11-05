from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.config import ProjectConfig
from cocoasense.models.base import Forecaster, NotFittedError
from cocoasense.models.registry import build_model
from cocoasense.models.tree import conformal_quantile

CUT = pd.Timestamp("2009-12-01")


@pytest.mark.parametrize("name", ["naive", "arima", "random_forest", "xgboost"])
def test_models_fit_and_predict(name: str, project: ProjectConfig, history: pd.DataFrame) -> None:
    model = build_model(name, project)
    assert isinstance(model, Forecaster)
    with pytest.raises((NotFittedError, KeyError)):
        model.predict(history.loc[:CUT], 1)
    model.fit(history.loc[:CUT])
    for h in (1, 3):
        fc = model.predict(history.loc[:"2012-06"], h)
        assert fc.decision_date == pd.Timestamp("2012-06-01")
        assert np.isfinite(fc.point) and fc.point > 0
        assert fc.lower is not None and fc.upper is not None
        assert fc.lower < fc.upper
    info = model.info()
    assert info["train_target_max"] == "2009-12-01"


def test_naive_is_last_price(project: ProjectConfig, history: pd.DataFrame) -> None:
    m = build_model("naive", project)
    m.fit(history.loc[:CUT])
    fc = m.predict(history.loc[:"2011-03"], 3)
    assert fc.point == pytest.approx(history.loc["2011-03-01", "price_usd_t"])
    assert fc.lower < fc.point < fc.upper  # type: ignore[operator]


def test_label_and_unknown_model(project: ProjectConfig) -> None:
    assert build_model("naive", project, label="naive_x").name == "naive_x"
    with pytest.raises(KeyError):
        build_model("transformer", project)


def test_conformal_quantile() -> None:
    res = np.arange(1, 11, dtype=float)  # 1..10
    assert conformal_quantile(res, 0.8) == 9.0  # ceil(11*0.8)=9 -> 9th smallest
    assert conformal_quantile(res, 0.99) == 10.0
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.8)


def test_tree_reports_label_window(project: ProjectConfig, history: pd.DataFrame) -> None:
    m = build_model("xgboost", project)
    m.fit(history.loc[:CUT])
    info = m.info()
    assert all(pd.Timestamp(d) <= CUT for d in info["last_label_date"].values())
    assert set(info["best_params"]) == {"1", "3"}

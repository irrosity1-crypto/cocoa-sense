"""LSTM forecaster unit tests (SYNTHETIC data, tiny networks)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from cocoasense.config import LstmConfig, ProjectConfig
from cocoasense.models.base import NotFittedError
from cocoasense.models.lstm import LstmForecaster, sequences
from cocoasense.models.registry import build_model
from tests.conftest import synthetic_history

CUT = pd.Timestamp("2009-12-01")


def test_sequences_end_at_their_row() -> None:
    frame = np.arange(20, dtype=float).reshape(10, 2)
    out = sequences(frame, np.array([3, 9]), lookback=4)
    assert out.shape == (2, 4, 2)
    np.testing.assert_array_equal(out[0], frame[0:4])
    np.testing.assert_array_equal(out[1], frame[6:10])
    with pytest.raises(ValueError, match="before the first row"):
        sequences(frame, np.array([2]), lookback=4)


def test_fit_predict(history: pd.DataFrame, project: ProjectConfig) -> None:
    assert project.lstm is not None
    model = build_model("lstm", project, pd.Timestamp("1985-01"))
    with pytest.raises(NotFittedError):
        model.predict(history.loc[:CUT], 1)
    model.fit(history.loc[:CUT])
    fc = model.predict(history.loc[:"2011-06"], 3)
    assert fc.decision_date == pd.Timestamp("2011-06-01") and fc.horizon == 3
    assert fc.lower is not None and fc.upper is not None
    assert 0 < fc.lower < fc.point < fc.upper
    info = model.info()
    assert pd.Timestamp(info["train_target_max"]) == CUT
    assert all(pd.Timestamp(d) <= CUT for d in info["last_label_date"].values())
    assert info["best_params"]["1"]["inputs"] in project.lstm.input_sets


def test_refit_is_bit_for_bit_reproducible(history: pd.DataFrame, project: ProjectConfig) -> None:
    a = build_model("lstm", project, pd.Timestamp("1985-01"))
    b = build_model("lstm", project, pd.Timestamp("1985-01"))
    a.fit(history.loc[:CUT])
    b.fit(history.loc[:CUT])
    assert a.predict(history, 1) == b.predict(history, 1)
    assert a.info() == b.info()


def test_rejects_unknown_inputs_and_short_history(project: ProjectConfig) -> None:
    assert project.lstm is not None
    bad = project.lstm.model_copy(update={"input_sets": [["ret_1m", "not_a_feature"]]})
    with pytest.raises(ValueError, match="not_a_feature"):
        LstmForecaster("lstm", bad, project.features, [1], pd.Timestamp("1985-01"), 0.8, 3, 0)
    model = build_model("lstm", project, pd.Timestamp("1985-01"))
    with pytest.raises(ValueError, match="training rows"):
        model.fit(synthetic_history(start="1984-01", end="1986-12"))


def test_incomplete_window_at_predict(history: pd.DataFrame, project: ProjectConfig) -> None:
    model = build_model("lstm", project, pd.Timestamp("1985-01"))
    model.fit(history.loc[:CUT])
    with pytest.raises(ValueError, match="incomplete input window"):
        model.predict(history.iloc[:4], 1)


def test_requires_lstm_config(project: ProjectConfig) -> None:
    with pytest.raises(FileNotFoundError, match=r"lstm\.yaml"):
        build_model("lstm", replace(project, lstm=None))


BASE_LSTM: dict[str, object] = {
    "input_sets": [["ret_1m"]],
    "lookback": [6],
    "hidden_size": [4],
    "epochs": 5,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "n_seeds": 1,
}


@pytest.mark.parametrize(
    "update",
    [{"input_sets": []}, {"input_sets": [[]]}, {"lookback": [0]}, {"hidden_size": []}],
)
def test_config_validation(update: dict[str, object]) -> None:
    LstmConfig.model_validate(BASE_LSTM)
    with pytest.raises(ValidationError):
        LstmConfig.model_validate({**BASE_LSTM, **update})

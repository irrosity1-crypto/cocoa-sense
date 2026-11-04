"""Common contracts for all forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class Forecast:
    """A forecast of the monthly average price ``horizon`` months after ``decision_date``.

    ``decision_date`` is the month-start label of the last month whose average is known; the
    decision itself is taken at the end of that month. Prices are in USD per tonne.
    """

    decision_date: pd.Timestamp
    horizon: int
    point: float
    lower: float | None
    upper: float | None


@runtime_checkable
class Forecaster(Protocol):
    """Interface every model implements.

    ``fit`` receives history truncated at the training cut-off and learns parameters from it.
    ``predict`` receives history up to the decision date and must not change any parameter.
    """

    name: str

    def fit(self, history: pd.DataFrame) -> None:
        """Learn parameters from ``history`` (all rows are training data)."""
        ...

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        """Forecast from the last row of ``history`` using frozen parameters."""
        ...

    def info(self) -> dict[str, Any]:
        """Fitted details for logging (chosen order or hyperparameters, training window)."""
        ...


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


def last_price(history: pd.DataFrame) -> tuple[pd.Timestamp, float]:
    """Decision date and price of the last row of ``history``."""
    if history.empty:
        raise ValueError("history is empty")
    return pd.Timestamp(history.index[-1]), float(history["price_usd_t"].iloc[-1])

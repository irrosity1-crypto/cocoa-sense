"""Naive baseline: the forecast for every horizon is the last observed monthly price."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cocoasense.features.build import build_targets
from cocoasense.models.base import Forecast, NotFittedError, last_price


class NaiveForecaster:
    """Random-walk forecast with an empirical interval.

    The interval uses quantiles of historical h-month log returns from the training window,
    so it is fitted on training data only.
    """

    name = "naive"

    def __init__(self, horizons: list[int], train_start: pd.Timestamp, level: float) -> None:
        self.horizons = horizons
        self.train_start = train_start
        self.level = level
        self._quantiles: dict[int, tuple[float, float]] = {}
        self._train_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        """Store empirical return quantiles for each horizon."""
        targets = build_targets(history, self.horizons)
        alpha = (1 - self.level) / 2
        for h in self.horizons:
            r = targets[f"target_{h}m"].loc[self.train_start :].dropna()
            if r.size < 12:
                raise ValueError(f"too few training returns for horizon {h}: {r.size}")
            self._quantiles[h] = (float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha)))
        self._train_end = pd.Timestamp(history.index[-1])

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        """Last price as the point forecast."""
        if horizon not in self._quantiles:
            raise NotFittedError(f"naive not fitted for horizon {horizon}")
        date, p = last_price(history)
        lo, hi = self._quantiles[horizon]
        return Forecast(date, horizon, p, p * float(np.exp(lo)), p * float(np.exp(hi)))

    def info(self) -> dict[str, Any]:
        """Training window and return quantiles."""
        return {
            "train_start": str(self.train_start.date()),
            "train_target_max": str(self._train_end.date()) if self._train_end else None,
            "return_quantiles": {str(h): q for h, q in self._quantiles.items()},
        }

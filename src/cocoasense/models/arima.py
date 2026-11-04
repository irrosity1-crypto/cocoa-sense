"""ARIMA on log price with the order chosen by AIC inside the training window.

After fitting, parameters are frozen: forecasts at later dates run the Kalman filter over the
newly observed data with ``refit=False``, so new observations update the state but never the
coefficients.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from cocoasense.config import ArimaConfig
from cocoasense.models.base import Forecast, NotFittedError, last_price

logger = logging.getLogger(__name__)


def _log_series(history: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    s = np.log(history["price_usd_t"].loc[start:].astype(float))
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    return s


class ArimaForecaster:
    """ARIMA(p, d, q) on log price; order selected by AIC on training data only."""

    name = "arima"

    def __init__(self, cfg: ArimaConfig, train_start: pd.Timestamp, level: float) -> None:
        self.cfg = cfg
        self.train_start = train_start
        self.level = level
        self._result: SARIMAXResults | None = None
        self._order: tuple[int, int, int] | None = None
        self._aic: float | None = None
        self._train_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        """Fit every candidate order and keep the lowest AIC."""
        y = _log_series(history, self.train_start)
        best: tuple[float, tuple[int, int, int], SARIMAXResults] | None = None
        for p, q in itertools.product(range(self.cfg.max_p + 1), range(self.cfg.max_q + 1)):
            order = (p, self.cfg.d, q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = SARIMAX(y, order=order, trend=self.cfg.trend).fit(disp=False)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logger.debug("ARIMA%s failed: %s", order, exc)
                    continue
            if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                best = (float(res.aic), order, res)
        if best is None:
            raise RuntimeError("no ARIMA order could be fitted")
        self._aic, self._order, self._result = best
        self._train_end = pd.Timestamp(history.index[-1])
        logger.info("ARIMA order %s selected (AIC %.1f)", self._order, self._aic)

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        """Filter new data with frozen parameters and forecast ``horizon`` steps ahead."""
        if self._result is None:
            raise NotFittedError("ARIMA not fitted")
        date, _ = last_price(history)
        y = _log_series(history, self.train_start)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self._result.apply(y, refit=False)
            fc = applied.get_forecast(steps=horizon)
        mean = float(fc.predicted_mean.iloc[-1])
        ci = fc.conf_int(alpha=1 - self.level).iloc[-1]
        return Forecast(
            date, horizon, float(np.exp(mean)), float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))
        )

    def info(self) -> dict[str, Any]:
        """Selected order, AIC and training window."""
        params = {} if self._result is None else self._result.params.round(6).to_dict()
        return {
            "train_start": str(self.train_start.date()),
            "train_target_max": str(self._train_end.date()) if self._train_end else None,
            "order": list(self._order) if self._order else None,
            "aic": self._aic,
            "params": params,
        }

"""Random Forest and XGBoost forecasters on lagged features, with split-conformal intervals.

For each horizon the model predicts the h-month log return; the price forecast is
``P_t * exp(prediction)``. Hyperparameters are chosen by grid search with ``TimeSeriesSplit``
(gap = horizon, so no CV training label overlaps a CV test month) inside the training rows only.
The interval half-width is the conformal quantile of out-of-fold absolute residuals from the
same splits, again using training rows only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from cocoasense.config import FeatureConfig
from cocoasense.features.build import build_features, feature_columns, training_table
from cocoasense.models.base import Forecast, NotFittedError, last_price

logger = logging.getLogger(__name__)

EstimatorFactory = Callable[[int], RegressorMixin]


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n+1)·level)/n."""
    n = abs_residuals.size
    if n == 0:
        raise ValueError("no residuals for conformal calibration")
    rank = min(n, math.ceil((n + 1) * level))
    return float(np.sort(abs_residuals)[rank - 1])


class TreeForecaster:
    """Shared fit/predict logic for tree ensembles."""

    def __init__(
        self,
        name: str,
        factory: EstimatorFactory,
        grid: dict[str, list[Any]],
        features: FeatureConfig,
        horizons: list[int],
        train_start: pd.Timestamp,
        level: float,
        cv_splits: int,
        seed: int,
    ) -> None:
        self.name = name
        self.factory = factory
        self.grid = grid
        self.features = features
        self.horizons = horizons
        self.train_start = train_start
        self.level = level
        self.cv_splits = cv_splits
        self.seed = seed
        self._models: dict[int, RegressorMixin] = {}
        self._q: dict[int, float] = {}
        self._best: dict[int, dict[str, Any]] = {}
        self._n_train: dict[int, int] = {}
        self._train_end: pd.Timestamp | None = None
        self._last_label_date: dict[int, pd.Timestamp] = {}

    @property
    def columns(self) -> list[str]:
        return feature_columns(self.features)

    def fit(self, history: pd.DataFrame) -> None:
        """Tune, calibrate and fit one estimator per horizon on training rows only."""
        for h in self.horizons:
            X, y = training_table(history, self.features, h, self.train_start)
            if len(X) < 10 * self.cv_splits:
                raise ValueError(f"{self.name}: only {len(X)} training rows for horizon {h}")
            cv = TimeSeriesSplit(n_splits=self.cv_splits, gap=h)
            search = GridSearchCV(
                self.factory(self.seed),
                self.grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                refit=True,
            )
            search.fit(X, y)
            best = search.best_estimator_
            residuals = []
            for tr, te in cv.split(X):
                m = clone(best).fit(X.iloc[tr], y.iloc[tr])
                residuals.append(np.abs(y.iloc[te].to_numpy() - m.predict(X.iloc[te])))
            self._q[h] = conformal_quantile(np.concatenate(residuals), self.level)
            self._models[h] = best
            self._best[h] = {k: _plain(v) for k, v in search.best_params_.items()}
            self._n_train[h] = len(X)
            self._last_label_date[h] = pd.Timestamp(X.index[-1]) + pd.DateOffset(months=h)
            logger.info(
                "%s h=%d best %s (CV MAE %.4f)", self.name, h, self._best[h], -search.best_score_
            )
        self._train_end = pd.Timestamp(history.index[-1])

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        """Predict the h-month log return from the last feature row and convert to a price."""
        model = self._models.get(horizon)
        if model is None:
            raise NotFittedError(f"{self.name} not fitted for horizon {horizon}")
        date, p = last_price(history)
        row = build_features(
            history.iloc[-(self.features.max_lookback + 2) :], None, self.features
        ).iloc[[-1]]
        if row.isna().any(axis=None):
            raise ValueError(f"{self.name}: incomplete features at {date:%Y-%m}")
        r = float(model.predict(row[self.columns])[0])
        q = self._q[horizon]
        return Forecast(date, horizon, p * math.exp(r), p * math.exp(r - q), p * math.exp(r + q))

    def model_for(self, horizon: int) -> RegressorMixin:
        """Fitted estimator for one horizon (used for permutation importance)."""
        return self._models[horizon]

    def info(self) -> dict[str, Any]:
        """Best hyperparameters, conformal half-widths and training window."""
        return {
            "train_start": str(self.train_start.date()),
            "train_target_max": str(self._train_end.date()) if self._train_end else None,
            "last_label_date": {str(h): str(d.date()) for h, d in self._last_label_date.items()},
            "features": self.columns,
            "best_params": {str(h): p for h, p in self._best.items()},
            "conformal_halfwidth_logret": {str(h): q for h, q in self._q.items()},
            "n_train_rows": {str(h): n for h, n in self._n_train.items()},
        }


def _plain(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def random_forest_factory(seed: int) -> RegressorMixin:
    """Seeded Random Forest regressor.

    Single-threaded on purpose: with ``n_jobs=-1`` the per-tree predictions are summed in a
    thread-dependent order, so refits differ in the last floating-point digit and reruns are not
    bit-for-bit reproducible.
    """
    return RandomForestRegressor(random_state=seed, n_jobs=1)


def xgboost_factory(seed: int) -> RegressorMixin:
    """Seeded XGBoost regressor (histogram method, single thread for reproducibility)."""
    return XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)

"""LSTM forecaster (stretch model) with split-conformal intervals.

Same task as the tree models: for each horizon h the network reads the last ``lookback`` months
of input channels (e.g. 1-month log return and log price) and predicts the h-month log return;
the price forecast is ``P_t * exp(prediction)``.

* Inputs and target are standardised with statistics of the rows being fitted only.
* Hyperparameters (input set, lookback, hidden size) are chosen by ``TimeSeriesSplit``
  (gap = horizon) inside the training rows, as for Random Forest and XGBoost. All candidates are
  scored on the same rows (those complete for the longest lookback).
* The interval half-width is the conformal quantile of out-of-fold absolute residuals of the
  chosen candidate.
* Each fit trains ``n_seeds`` independently seeded networks and averages them, single-threaded,
  so refits are bit-for-bit reproducible.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn

from cocoasense.config import FeatureConfig, LstmConfig
from cocoasense.features.build import build_features, build_targets, feature_columns
from cocoasense.models.base import Forecast, NotFittedError, last_price
from cocoasense.models.tree import conformal_quantile

logger = logging.getLogger(__name__)


class _Net(nn.Module):
    """One LSTM layer followed by a linear head on the last hidden state."""

    def __init__(self, n_inputs: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        result: torch.Tensor = self.head(out[:, -1, :]).squeeze(-1)
        return result


@dataclass(frozen=True)
class Candidate:
    """One point of the hyperparameter grid."""

    inputs: tuple[str, ...]
    lookback: int
    hidden: int

    def as_dict(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs), "lookback": self.lookback, "hidden_size": self.hidden}


@dataclass
class Ensemble:
    """Trained networks plus the scaling statistics they were trained with."""

    nets: list[_Net]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble-mean prediction in log-return units for windows ``X``."""
        xt = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            out = torch.stack([net(xt) for net in self.nets]).mean(dim=0).numpy()
        return np.asarray(out, dtype=float) * self.y_std + self.y_mean


@contextmanager
def _single_thread() -> Iterator[None]:
    """Run torch on one thread so the floating-point summation order is fixed."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def sequences(frame: np.ndarray, ends: np.ndarray, lookback: int) -> np.ndarray:
    """Stack windows ``frame[i - lookback + 1 : i + 1]`` for each end position ``i``.

    Row ``i`` of ``frame`` holds inputs known at the end of month ``i``, so a window ending at
    ``i`` uses nothing later than ``i``. Returns an array of shape (len(ends), lookback, n_inputs).
    """
    if (ends < lookback - 1).any():
        raise ValueError("window would start before the first row")
    offsets = np.arange(-lookback + 1, 1)
    return frame[ends[:, None] + offsets[None, :]]


class LstmForecaster:
    """LSTM on monthly input sequences, one small ensemble per horizon."""

    def __init__(
        self,
        name: str,
        cfg: LstmConfig,
        features: FeatureConfig,
        horizons: list[int],
        train_start: pd.Timestamp,
        level: float,
        cv_splits: int,
        seed: int,
    ) -> None:
        unknown = {c for s in cfg.input_sets for c in s} - set(feature_columns(features))
        if unknown:
            raise ValueError(f"LSTM inputs not produced by the feature builder: {sorted(unknown)}")
        self.name = name
        self.cfg = cfg
        self.features = features
        self.horizons = horizons
        self.train_start = train_start
        self.level = level
        self.cv_splits = cv_splits
        self.seed = seed
        self._fitted: dict[int, tuple[Candidate, Ensemble]] = {}
        self._q: dict[int, float] = {}
        self._cv_mae: dict[int, float] = {}
        self._n_train: dict[int, int] = {}
        self._last_label_date: dict[int, pd.Timestamp] = {}
        self._train_end: pd.Timestamp | None = None

    @property
    def candidates(self) -> list[Candidate]:
        return [
            Candidate(tuple(i), lb, hs)
            for i, lb, hs in itertools.product(
                self.cfg.input_sets, self.cfg.lookback, self.cfg.hidden_size
            )
        ]

    # ---------------------------------------------------------------- training

    def _train(self, X: np.ndarray, y: np.ndarray, hidden: int) -> Ensemble:
        """Standardise on (X, y) and train the seeded ensemble with full-batch Adam."""
        flat = X.reshape(-1, X.shape[-1])
        x_mean, x_std = flat.mean(axis=0), flat.std(axis=0)
        x_std[x_std == 0] = 1.0
        y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
        xt = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
        yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        loss_fn = nn.MSELoss()
        nets = []
        for k in range(self.cfg.n_seeds):
            torch.manual_seed(self.seed + k)
            net = _Net(X.shape[-1], hidden)
            opt = torch.optim.Adam(
                net.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
            )
            net.train()
            for _ in range(self.cfg.epochs):
                opt.zero_grad()
                loss_fn(net(xt), yt).backward()
                opt.step()
            net.eval()
            nets.append(net)
        return Ensemble(nets, x_mean, x_std, y_mean, y_std)

    def fit(self, history: pd.DataFrame) -> None:
        """Tune, calibrate and fit one ensemble per horizon on training rows only."""
        feats = build_features(history, None, self.features)
        with _single_thread():
            for h in self.horizons:
                self._fit_horizon(history, feats, h)
        self._train_end = pd.Timestamp(history.index[-1])

    def _fit_horizon(self, history: pd.DataFrame, feats: pd.DataFrame, h: int) -> None:
        y_all = build_targets(history, [h])[f"target_{h}m"].to_numpy()
        longest = max(self.cfg.lookback)
        all_inputs = sorted({c for s in self.cfg.input_sets for c in s})
        complete = pd.Series(feats[all_inputs].notna().all(axis=1).to_numpy(dtype=float))
        # Usable rows: on/after train_start, target known (so the label month is inside
        # ``history``), and every input present over the longest window.
        window_ok = complete.rolling(longest).sum().to_numpy() == longest
        ends = np.flatnonzero((feats.index >= self.train_start) & ~np.isnan(y_all) & window_ok)
        if len(ends) < 10 * self.cv_splits:
            raise ValueError(f"{self.name}: only {len(ends)} training rows for horizon {h}")
        y = y_all[ends]
        folds = list(TimeSeriesSplit(n_splits=self.cv_splits, gap=h).split(ends))

        best: tuple[float, Candidate, np.ndarray] | None = None
        for cand in self.candidates:
            X = sequences(feats[list(cand.inputs)].to_numpy(dtype=float), ends, cand.lookback)
            maes, residuals = [], []
            for tr, te in folds:
                err = np.abs(y[te] - self._train(X[tr], y[tr], cand.hidden).predict(X[te]))
                maes.append(float(err.mean()))
                residuals.append(err)
            score = float(np.mean(maes))
            logger.info("%s h=%d %s CV MAE %.5f", self.name, h, cand.as_dict(), score)
            if best is None or score < best[0]:  # ties keep the earlier (simpler) candidate
                best = (score, cand, np.concatenate(residuals))
        assert best is not None
        score, cand, oof = best
        X = sequences(feats[list(cand.inputs)].to_numpy(dtype=float), ends, cand.lookback)
        self._fitted[h] = (cand, self._train(X, y, cand.hidden))
        self._q[h] = conformal_quantile(oof, self.level)
        self._cv_mae[h] = score
        self._n_train[h] = len(ends)
        self._last_label_date[h] = pd.Timestamp(feats.index[ends[-1]]) + pd.DateOffset(months=h)
        logger.info("%s h=%d best %s (CV MAE %.5f)", self.name, h, cand.as_dict(), score)

    # -------------------------------------------------------------- prediction

    def predict(self, history: pd.DataFrame, horizon: int) -> Forecast:
        """Predict the h-month log return from the last ``lookback`` months and convert it."""
        if horizon not in self._fitted:
            raise NotFittedError(f"{self.name} not fitted for horizon {horizon}")
        cand, ensemble = self._fitted[horizon]
        date, p = last_price(history)
        tail = history.iloc[-(cand.lookback + self.features.max_lookback + 2) :]
        frame = build_features(tail, None, self.features)[list(cand.inputs)].to_numpy(dtype=float)
        window = frame[-cand.lookback :]
        if len(window) < cand.lookback or np.isnan(window).any():
            raise ValueError(f"{self.name}: incomplete input window at {date:%Y-%m}")
        with _single_thread():
            r = float(ensemble.predict(window[None, :, :])[0])
        q = self._q[horizon]
        return Forecast(date, horizon, p * math.exp(r), p * math.exp(r - q), p * math.exp(r + q))

    def info(self) -> dict[str, Any]:
        """Chosen hyperparameters, conformal half-widths and training window."""
        return {
            "train_start": str(self.train_start.date()),
            "train_target_max": str(self._train_end.date()) if self._train_end else None,
            "last_label_date": {str(h): str(d.date()) for h, d in self._last_label_date.items()},
            "best_params": {str(h): c.as_dict() for h, (c, _) in self._fitted.items()},
            "cv_mae_logret": {str(h): round(v, 6) for h, v in self._cv_mae.items()},
            "conformal_halfwidth_logret": {str(h): q for h, q in self._q.items()},
            "n_train_rows": {str(h): n for h, n in self._n_train.items()},
            "training": {
                "epochs": self.cfg.epochs,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "n_seeds": self.cfg.n_seeds,
            },
        }

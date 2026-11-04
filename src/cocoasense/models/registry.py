"""Create models by name from configuration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cocoasense.config import FeatureConfig, ProjectConfig
from cocoasense.models.arima import ArimaForecaster
from cocoasense.models.base import Forecaster
from cocoasense.models.naive import NaiveForecaster
from cocoasense.models.tree import TreeForecaster, random_forest_factory, xgboost_factory

MODEL_NAMES = ("naive", "arima", "random_forest", "xgboost")


def build_model(
    name: str,
    cfg: ProjectConfig,
    train_start: pd.Timestamp | None = None,
    features: FeatureConfig | None = None,
    label: str | None = None,
) -> Forecaster:
    """Instantiate a model.

    ``train_start`` and ``features`` override the defaults from config (used by validation
    experiments and by the frozen final configuration). ``label`` renames the model in outputs,
    e.g. ``random_forest_fx``.
    """
    start = train_start if train_start is not None else cfg.evaluation.ts("train_start")
    feats = features or cfg.features
    level = cfg.models.interval_level
    model: Any
    if name == "naive":
        model = NaiveForecaster(feats.horizons, start, level)
    elif name == "arima":
        model = ArimaForecaster(cfg.models.models.arima, start, level)
    elif name in ("random_forest", "xgboost"):
        grid_cfg = getattr(cfg.models.models, name)
        factory = random_forest_factory if name == "random_forest" else xgboost_factory
        model = TreeForecaster(
            name,
            factory,
            grid_cfg.grid,
            feats,
            feats.horizons,
            start,
            level,
            cfg.models.cv_splits,
            cfg.models.seed,
        )
    else:
        raise KeyError(f"unknown model '{name}'; choose from {MODEL_NAMES}")
    if label:
        model.name = label
    return model

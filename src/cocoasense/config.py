"""Validated configuration loaded from ``configs/*.yaml``.

Everything tunable lives in YAML; this module turns it into typed objects so that a typo in a
config file fails loudly at start-up instead of silently changing a result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- data


class SourceConfig(_Strict):
    """One downloadable source."""

    description: str
    url: str
    landing_page: str
    file: str
    licence: str


class ValidationThresholds(_Strict):
    """Plausibility thresholds used by the validation layer."""

    price_min_usd_t: float = Field(gt=0)
    price_max_usd_t: float = Field(gt=0)
    fx_min_ghs_per_usd: float = Field(gt=0)
    fx_max_ghs_per_usd: float = Field(gt=0)
    max_abs_monthly_log_change_warn: float = Field(gt=0)
    crosscheck_mean_tolerance_pct: float = Field(gt=0)
    crosscheck_month_flag_pct: float = Field(gt=0)


class DataConfig(_Strict):
    """Contents of ``configs/data.yaml``."""

    sources: dict[str, SourceConfig]
    primary_price: str
    crosscheck_price: str
    validation: ValidationThresholds

    @model_validator(mode="after")
    def _known_sources(self) -> DataConfig:
        for key in (self.primary_price, self.crosscheck_price):
            if key not in self.sources:
                raise ValueError(f"unknown source '{key}' in data.yaml")
        return self


# ----------------------------------------------------------------------- features


class FeatureConfig(_Strict):
    """Contents of ``configs/features.yaml``."""

    return_windows: list[int]
    rolling_windows: list[int]
    include_level: bool = True
    include_seasonality: bool = True
    use_fx: bool = False
    fx_return_windows: list[int] = [1, 3, 12]
    horizons: list[int] = [1, 3]

    @field_validator("return_windows", "rolling_windows", "fx_return_windows", "horizons")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("windows and horizons must be non-empty positive integers")
        return sorted(set(v))

    @field_validator("rolling_windows")
    @classmethod
    def _rolling_min(cls, v: list[int]) -> list[int]:
        if any(x < 2 for x in v):
            raise ValueError("rolling windows must be >= 2 to compute a volatility")
        return v

    @property
    def max_lookback(self) -> int:
        """Months of history needed before the first complete feature row."""
        windows = [*self.return_windows, *self.rolling_windows]
        if self.use_fx:
            windows += self.fx_return_windows
        return max(windows)


# ------------------------------------------------------------------------- models


class ArimaConfig(_Strict):
    """Order search space for ARIMA on log price."""

    max_p: int = Field(ge=0, le=5)
    max_q: int = Field(ge=0, le=5)
    d: int = Field(ge=0, le=2)
    trend: Literal["n", "c", "t", "ct"] = "n"


class GridConfig(_Strict):
    """Hyperparameter grid for a tree model."""

    grid: dict[str, list[Any]]


class ModelsSection(_Strict):
    naive: dict[str, Any] = {}
    arima: ArimaConfig
    random_forest: GridConfig
    xgboost: GridConfig


class LstmConfig(_Strict):
    """Contents of ``configs/lstm.yaml`` (stretch model, see docs/decisions.md D16).

    ``grid`` is searched with time-series cross-validation inside the training window, exactly
    like the tree models. ``input_sets`` name per-month input channels from
    :func:`cocoasense.features.build.build_features`.
    """

    input_sets: list[list[str]]
    lookback: list[int]
    hidden_size: list[int]
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    n_seeds: int = Field(ge=1)

    @field_validator("input_sets")
    @classmethod
    def _non_empty(cls, v: list[list[str]]) -> list[list[str]]:
        if not v or any(not s for s in v):
            raise ValueError("input_sets must be a non-empty list of non-empty lists")
        return v

    @field_validator("lookback", "hidden_size")
    @classmethod
    def _positive_list(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("lookback and hidden_size must be non-empty positive integers")
        return v


class ModelsConfig(_Strict):
    """Contents of ``configs/models.yaml``."""

    seed: int
    cv_splits: int = Field(ge=2)
    interval_level: float = Field(gt=0, lt=1)
    models: ModelsSection


# --------------------------------------------------------------------- evaluation


def _month(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).to_period("M").to_timestamp()


class EvaluationConfig(_Strict):
    """Contents of ``configs/evaluation.yaml``."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    final_train_end: str
    test_start: str
    test_end: str | None = None
    periods: dict[str, list[str | None]]
    surge_start: str
    train_start_candidates: list[str]

    @model_validator(mode="after")
    def _chronology(self) -> EvaluationConfig:
        ts = self.ts
        if not (ts("train_start") < ts("train_end") < ts("validation_start")):
            raise ValueError("require train_start < train_end < validation_start")
        if not (ts("validation_start") <= ts("validation_end") <= ts("final_train_end")):
            raise ValueError("require validation_start <= validation_end <= final_train_end")
        if not ts("final_train_end") < ts("test_start"):
            raise ValueError("final_train_end must be strictly before test_start (no overlap)")
        for name, bounds in self.periods.items():
            if len(bounds) != 2 or bounds[0] is None:
                raise ValueError(f"period '{name}' needs [start, end-or-null]")
        return self

    def ts(self, field: str) -> pd.Timestamp:
        """Return a month-start timestamp for a date field."""
        value = _month(getattr(self, field))
        if value is None:
            raise ValueError(f"{field} is not set")
        return value

    def period_bounds(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]:
        """Return named test sub-periods as (start, end-or-None) month timestamps."""
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {}
        for name, (start, end) in self.periods.items():
            start_ts = _month(start)
            assert start_ts is not None
            out[name] = (start_ts, _month(end))
        return out


# --------------------------------------------------------------------- simulation


class SimulationConfig(_Strict):
    """Contents of ``configs/simulation.yaml``."""

    horizon: int = Field(ge=1)
    quantity_t: float = Field(gt=0)
    financing_rate_annual: float = Field(ge=0)
    forward_price_mode: Literal["spot_proxy", "futures"] = "spot_proxy"
    futures_csv: str | None = None
    a_grid: list[float]
    scenarios: dict[str, float]
    sensitivity: dict[str, list[float]]


# ------------------------------------------------------------------------ project


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, rooted at the project directory."""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def manual(self) -> Path:
        return self.raw / "manual"

    @property
    def manifest(self) -> Path:
        return self.raw / "manifest.json"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def forecasts(self) -> Path:
        return self.artifacts / "forecasts"

    @property
    def metrics(self) -> Path:
        return self.artifacts / "metrics"

    @property
    def simulation(self) -> Path:
        return self.artifacts / "simulation"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @property
    def frozen(self) -> Path:
        return self.configs / "frozen.yaml"

    @property
    def frozen_lstm(self) -> Path:
        return self.configs / "frozen_lstm.yaml"

    def ensure(self) -> None:
        """Create output directories if they do not exist."""
        for p in (
            self.raw,
            self.manual,
            self.interim,
            self.processed,
            self.forecasts,
            self.metrics,
            self.simulation,
            self.figures,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectConfig:
    """All configuration for one run."""

    paths: Paths
    data: DataConfig
    features: FeatureConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    simulation: SimulationConfig
    lstm: LstmConfig | None = None  # optional stretch model (configs/lstm.yaml)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return content


def load_config(root: Path | None = None) -> ProjectConfig:
    """Load and validate every config file under ``<root>/configs``."""
    paths = Paths(root or PROJECT_ROOT)
    cfg_dir = paths.configs
    return ProjectConfig(
        paths=paths,
        data=DataConfig(**_read_yaml(cfg_dir / "data.yaml")),
        features=FeatureConfig(**_read_yaml(cfg_dir / "features.yaml")),
        models=ModelsConfig(**_read_yaml(cfg_dir / "models.yaml")),
        evaluation=EvaluationConfig(**_read_yaml(cfg_dir / "evaluation.yaml")),
        simulation=SimulationConfig(**_read_yaml(cfg_dir / "simulation.yaml")),
        lstm=LstmConfig(**_read_yaml(cfg_dir / "lstm.yaml"))
        if (cfg_dir / "lstm.yaml").exists()
        else None,
    )


def load_frozen(paths: Paths) -> dict[str, Any]:
    """Load ``configs/frozen.yaml`` (written by ``cocoasense freeze``)."""
    return _read_yaml(paths.frozen)


def load_frozen_lstm(paths: Paths) -> dict[str, Any]:
    """Load ``configs/frozen_lstm.yaml`` (written by ``cocoasense lstm freeze``)."""
    return _read_yaml(paths.frozen_lstm)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI use."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

"""Frozen-origin walk-forward forecasting.

Each model is fitted ONCE on history truncated at the training cut-off. It then forecasts every
evaluation month in turn, receiving the real history up to each decision date as input but
never being refitted. This enforces the project rule that no model is trained on data after the
cut-off, while still producing a realistic month-by-month forecast record.

The test stage is locked behind ``final=True`` and every final run is appended to a log.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cocoasense.ingestion.manifest import sha256_of
from cocoasense.models.base import Forecaster

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = [
    "stage",
    "model",
    "decision_date",
    "target_date",
    "horizon",
    "last_obs",
    "y_true",
    "y_pred",
    "lower",
    "upper",
    "in_surge_period",
]


class TestPeriodLockedError(RuntimeError):
    """Raised when the test period is requested without ``final=True``."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Window:
    """Training cut-off and the range of target months to forecast."""

    stage: str  # "validation" or "test"
    fit_end: pd.Timestamp  # last month of history (and of any label) the model is fitted on
    target_start: pd.Timestamp
    target_end: pd.Timestamp


def decision_dates(index: pd.DatetimeIndex, window: Window, horizon: int) -> list[pd.Timestamp]:
    """Decision months whose target month (t + h) falls inside the window and the data."""
    out = []
    for t in index:
        target = t + pd.DateOffset(months=horizon)
        if window.target_start <= target <= window.target_end and target in index:
            out.append(pd.Timestamp(t))
    return out


def forecast_window(
    model: Forecaster,
    history: pd.DataFrame,
    window: Window,
    horizons: Iterable[int],
    surge_start: pd.Timestamp,
    final: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit ``model`` on history up to ``window.fit_end`` and forecast the window.

    Returns a tidy prediction table and the model's fitted info (with run time).
    """
    if window.stage == "test" and not final:
        raise TestPeriodLockedError(
            "The test period is locked. Freeze the configuration, commit configs/frozen.yaml "
            "and rerun with --final."
        )
    if window.target_start <= window.fit_end:
        raise ValueError("evaluation targets must start after the training cut-off")
    start = time.perf_counter()
    train_history = history.loc[: window.fit_end]
    model.fit(train_history)
    fit_seconds = time.perf_counter() - start

    price = history["price_usd_t"]
    rows = []
    idx = pd.DatetimeIndex(history.index)
    for h in horizons:
        for t in decision_dates(idx, window, h):
            fc = model.predict(history.loc[:t], h)
            target = t + pd.DateOffset(months=h)
            rows.append(
                {
                    "stage": window.stage,
                    "model": model.name,
                    "decision_date": t,
                    "target_date": pd.Timestamp(target),
                    "horizon": h,
                    "last_obs": float(price.loc[t]),
                    "y_true": float(price.loc[target]),
                    "y_pred": fc.point,
                    "lower": np.nan if fc.lower is None else fc.lower,
                    "upper": np.nan if fc.upper is None else fc.upper,
                    "in_surge_period": bool(target >= surge_start),
                }
            )
    info = model.info()
    info["fit_seconds"] = round(fit_seconds, 2)
    info["total_seconds"] = round(time.perf_counter() - start, 2)
    info["fit_end"] = str(window.fit_end.date())
    _assert_no_training_leak(info, window)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), info


def _assert_no_training_leak(info: dict[str, Any], window: Window) -> None:
    """Guard: the model must report no training label later than the cut-off."""
    tmax = info.get("train_target_max")
    if tmax is not None and pd.Timestamp(tmax) > window.fit_end:
        raise AssertionError(f"model trained on data up to {tmax}, after cut-off {window.fit_end}")
    for d in (info.get("last_label_date") or {}).values():
        if pd.Timestamp(d) > window.fit_end:
            raise AssertionError(f"training label dated {d} is after cut-off {window.fit_end}")


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "configs", "src"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_final_run(log_path: Path, root: Path, frozen_path: Path, extra: dict[str, Any]) -> None:
    """Append one JSON line describing a final (test-period) run."""
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(root),
        "frozen_config_sha256": sha256_of(frozen_path) if frozen_path.exists() else None,
        **extra,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    logger.info("final run logged to %s", log_path)

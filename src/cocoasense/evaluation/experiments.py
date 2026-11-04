"""Development-stage experiments on the validation period, and the freeze step.

Everything here uses models fitted on data up to ``train_end`` (December 2015) and scored on
validation targets (2016–2019). The test period is never touched. ``choose_frozen`` turns the
validation results into ``configs/frozen.yaml``, which must be committed before ``--final``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cocoasense.config import FeatureConfig, ProjectConfig
from cocoasense.evaluation.metrics import score
from cocoasense.evaluation.walk_forward import Window, forecast_window
from cocoasense.features.build import build_features, build_targets, feature_columns
from cocoasense.models.registry import build_model
from cocoasense.models.tree import TreeForecaster
from cocoasense.simulation.engine import simulate
from cocoasense.simulation.futures_provider import SpotProxyProvider
from cocoasense.simulation.metrics import strategy_metrics
from cocoasense.simulation.strategies import ForecastDriven

logger = logging.getLogger(__name__)

FX_START = "1991-01"  # first month with 12 months of exchange-rate history (IMF series starts 1990)


@dataclass(frozen=True)
class Variant:
    """One model configuration evaluated on the validation period."""

    label: str
    model: str
    train_start: pd.Timestamp
    use_fx: bool
    experiment: str  # "train_start" or "fx"


def validation_window(cfg: ProjectConfig) -> Window:
    """Fit on targets up to train_end; forecast validation targets."""
    ev = cfg.evaluation
    return Window(
        "validation", ev.ts("train_end"), ev.ts("validation_start"), ev.ts("validation_end")
    )


def variants(cfg: ProjectConfig) -> list[Variant]:
    """All validation variants: training-start candidates for each model, plus the FX pair."""
    out = [Variant("naive", "naive", cfg.evaluation.ts("train_start"), False, "train_start")]
    for model in ("arima", "random_forest", "xgboost"):
        for start in cfg.evaluation.train_start_candidates:
            ts = pd.Timestamp(start)
            out.append(Variant(f"{model}@{ts:%Y}", model, ts, False, "train_start"))
    fx_ts = pd.Timestamp(FX_START)
    for model in ("random_forest", "xgboost"):
        out.append(Variant(f"{model}_nofx@{fx_ts:%Y}", model, fx_ts, False, "fx"))
        out.append(Variant(f"{model}_fx@{fx_ts:%Y}", model, fx_ts, True, "fx"))
    return out


def features_for(cfg: ProjectConfig, use_fx: bool) -> FeatureConfig:
    """Feature config with ``use_fx`` overridden."""
    return cfg.features.model_copy(update={"use_fx": use_fx})


def run_validation(
    cfg: ProjectConfig, history: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Fit every variant on the training window and forecast the validation period.

    Returns (predictions, fitted info by label, permutation importance table).
    """
    window = validation_window(cfg)
    preds, infos, importances = [], {}, []
    for v in variants(cfg):
        feats = features_for(cfg, v.use_fx)
        model = build_model(v.model, cfg, v.train_start, feats, label=v.label)
        logger.info("validation: fitting %s", v.label)
        p, info = forecast_window(
            model, history, window, feats.horizons, cfg.evaluation.ts("surge_start")
        )
        info.update({"experiment": v.experiment, "base_model": v.model, "use_fx": v.use_fx})
        preds.append(
            p.assign(
                experiment=v.experiment,
                base_model=v.model,
                use_fx=v.use_fx,
                train_start=v.train_start,
            )
        )
        infos[v.label] = info
        if isinstance(model, TreeForecaster):
            importances.append(_permutation_importance(model, history, window, feats, cfg))
    imp = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return pd.concat(preds, ignore_index=True), infos, imp


def _permutation_importance(
    model: TreeForecaster,
    history: pd.DataFrame,
    window: Window,
    feats: FeatureConfig,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """Permutation importance on validation rows (MAE of the log-return prediction)."""
    rows = []
    X_all = build_features(history, None, feats)
    for h in feats.horizons:
        y_all = build_targets(history, [h])[f"target_{h}m"]
        target_month = X_all.index + pd.DateOffset(months=h)
        mask = (target_month >= window.target_start) & (target_month <= window.target_end)
        X, y = X_all.loc[mask, feature_columns(feats)], y_all.loc[mask]
        keep = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[keep], y.loc[keep]
        res = permutation_importance(
            model.model_for(h),
            X,
            y,
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=cfg.models.seed,
        )
        for col, mean, std in zip(
            X.columns, res.importances_mean, res.importances_std, strict=True
        ):
            rows.append(
                {
                    "model": model.name,
                    "horizon": h,
                    "feature": col,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
    return pd.DataFrame(rows)


def score_validation(preds: pd.DataFrame) -> pd.DataFrame:
    """Validation metrics for every variant against the naive baseline."""
    return score(preds.assign(period="validation"))


def choose_frozen(
    cfg: ProjectConfig, preds: pd.DataFrame, history: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pick each model's training start and the simulation rule from validation results only.

    * Training start: lowest validation MAE averaged over horizons (same months for every
      candidate, so MAE is comparable).
    * Main models use price-only features so that they cover the whole test period (the
      exchange-rate series ends earlier); the FX question is answered by a separate paired test.
    * Simulation: the (driver model, a) pair with the highest mean validation revenue per tonne;
      ties go to the smaller a (closer to the 50/50 benchmark).
    """
    table = score_validation(preds)
    ts_rows = table[
        table["model"].isin(preds.loc[preds["experiment"] == "train_start", "model"].unique())
    ]
    chosen: dict[str, Any] = {"naive": {"train_start": cfg.evaluation.train_start, "use_fx": False}}
    for base in ("arima", "random_forest", "xgboost"):
        cand = ts_rows[ts_rows["model"].str.startswith(f"{base}@")]
        mean_mae = cand.groupby("model")["mae"].mean().sort_index()
        best_label = str(mean_mae.idxmin())
        start = preds.loc[preds["model"] == best_label, "train_start"].iloc[0]
        chosen[base] = {
            "train_start": f"{pd.Timestamp(start):%Y-%m}",
            "use_fx": False,
            "validation_label": best_label,
            "validation_mean_mae": round(float(mean_mae.min()), 2),
            "candidates_mean_mae": {k: round(float(v), 2) for k, v in mean_mae.items()},
        }

    sim = cfg.simulation
    provider = SpotProxyProvider(history["price_usd_t"])
    grid_rows = []
    for base in ("arima", "random_forest", "xgboost"):
        label = chosen[base]["validation_label"]
        p = preds[(preds["model"] == label) & (preds["horizon"] == sim.horizon)]
        for a in sim.a_grid:
            res = simulate(
                p, [ForecastDriven(a)], provider, sim.quantity_t, sim.financing_rate_annual
            )
            m = strategy_metrics(res).iloc[0]
            grid_rows.append(
                {
                    "driver": base,
                    "a": a,
                    "mean_revenue_per_t": float(m["mean_revenue_per_t"]),
                    "worst10_mean_per_t": float(m["worst10_mean_per_t"]),
                }
            )
    a_table = pd.DataFrame(grid_rows)
    best_rev = a_table["mean_revenue_per_t"].max()
    best = a_table[np.isclose(a_table["mean_revenue_per_t"], best_rev)].sort_values("a").iloc[0]

    frozen = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "note": (
            "Chosen on the validation period (targets 2016-01..2019-12) with models trained on "
            "targets up to 2015-12. Written before any test-period evaluation. Final models are "
            "refitted with these settings on targets up to final_train_end and then frozen."
        ),
        "final_train_end": cfg.evaluation.final_train_end,
        "models": chosen,
        "fx_experiment": {"train_start": FX_START, "models": ["random_forest", "xgboost"]},
        "simulation": {
            "driver_model": str(best["driver"]),
            "a": float(best["a"]),
            "horizon": sim.horizon,
            "validation_mean_revenue_per_t": round(float(best["mean_revenue_per_t"]), 2),
        },
        "config_sha256": config_hashes(cfg),
    }
    return frozen, a_table


def config_hashes(cfg: ProjectConfig) -> dict[str, str]:
    """SHA-256 of each base config file, recorded in the frozen config."""
    out = {}
    for name in ("data", "features", "models", "evaluation", "simulation"):
        path = cfg.paths.configs / f"{name}.yaml"
        out[f"{name}.yaml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out

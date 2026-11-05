"""Pipeline stages. The CLI calls these; each stage reads only the outputs of earlier stages."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cocoasense.config import (
    FeatureConfig,
    InsightsConfig,
    ProjectConfig,
    load_frozen,
    load_frozen_lstm,
)
from cocoasense.evaluation import figures
from cocoasense.evaluation.experiments import (
    FX_START,
    choose_frozen,
    config_hashes,
    features_for,
    run_validation,
    score_validation,
    validation_window,
)
from cocoasense.evaluation.explain import contributions
from cocoasense.evaluation.improve import (
    apply_guard,
    ensemble,
    guard_dates,
    scale_intervals,
    scale_intervals_expanding,
)
from cocoasense.evaluation.insights import (
    brier,
    confidence_check,
    ex_ante_confidence,
    explain,
    outcomes,
    probability_calibration,
    scorecard,
)
from cocoasense.evaluation.learning import (
    LEARNING_COLUMNS,
    fill_actuals,
    fixed_grid,
    merge_predictions,
    refit_walk_forward,
)
from cocoasense.evaluation.metrics import assign_periods, score
from cocoasense.evaluation.reports import leaderboard, write_table
from cocoasense.evaluation.walk_forward import Window, forecast_window, log_final_run
from cocoasense.features.build import build_features, build_targets
from cocoasense.ingestion.download import DownloadError, download
from cocoasense.ingestion.fx_live import FxUnavailableError, update_fx
from cocoasense.ingestion.manifest import ManifestEntry, sha256_of
from cocoasense.ingestion.sources import Downloader, check_source, ingest_all
from cocoasense.models.base import Forecaster
from cocoasense.models.registry import build_model
from cocoasense.simulation.currency import annual_cedi, to_cedis
from cocoasense.simulation.engine import scenario_delivery, simulate
from cocoasense.simulation.futures_provider import (
    CsvFuturesProvider,
    FuturesProvider,
    SpotProxyProvider,
)
from cocoasense.simulation.metrics import metrics_by_period, strategy_metrics
from cocoasense.simulation.strategies import FIXED_STRATEGIES, ForecastDriven
from cocoasense.validation.checks import DataQualityError, validate
from cocoasense.validation.parsers import PARSERS

logger = logging.getLogger(__name__)


class MissingArtifactError(RuntimeError):
    """Raised when a stage needs the output of an earlier stage that has not been run."""


def _require(path: Any, hint: str) -> None:
    if not path.exists():
        raise MissingArtifactError(f"{path} not found. Run `cocoasense {hint}` first.")


# --------------------------------------------------------------------------- data


def stage_ingest(
    cfg: ProjectConfig, refresh: bool = False, downloader: Downloader = download
) -> list[ManifestEntry]:
    """Download or reuse every raw file and update the manifest."""
    return ingest_all(cfg, refresh=refresh, downloader=downloader)


def stage_validate(cfg: ProjectConfig) -> pd.DataFrame:
    """Parse, check and align the series; write ``data/interim/monthly.parquet``."""
    p = cfg.paths
    d = cfg.data
    files = {k: p.raw / s.file for k, s in d.sources.items()}
    for f in files.values():
        _require(f, "ingest")
    price = PARSERS[d.primary_price](files[d.primary_price])
    check = PARSERS[d.crosscheck_price](files[d.crosscheck_price])
    fx = PARSERS["imf_er_ghs_usd"](files["imf_er_ghs_usd"])
    price, fx = price.dropna(), fx.dropna()
    report = validate(price, fx, check.dropna(), d.validation)
    report.save(p.artifacts)
    if report.errors:
        raise DataQualityError(
            f"{len(report.errors)} data-quality errors; see {p.artifacts / 'data_quality.md'}"
        )
    monthly = pd.concat([price, fx], axis=1).loc[price.index.min() : price.index.max()]
    monthly["crosscheck_price_usd_t"] = check.reindex(monthly.index)
    monthly.index.name = "date"
    p.interim.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(p.interim / "monthly.parquet")
    logger.info(
        "validated: %d months %s..%s",
        len(monthly),
        monthly.index.min().date(),
        monthly.index.max().date(),
    )
    return monthly


def load_monthly(cfg: ProjectConfig) -> pd.DataFrame:
    """Validated monthly table (``price_usd_t``, ``ghs_per_usd``)."""
    path = cfg.paths.interim / "monthly.parquet"
    _require(path, "validate")
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df


def stage_features(cfg: ProjectConfig) -> pd.DataFrame:
    """Write the full feature and target table (with FX features) for inspection and reports.

    Models rebuild features from history at each decision date, so this table is never read
    back by the models; it documents the feature set and its warm-up rows.
    """
    monthly = load_monthly(cfg)
    feats = build_features(monthly, None, features_for(cfg, use_fx=True))
    table = pd.concat(
        [
            monthly[["price_usd_t", "ghs_per_usd"]],
            feats,
            build_targets(monthly, cfg.features.horizons),
        ],
        axis=1,
    )
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg.paths.processed / "features.parquet")
    summary = {
        "rows": len(table),
        "columns": list(table.columns),
        "first_complete_price_feature_row": str(
            build_features(monthly, None, cfg.features).dropna().index.min().date()
        ),
        "first_complete_fx_feature_row": str(feats.dropna().index.min().date()),
    }
    (cfg.paths.processed / "features_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return table


# ------------------------------------------------------------------ development


def stage_train(cfg: ProjectConfig) -> pd.DataFrame:
    """Development stage: fit on targets <= train_end, forecast the validation period."""
    monthly = load_monthly(cfg)
    preds, infos, importance = run_validation(cfg, monthly)
    p = cfg.paths
    p.forecasts.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(p.forecasts / "validation_predictions.parquet")
    (p.metrics / "validation_model_info.json").write_text(
        json.dumps(infos, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not importance.empty:
        write_table(importance, p.metrics / "validation_permutation_importance.csv")
    stage_score_validation(cfg)
    return preds


def stage_score_validation(cfg: ProjectConfig) -> pd.DataFrame:
    """Score stored validation predictions and write leaderboards."""
    path = cfg.paths.forecasts / "validation_predictions.parquet"
    _require(path, "train")
    preds = pd.read_parquet(path)
    table = score_validation(preds)
    write_table(table, cfg.paths.metrics / "validation_metrics.csv")
    for h in sorted(table["horizon"].unique()):
        write_table(
            leaderboard(table, "validation", "validation", int(h)),
            cfg.paths.metrics / f"validation_leaderboard_h{h}.csv",
            f"Validation leaderboard, {h}-month horizon (targets 2016–2019)",
        )
    fx = preds[preds["experiment"] == "fx"]
    if not fx.empty:
        write_table(
            fx_pairs_table(fx, "validation"),
            cfg.paths.metrics / "validation_fx_experiment.csv",
            "Exchange-rate experiment (validation period)",
        )
    return table


def stage_freeze(cfg: ProjectConfig, overwrite: bool = False) -> dict[str, Any]:
    """Write ``configs/frozen.yaml`` from validation results only."""
    if cfg.paths.frozen.exists() and not overwrite:
        raise FileExistsError(
            f"{cfg.paths.frozen} already exists. The frozen configuration must not change after "
            "a final run; pass --overwrite only if no final run has used it."
        )
    path = cfg.paths.forecasts / "validation_predictions.parquet"
    _require(path, "train")
    frozen, a_table = choose_frozen(cfg, pd.read_parquet(path), load_monthly(cfg))
    write_table(
        a_table,
        cfg.paths.metrics / "validation_simulation_a_grid.csv",
        "Forecast-driven rule on the validation period: mean revenue by driver and a",
    )
    with cfg.paths.frozen.open("w", encoding="utf-8") as fh:
        fh.write("# Frozen configuration. Generated by `cocoasense freeze` from validation\n")
        fh.write("# results only. Commit this file BEFORE running `cocoasense evaluate --final`.\n")
        yaml.safe_dump(frozen, fh, sort_keys=False)
    logger.info("wrote %s", cfg.paths.frozen)
    return frozen


# ------------------------------------------------------------------------- final


def test_window(cfg: ProjectConfig, last: pd.Timestamp) -> Window:
    """Fit on targets <= final_train_end; forecast every test target month in the data."""
    ev = cfg.evaluation
    end = ev.ts("test_end") if ev.test_end else last
    return Window("test", ev.ts("final_train_end"), ev.ts("test_start"), min(end, last))


def stage_evaluate_final(cfg: ProjectConfig) -> pd.DataFrame:
    """Refit the frozen configuration on data up to the cut-off and forecast the test period."""
    p = cfg.paths
    _require(p.frozen, "freeze")
    frozen = load_frozen(p)
    monthly = load_monthly(cfg)
    window = test_window(cfg, monthly.index.max())
    surge = cfg.evaluation.ts("surge_start")
    preds, infos = [], {}
    for name, spec in frozen["models"].items():
        feats = features_for(cfg, bool(spec["use_fx"]))
        model = build_model(name, cfg, pd.Timestamp(spec["train_start"]), feats)
        logger.info("final: fitting %s (train_start %s)", name, spec["train_start"])
        pr, info = forecast_window(model, monthly, window, feats.horizons, surge, final=True)
        preds.append(pr.assign(experiment="main"))
        infos[name] = info
    # Sub-question 2: paired FX experiment on the months where the exchange rate is available.
    fx_last = monthly["ghs_per_usd"].last_valid_index()
    fx_hist = monthly.loc[:fx_last]
    fx_window = Window("test", window.fit_end, window.target_start, min(window.target_end, fx_last))
    for name in frozen["fx_experiment"]["models"]:
        for use_fx in (False, True):
            label = f"{name}_{'fx' if use_fx else 'nofx'}@{FX_START[:4]}"
            feats = features_for(cfg, use_fx)
            model = build_model(
                name, cfg, pd.Timestamp(frozen["fx_experiment"]["train_start"]), feats, label=label
            )
            pr, info = forecast_window(model, fx_hist, fx_window, feats.horizons, surge, final=True)
            preds.append(pr.assign(experiment="fx"))
            infos[label] = info
    all_preds = pd.concat(preds, ignore_index=True)
    p.forecasts.mkdir(parents=True, exist_ok=True)
    all_preds.to_parquet(p.forecasts / "test_predictions.parquet")
    (p.metrics / "test_model_info.json").write_text(
        json.dumps(infos, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = stage_score_test(cfg)
    log_final_run(
        p.artifacts / "final_runs.log",
        p.root,
        p.frozen,
        {
            "test_window": [str(window.target_start.date()), str(window.target_end.date())],
            "n_predictions": len(all_preds),
            "headline_mae": table[(table["period"] == "full")]
            .set_index(["model", "horizon"])["mae"]
            .round(2)
            .reset_index()
            .to_dict(orient="records"),
        },
    )
    return all_preds


def stage_score_test(cfg: ProjectConfig) -> pd.DataFrame:
    """Score stored test predictions by period and write tables."""
    path = cfg.paths.forecasts / "test_predictions.parquet"
    _require(path, "evaluate --final")
    preds = pd.read_parquet(path)
    periods = cfg.evaluation.period_bounds()
    main = preds[preds["experiment"] == "main"]
    table = score(assign_periods(main, periods))
    write_table(table, cfg.paths.metrics / "test_metrics.csv")
    for h in sorted(table["horizon"].unique()):
        for period in ("full", *periods):
            lb = leaderboard(table, "test", period, int(h))
            if not lb.empty:
                write_table(
                    lb,
                    cfg.paths.metrics / f"test_leaderboard_{period}_h{h}.csv",
                    f"Test leaderboard, {period}, {h}-month horizon",
                )
    fx = preds[preds["experiment"] == "fx"]
    if not fx.empty:
        write_table(
            fx_pairs_table(fx, "fx_window"),
            cfg.paths.metrics / "test_fx_experiment.csv",
            "Exchange-rate experiment (test period, months with FX data)",
        )
    return table


def fx_pairs_table(fx: pd.DataFrame, period: str) -> pd.DataFrame:
    """Score each FX arm against the price-only arm of the same model on identical months."""
    parts = []
    for nofx in sorted(m for m in fx["model"].unique() if "_nofx@" in m):
        base = nofx.split("_nofx@")[0]
        pair = fx[fx["model"].str.startswith(f"{base}_")]
        parts.append(score(pair.assign(period=period), baseline=nofx))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------ LSTM stretch model
#
# The LSTM was added after the main final run (docs/decisions.md D16). It follows the same
# protocol in its own files so that the original frozen configuration and test results stay
# exactly as they were: develop on validation -> write configs/frozen_lstm.yaml and commit it
# -> one logged final run on the test period. Nothing is refitted on data after Dec 2019.

LSTM_VALIDATION = "lstm_validation_predictions.parquet"
LSTM_TEST = "lstm_test_predictions.parquet"


def stage_lstm_train(cfg: ProjectConfig) -> pd.DataFrame:
    """Fit the LSTM on targets <= train_end for each training-start candidate; score validation."""
    _require(cfg.paths.forecasts / "validation_predictions.parquet", "train")
    monthly = load_monthly(cfg)
    window = validation_window(cfg)
    surge = cfg.evaluation.ts("surge_start")
    preds, infos = [], {}
    for start in cfg.evaluation.train_start_candidates:
        ts = pd.Timestamp(start)
        label = f"lstm@{ts:%Y}"
        logger.info("validation: fitting %s", label)
        model = build_model("lstm", cfg, ts, cfg.features, label=label)
        pr, info = forecast_window(model, monthly, window, cfg.features.horizons, surge)
        preds.append(pr.assign(experiment="lstm", base_model="lstm", use_fx=False, train_start=ts))
        infos[label] = info
    out = pd.concat(preds, ignore_index=True)
    out.to_parquet(cfg.paths.forecasts / LSTM_VALIDATION)
    (cfg.paths.metrics / "lstm_validation_model_info.json").write_text(
        json.dumps(infos, indent=2, default=str) + "\n", encoding="utf-8"
    )
    stage_lstm_score_validation(cfg)
    return out


def stage_lstm_score_validation(cfg: ProjectConfig) -> pd.DataFrame:
    """Validation metrics of every LSTM candidate against the same naive forecasts."""
    p = cfg.paths
    _require(p.forecasts / LSTM_VALIDATION, "lstm train")
    main = pd.read_parquet(p.forecasts / "validation_predictions.parquet")
    lstm = pd.read_parquet(p.forecasts / LSTM_VALIDATION)
    table = score_validation(pd.concat([main[main["model"] == "naive"], lstm], ignore_index=True))
    write_table(
        table,
        p.metrics / "lstm_validation_metrics.csv",
        "LSTM validation metrics (targets 2016–2019, fitted on targets up to 2015-12)",
    )
    return table


def stage_lstm_freeze(cfg: ProjectConfig, overwrite: bool = False) -> dict[str, Any]:
    """Choose the LSTM training start from validation MAE and write configs/frozen_lstm.yaml."""
    p = cfg.paths
    if p.frozen_lstm.exists() and not overwrite:
        raise FileExistsError(
            f"{p.frozen_lstm} already exists; pass --overwrite only if no final run has used it."
        )
    _require(p.frozen, "freeze")
    table = stage_lstm_score_validation(cfg)
    lstm = pd.read_parquet(p.forecasts / LSTM_VALIDATION)
    mean_mae = (
        table[table["model"].str.startswith("lstm@")].groupby("model")["mae"].mean().sort_index()
    )
    best = str(mean_mae.idxmin())
    start = pd.Timestamp(lstm.loc[lstm["model"] == best, "train_start"].iloc[0])
    frozen = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "note": (
            "LSTM stretch model, added after the main final run (docs/decisions.md D16). "
            "Training start chosen on the validation period (targets 2016-01..2019-12) with "
            "models trained on targets up to 2015-12, before any LSTM test-period evaluation. "
            "The final model is refitted on targets up to final_train_end and then frozen."
        ),
        "final_train_end": cfg.evaluation.final_train_end,
        "model": {
            "train_start": f"{start:%Y-%m}",
            "use_fx": False,
            "validation_label": best,
            "validation_mean_mae": round(float(mean_mae.min()), 2),
            "candidates_mean_mae": {k: round(float(v), 2) for k, v in mean_mae.items()},
        },
        "lstm_config_sha256": sha256_of(p.configs / "lstm.yaml"),
        "main_frozen_sha256": sha256_of(p.frozen),
        "config_sha256": config_hashes(cfg),
    }
    with p.frozen_lstm.open("w", encoding="utf-8") as fh:
        fh.write("# Frozen LSTM configuration. Generated by `cocoasense lstm freeze` from\n")
        fh.write(
            "# validation results only. Commit it BEFORE `cocoasense lstm evaluate --final`.\n"
        )
        yaml.safe_dump(frozen, fh, sort_keys=False)
    logger.info("wrote %s", p.frozen_lstm)
    return frozen


def stage_lstm_evaluate_final(cfg: ProjectConfig) -> pd.DataFrame:
    """Refit the frozen LSTM on targets <= Dec 2019, freeze it and forecast the test period."""
    p = cfg.paths
    _require(p.frozen_lstm, "lstm freeze")
    _require(p.forecasts / "test_predictions.parquet", "evaluate --final")
    spec = load_frozen_lstm(p)["model"]
    monthly = load_monthly(cfg)
    window = test_window(cfg, monthly.index.max())
    model = build_model("lstm", cfg, pd.Timestamp(spec["train_start"]), cfg.features)
    logger.info("final: fitting lstm (train_start %s)", spec["train_start"])
    preds, info = forecast_window(
        model, monthly, window, cfg.features.horizons, cfg.evaluation.ts("surge_start"), final=True
    )
    preds = preds.assign(experiment="main")
    preds.to_parquet(p.forecasts / LSTM_TEST)
    (p.metrics / "lstm_test_model_info.json").write_text(
        json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8"
    )
    table = stage_lstm_score_test(cfg)
    log_final_run(
        p.artifacts / "final_runs.log",
        p.root,
        p.frozen_lstm,
        {
            "run": "lstm_addendum",
            "test_window": [str(window.target_start.date()), str(window.target_end.date())],
            "n_predictions": len(preds),
            "headline_mae": table[(table["period"] == "full") & (table["model"] == "lstm")]
            .set_index(["model", "horizon"])["mae"]
            .round(2)
            .reset_index()
            .to_dict(orient="records"),
        },
    )
    return preds


def stage_lstm_score_test(cfg: ProjectConfig) -> pd.DataFrame:
    """Score the LSTM next to the four original models on identical test months."""
    p = cfg.paths
    _require(p.forecasts / LSTM_TEST, "lstm evaluate --final")
    periods = cfg.evaluation.period_bounds()
    table = score(assign_periods(test_predictions_with_lstm(cfg), periods))
    write_table(table, p.metrics / "test_metrics_with_lstm.csv")
    for h in sorted(table["horizon"].unique()):
        for period in ("full", *periods):
            lb = leaderboard(table, "test", period, int(h))
            if not lb.empty:
                write_table(
                    lb,
                    p.metrics / f"test_leaderboard_with_lstm_{period}_h{h}.csv",
                    f"Test leaderboard with the LSTM, {period}, {h}-month horizon",
                )
    return table


def test_predictions_with_lstm(cfg: ProjectConfig) -> pd.DataFrame:
    """Main test forecasts of the original models plus the LSTM (if it has been run)."""
    p = cfg.paths
    main = pd.read_parquet(p.forecasts / "test_predictions.parquet")
    main = main[main["experiment"] == "main"]
    lstm_path = p.forecasts / LSTM_TEST
    if not lstm_path.exists():
        return main
    return pd.concat([main, pd.read_parquet(lstm_path)], ignore_index=True)


# -------------------------------------------------------------------- simulation


def _provider(cfg: ProjectConfig, monthly: pd.DataFrame) -> FuturesProvider:
    sim = cfg.simulation
    if sim.forward_price_mode == "futures":
        if not sim.futures_csv:
            raise ValueError("forward_price_mode is 'futures' but futures_csv is not set")
        return CsvFuturesProvider(cfg.paths.root / sim.futures_csv)
    return SpotProxyProvider(monthly["price_usd_t"])


def stage_simulate(cfg: ProjectConfig) -> pd.DataFrame:
    """Run the forward-sales simulation on the test period with the frozen rule."""
    p = cfg.paths
    _require(p.frozen, "freeze")
    _require(p.forecasts / "test_predictions.parquet", "evaluate --final")
    frozen = load_frozen(p)["simulation"]
    monthly = load_monthly(cfg)
    preds = test_predictions_with_lstm(cfg)  # the LSTM appears only in sensitivity.csv
    provider = _provider(cfg, monthly)
    sim = cfg.simulation
    periods = cfg.evaluation.period_bounds()
    driver = frozen["driver_model"]
    a = float(frozen["a"])
    strategies = [*FIXED_STRATEGIES, ForecastDriven(a)]

    base = preds[(preds["model"] == driver) & (preds["horizon"] == frozen["horizon"])]
    results = simulate(base, strategies, provider, sim.quantity_t, sim.financing_rate_annual)
    p.simulation.mkdir(parents=True, exist_ok=True)
    results.to_parquet(p.simulation / "strategy_results.parquet")
    write_table(
        metrics_by_period(results, periods),
        p.simulation / "strategy_metrics.csv",
        f"Strategy metrics (test period, driver {driver}, a={a}, h={frozen['horizon']}, "
        f"{provider.mode})",
    )
    cedi_view = write_cedi_view(cfg, results, monthly)

    scen_rows = []
    for name, factor in sim.scenarios.items():
        res = simulate(
            base,
            strategies,
            provider,
            sim.quantity_t,
            sim.financing_rate_annual,
            delivery_override=scenario_delivery(base, provider, factor),
        )
        scen_rows.append(strategy_metrics(res).assign(scenario=name, factor=factor))
    write_table(
        pd.concat(scen_rows, ignore_index=True),
        p.simulation / "stress_scenarios.csv",
        "HYPOTHETICAL stress scenarios: delivery price = factor × decision-date price",
    )

    sens = []
    for a_val in sim.a_grid:
        res = simulate(
            base, [ForecastDriven(a_val)], provider, sim.quantity_t, sim.financing_rate_annual
        )
        sens.append(strategy_metrics(res).assign(parameter="a", value=a_val))
    for rate in sim.sensitivity.get("financing_rates", []):
        res = simulate(base, strategies, provider, sim.quantity_t, rate)
        sens.append(strategy_metrics(res).assign(parameter="financing_rate", value=rate))
    for h in sim.sensitivity.get("horizons", []):
        hp = preds[(preds["model"] == driver) & (preds["horizon"] == int(h))]
        if not hp.empty:
            res = simulate(hp, strategies, provider, sim.quantity_t, sim.financing_rate_annual)
            sens.append(strategy_metrics(res).assign(parameter="horizon", value=h))
    for model in sorted(preds["model"].unique()):
        mp = preds[(preds["model"] == model) & (preds["horizon"] == frozen["horizon"])]
        res = simulate(mp, [ForecastDriven(a)], provider, sim.quantity_t, sim.financing_rate_annual)
        sens.append(strategy_metrics(res).assign(parameter="driver_model", value=model))
    write_table(
        pd.concat(sens, ignore_index=True),
        p.simulation / "sensitivity.csv",
        "Sensitivity analysis (reported only; nothing here was used to choose settings)",
    )
    (p.simulation / "simulation_meta.json").write_text(
        json.dumps(
            {
                "forward_price_mode": provider.mode,
                "driver_model": driver,
                "a": a,
                "horizon": frozen["horizon"],
                "quantity_t": sim.quantity_t,
                "financing_rate_annual": sim.financing_rate_annual,
                "cedi_view": cedi_view,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def write_cedi_view(
    cfg: ProjectConfig, results: pd.DataFrame, monthly: pd.DataFrame
) -> dict[str, Any]:
    """Write the frozen simulation in cedis (delivery-month IMF rate); return its coverage.

    A reporting view of ``results``: nothing is re-simulated and no setting is chosen here.
    """
    p = cfg.paths
    cedi = to_cedis(results, monthly["ghs_per_usd"])
    coverage: dict[str, Any] = {
        "deliveries": int(cedi["decision_date"].nunique()),
        "of": int(results["decision_date"].nunique()),
        "first_delivery": None,
        "last_delivery": None,
    }
    if cedi.empty:
        logger.warning("no delivery month has an observed exchange rate; cedi view skipped")
        return coverage
    first, last = cedi["target_date"].min(), cedi["target_date"].max()
    coverage.update(first_delivery=f"{first:%Y-%m}", last_delivery=f"{last:%Y-%m}")
    span = (
        f"deliveries {first:%b %Y} – {last:%b %Y}, {coverage['deliveries']} of "
        f"{coverage['of']} with an observed rate"
    )
    write_table(
        metrics_by_period(cedi, cfg.evaluation.period_bounds(), value="revenue_ghs_per_t"),
        p.simulation / "cedi_strategy_metrics.csv",
        f"Cedi view of the frozen simulation: Ghana cedis per tonne at the delivery-month "
        f"GHS/US$ rate ({span}). Simulated, not Ghana's actual revenue.",
    )
    write_table(
        annual_cedi(cedi),
        p.simulation / "cedi_annual.csv",
        f"Cedi view by delivery year: mean US$/t, mean GHS/US$ and mean GHS/t ({span})",
    )
    return coverage


# ------------------------------------------------------------------------ report


def stage_report(cfg: ProjectConfig) -> None:
    """Write figures for whichever artifacts exist."""
    p = cfg.paths
    monthly = load_monthly(cfg)
    f = p.figures
    figures.price_history(monthly, cfg.evaluation, f / "eda_price_history.png")
    figures.returns_and_volatility(monthly, f / "eda_returns_volatility.png")
    figures.exchange_rate(monthly, f / "eda_exchange_rate.png")
    val = p.forecasts / "validation_predictions.parquet"
    if val.exists():
        vp = pd.read_parquet(val)
        vp = vp[(vp["experiment"] == "train_start")]
        if p.frozen.exists():
            labels = ["naive"] + [
                m["validation_label"] for k, m in load_frozen(p)["models"].items() if k != "naive"
            ]
            vp = vp[vp["model"].isin(labels)]
        for h in sorted(vp["horizon"].unique()):
            figures.actual_vs_predicted(
                vp,
                int(h),
                f"Validation forecasts, {h}-month horizon",
                f / f"validation_actual_vs_pred_h{h}.png",
            )
    test = p.forecasts / "test_predictions.parquet"
    if test.exists():
        tp = pd.read_parquet(test)
        tp = tp[tp["experiment"] == "main"]
        for h in sorted(tp["horizon"].unique()):
            figures.actual_vs_predicted(
                tp,
                int(h),
                f"Test forecasts, {h}-month horizon (models frozen at Dec 2019)",
                f / f"test_actual_vs_pred_h{h}.png",
            )
            figures.rolling_error(tp, int(h), 6, f / f"test_rolling_mape_h{h}.png")
        figures.error_by_period(
            pd.read_csv(p.metrics / "test_metrics.csv"), f / "test_error_by_period.png"
        )
        write_table(
            largest_misses(tp, 10),
            p.metrics / "test_largest_misses.csv",
            "Largest absolute percentage errors in the test period",
        )
    if (p.forecasts / LSTM_TEST).exists():
        tl = test_predictions_with_lstm(cfg)
        for h in sorted(tl["horizon"].unique()):
            figures.actual_vs_predicted(
                tl,
                int(h),
                f"Test forecasts with the LSTM, {h}-month horizon (frozen at Dec 2019)",
                f / f"test_actual_vs_pred_with_lstm_h{h}.png",
            )
    sim = p.simulation / "strategy_results.parquet"
    if sim.exists():
        sim_results = pd.read_parquet(sim)
        figures.strategy_outcomes(sim_results, f / "simulation_strategy_outcomes.png")
        cedi = to_cedis(sim_results, monthly["ghs_per_usd"])
        if not cedi.empty:
            figures.cedi_revenue(cedi, f / "simulation_cedi_revenue.png")


def largest_misses(preds: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n absolute percentage errors per horizon, with the actual move over the horizon."""
    p = preds.copy()
    p["error_pct"] = (p["y_pred"] - p["y_true"]) / p["y_true"] * 100
    p["actual_move_pct"] = (p["y_true"] / p["last_obs"] - 1) * 100
    p["abs_error_pct"] = p["error_pct"].abs()
    top = p.sort_values("abs_error_pct", ascending=False).groupby("horizon").head(n)
    cols = [
        "horizon",
        "model",
        "decision_date",
        "target_date",
        "last_obs",
        "y_true",
        "y_pred",
        "error_pct",
        "actual_move_pct",
    ]
    return top.sort_values(["horizon", "abs_error_pct"], ascending=[True, False])[cols]


def stage_all(cfg: ProjectConfig, refresh: bool = False) -> None:
    """Run every stage from raw data to artifacts (requires a committed frozen config)."""
    stage_ingest(cfg, refresh=refresh)
    stage_validate(cfg)
    stage_features(cfg)
    stage_train(cfg)
    if not cfg.paths.frozen.exists():
        logger.warning("configs/frozen.yaml missing: run `cocoasense freeze`, commit it, rerun.")
        stage_report(cfg)
        return
    stage_evaluate_final(cfg)
    if cfg.lstm is not None:
        stage_lstm_train(cfg)
        if cfg.paths.frozen_lstm.exists():
            stage_lstm_evaluate_final(cfg)
    stage_simulate(cfg)
    stage_report(cfg)


# --------------------------------------------- learning track, insights, update

LEARNING = "learning_predictions.parquet"
COIN_FLIP_BRIER = 0.25
INSIGHTS = "insights_predictions.parquet"
ROW_COLUMNS = [
    "track",
    "model",
    "decision_date",
    "target_date",
    "horizon",
    "last_obs",
    "y_true",
    "y_pred",
    "lower",
    "upper",
]


def _insights_cfg(cfg: ProjectConfig) -> InsightsConfig:
    if cfg.insights is None:
        raise MissingArtifactError("configs/insights.yaml not found")
    return cfg.insights


def learning_models(cfg: ProjectConfig) -> dict[str, Callable[[], Forecaster]]:
    """Model factories for the learning track.

    Each uses the frozen training start, features and hyperparameters (from the final run's
    ``test_model_info.json``, tuned on pre-2020 data) and is refitted at every decision month.
    """
    p = cfg.paths
    _require(p.frozen, "freeze")
    info_path = p.metrics / "test_model_info.json"
    _require(info_path, "evaluate --final")
    frozen = load_frozen(p)["models"]
    infos = json.loads(info_path.read_text(encoding="utf-8"))
    factories: dict[str, Callable[[], Forecaster]] = {}
    for name in _insights_cfg(cfg).learning.models:
        spec = frozen[name]
        feats = features_for(cfg, bool(spec["use_fx"]))
        start = pd.Timestamp(spec["train_start"])
        best = (infos.get(name) or {}).get("best_params")

        def make(
            name: str = name,
            start: pd.Timestamp = start,
            feats: FeatureConfig = feats,
            best: dict[str, dict[str, Any]] | None = best,
        ) -> Forecaster:
            model = build_model(name, cfg, start, feats)
            if best:
                model.grid = fixed_grid(best)  # type: ignore[attr-defined]
            return model

        factories[name] = make
    return factories


def stage_learn(cfg: ProjectConfig) -> pd.DataFrame:
    """Learning track: refit every month on all prices known so far and forecast ahead.

    Incremental: stored rows are kept (a live forecast keeps what was forecast at the time) and
    only new (decision month, horizon) pairs are fitted. Actual prices are filled in as they
    are published.
    """
    p = cfg.paths
    ins = _insights_cfg(cfg)
    monthly = load_monthly(cfg)
    path = p.forecasts / LEARNING
    old = pd.read_parquet(path) if path.exists() else None
    new_parts = []
    for name, make in learning_models(cfg).items():
        done: set[tuple[pd.Timestamp, int]] = set()
        if old is not None:
            mine = old[old["model"] == name]
            done = set(zip(mine["decision_date"], mine["horizon"], strict=True))
        new_parts.append(
            refit_walk_forward(make, monthly, ins.learning.ts(), cfg.features.horizons, skip=done)
        )
    new_parts = [part for part in new_parts if not part.empty]
    new = (
        pd.concat(new_parts, ignore_index=True)
        if new_parts
        else pd.DataFrame(columns=LEARNING_COLUMNS)
    )
    merged = merge_predictions(old, new)
    merged = fill_actuals(merged, monthly["price_usd_t"])
    p.forecasts.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    logger.info("learning track: %d rows (%d live)", len(merged), merged["y_true"].isna().sum())
    return merged


def _frozen_rows(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen-track test rows (main models + LSTM) and their validation-period error seed."""
    p = cfg.paths
    test = test_predictions_with_lstm(cfg).assign(track="frozen")
    seed_parts = []
    val_path = p.forecasts / "validation_predictions.parquet"
    if val_path.exists():
        val = pd.read_parquet(val_path)
        val = val[val["experiment"] == "train_start"]
        labels = {"naive": "naive"} | {
            m["validation_label"]: k
            for k, m in load_frozen(p)["models"].items()
            if "validation_label" in m
        }
        val = val[val["model"].isin(labels)]
        seed_parts.append(val.assign(model=val["model"].map(labels)))
    lstm_val = p.forecasts / LSTM_VALIDATION
    if lstm_val.exists() and p.frozen_lstm.exists():
        label = load_frozen_lstm(p)["model"]["validation_label"]
        lv = pd.read_parquet(lstm_val)
        seed_parts.append(lv[lv["model"] == label].assign(model="lstm"))
    seed = pd.concat(seed_parts, ignore_index=True) if seed_parts else test.iloc[:0]
    return test[ROW_COLUMNS], seed.assign(track="frozen")[ROW_COLUMNS]


def _live_ranges(cfg: ProjectConfig, learning: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Apply the adopted volatility-aware ranges (D23) to the learning track, if frozen."""
    path = cfg.paths.configs / "frozen_improve.yaml"
    if not path.exists():
        return learning
    decisions = yaml.safe_load(path.read_text(encoding="utf-8"))["decisions"]
    models = [
        m for m in decisions["volatility_scaled_intervals"]["adopt_for"] if m in IMPROVE_MODELS
    ]
    window = int(_improve_cfg(cfg)["volatility_scaled_intervals"]["window"])
    return scale_intervals_expanding(learning, price, window, models)


def stage_insights(cfg: ProjectConfig) -> pd.DataFrame:
    """Confidence for every forecast, success scorecards, calibration and live forecasts."""
    p = cfg.paths
    ins = _insights_cfg(cfg)
    price = load_monthly(cfg)["price_usd_t"]
    thresholds = ins.success_thresholds_pct
    test_start = cfg.evaluation.ts("test_start")

    frozen, seed = _frozen_rows(cfg)
    parts = [ex_ante_confidence(frozen, pd.concat([seed, frozen]), price, ins)]
    learn_path = p.forecasts / LEARNING
    if learn_path.exists():
        learning = fill_actuals(pd.read_parquet(learn_path), price)[ROW_COLUMNS]
        learning = _live_ranges(cfg, learning, price)
        parts.append(ex_ante_confidence(learning, learning, price, ins))
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows["target_date"] >= test_start].reset_index(drop=True)
    rows.to_parquet(p.forecasts / INSIGHTS)

    scored = outcomes(rows, thresholds)
    long = assign_periods(scored, cfg.evaluation.period_bounds())
    write_table(
        scorecard(long, ["track", "period", "horizon", "model"], thresholds),
        p.metrics / "scorecard.csv",
        "Success scorecard (test targets from 2020): % of forecasts within 5 % / 10 % of the "
        "actual price, direction correct, actual inside the 80 % range",
    )
    write_table(
        pd.concat(
            [
                confidence_check(g).assign(track=t, horizon=h)
                for (t, h), g in scored.groupby(["track", "horizon"])
            ],
            ignore_index=True,
        ),
        p.metrics / "confidence_check.csv",
        "Confidence labels vs what happened: actual hit rates per label (labels set ex ante)",
    )
    write_table(
        pd.concat(
            [
                probability_calibration(g).assign(track=t, horizon=h, brier=brier(g))
                for (t, h), g in scored.groupby(["track", "horizon"])
            ],
            ignore_index=True,
        ),
        p.metrics / "probability_calibration.csv",
        "Chance of a rise (p_up) vs how often the price rose; Brier 0 = perfect, 0.25 = coin flip",
    )
    if learn_path.exists():
        both = scorecard(long, ["period", "horizon", "model", "track"], thresholds)
        wide = both.pivot_table(
            index=["period", "horizon", "model"],
            columns="track",
            values=["mape", "within_10pct_pct", "n"],
        )
        wide.columns = [f"{v}_{t}" for v, t in wide.columns]
        wide = wide.dropna(subset=["mape_frozen", "mape_learning"]).reset_index()
        wide["mape_change"] = wide["mape_learning"] - wide["mape_frozen"]
        write_table(
            wide,
            p.metrics / "learning_vs_frozen.csv",
            "Learning track (monthly refit) vs frozen track: MAPE and % within 10 %, same months",
        )
        live = rows[(rows["track"] == "learning") & rows["y_true"].isna()].copy()
        reliable = bool(brier(scored[scored["track"] == "learning"]) < COIN_FLIP_BRIER)
        live["explanation"] = [
            explain(r, model_label(str(r["model"])), reliable) for _, r in live.iterrows()
        ]
        write_table(
            live.drop(columns=["y_true"]),
            p.metrics / "live_forecasts.csv",
            f"Live forecasts from data through {price.index.max():%b %Y} (learning track)",
        )
    return rows


def model_label(name: str) -> str:
    """Display name of a model."""
    return {
        "naive": "Naive (last price)",
        "arima": "ARIMA",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
    }.get(name, name)


def stage_update(cfg: ProjectConfig, refresh: bool = True) -> dict[str, Any]:
    """Fetch the latest public prices, score earlier forecasts, refit and forecast ahead."""
    interim = cfg.paths.interim / "monthly.parquet"
    before = load_monthly(cfg).index.max() if interim.exists() else None
    stage_ingest(cfg, refresh=refresh)
    monthly = stage_validate(cfg)
    stage_learn(cfg)
    stage_insights(cfg)
    stage_explain(cfg)
    fx = stage_fx(cfg) if refresh else {"status": "skipped"}
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "refreshed": refresh,
        "fx": fx,
        "data_through_before": None if before is None else f"{before:%Y-%m}",
        "data_through": f"{monthly.index.max():%Y-%m}",
    }
    with (cfg.paths.artifacts / "update_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def stage_auto_update(
    cfg: ProjectConfig, downloader: Downloader = download, fx_getter: Any = None
) -> dict[str, Any]:
    """Unattended update: retrain and re-forecast only when a source has new data.

    Run by the scheduled task. Offline or blocked sources are skipped (the next run retries);
    a data-quality failure leaves the previous data and forecasts in place.
    """
    p = cfg.paths
    changed, failed = [], []
    for key, src in cfg.data.sources.items():
        try:
            if check_source(key, src, cfg, downloader):
                changed.append(key)
        except DownloadError as exc:
            failed.append(key)
            logger.warning("%s not checked: %s", key, str(exc).splitlines()[0])
    first_run = not (p.forecasts / LEARNING).exists() or not (p.forecasts / INSIGHTS).exists()
    status = "offline" if len(failed) == len(cfg.data.sources) else "no new data"
    if changed or first_run:
        try:
            stage_validate(cfg)
            stage_learn(cfg)
            stage_insights(cfg)
            stage_explain(cfg)
            status = "retrained on new data" if changed else "built forecasts"
        except DataQualityError as exc:
            status = "new data failed checks"
            logger.error("new data failed validation; previous forecasts kept: %s", exc)
        except MissingArtifactError as exc:
            status = "project not built yet: run `cocoasense all`"
            logger.error("%s", exc)
    interim = p.interim / "monthly.parquet"
    fx = stage_fx(cfg, fx_getter) if interim.exists() else {"status": "no data"}
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": "auto",
        "status": status,
        "fx": fx,
        "changed": changed,
        "not_checked": failed,
        "data_through": f"{load_monthly(cfg).index.max():%Y-%m}" if interim.exists() else None,
    }
    p.artifacts.mkdir(parents=True, exist_ok=True)
    with (p.artifacts / "update_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


# ------------------------------------------------------------- live cedi rate

FX_DAILY = "usd_ghs_daily.csv"


def stage_fx(cfg: ProjectConfig, getter: Any = None) -> dict[str, Any]:
    """Fetch the latest US$ to cedi rate and backfill months after the IMF series (D21).

    Display only: no model reads these rates. Offline runs return ``status: offline``.
    """
    path = cfg.paths.artifacts / "fx" / FX_DAILY
    imf = load_monthly(cfg)["ghs_per_usd"]
    try:
        rec = update_fx(path, imf, getter=getter) if getter else update_fx(path, imf)
        return {"status": "ok", **rec}
    except FxUnavailableError as exc:
        logger.warning("cedi rate not updated (offline?): %s", str(exc)[:200])
        return {"status": "offline"}


# ----------------------------------------------- improvement addendum (D23)

IMPROVE_MODELS = ["naive", "arima", "random_forest", "xgboost"]
KEY_COLS = ["decision_date", "target_date", "horizon"]


def _improve_cfg(cfg: ProjectConfig) -> dict[str, Any]:
    path = cfg.paths.configs / "improve.yaml"
    _require(path, "(restore configs/improve.yaml)")
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _validation_rows(cfg: ProjectConfig) -> pd.DataFrame:
    """Development-stage forecasts (2016-2019) of the frozen model choices, renamed to bases."""
    p = cfg.paths
    val = pd.read_parquet(p.forecasts / "validation_predictions.parquet")
    val = val[val["experiment"] == "train_start"]
    labels = {"naive": "naive"} | {
        m["validation_label"]: k
        for k, m in load_frozen(p)["models"].items()
        if "validation_label" in m
    }
    val = val[val["model"].isin(labels)]
    return val.assign(model=val["model"].map(labels))


def _candidates(
    rows: pd.DataFrame, price: pd.Series, imp: dict[str, Any], cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Original forecasts, the ensemble, and volatility-scaled versions of both."""
    cols = [*KEY_COLS, "model", "last_obs", "y_true", "y_pred", "lower", "upper"]
    base = rows[rows["model"].isin(IMPROVE_MODELS)][cols]
    ens = ensemble(base, imp["ensemble"]["members"])
    both = pd.concat([base, ens], ignore_index=True)
    window = imp["volatility_scaled_intervals"]["window"]
    scaled = scale_intervals(both, price, window, cutoff).drop(columns="vol_factor")
    scaled = scaled.assign(model=scaled["model"] + "+volrange")
    return pd.concat([both, scaled], ignore_index=True)


def _guarded_simulation(
    cfg: ProjectConfig,
    preds: pd.DataFrame,
    price: pd.Series,
    imp: dict[str, Any],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Frozen forecast-driven rule with and without the volatility guard (revenue per tonne)."""
    sim = cfg.simulation
    frozen = load_frozen(cfg.paths)["simulation"]
    driver = preds[
        (preds["model"] == frozen["driver_model"]) & (preds["horizon"] == frozen["horizon"])
    ]
    provider = _provider(cfg, load_monthly(cfg))
    strategies = [*FIXED_STRATEGIES, ForecastDriven(float(frozen["a"]))]
    res = simulate(driver, strategies, provider, sim.quantity_t, sim.financing_rate_annual)
    dates = guard_dates(price, imp["volatility_guard"]["window"], cutoff)
    guarded = apply_guard(res, dates)
    rows = []
    for label, frame in (("forecast_driven", res), ("forecast_driven+guard", guarded)):
        fd = frame[frame["strategy"] == "forecast_driven"]
        rows.append(
            {
                "rule": label,
                "mean_revenue_per_t": float(fd["revenue_per_t"].mean()),
                "guarded_months": int(fd["decision_date"].isin(dates).sum())
                if label.endswith("guard")
                else 0,
                "n_dates": len(fd),
            }
        )
    half = res[res["strategy"] == "fixed_50_50"]
    rows.append(
        {
            "rule": "fixed_50_50",
            "mean_revenue_per_t": float(half["revenue_per_t"].mean()),
            "guarded_months": 0,
            "n_dates": len(half),
        }
    )
    return pd.DataFrame(rows)


def stage_improve_validate(cfg: ProjectConfig) -> dict[str, Any]:
    """Score the pre-registered candidates on 2016-2019 and apply the adoption rules."""
    p = cfg.paths
    imp = _improve_cfg(cfg)
    price = load_monthly(cfg)["price_usd_t"]
    cutoff = cfg.evaluation.ts("train_end")
    cand = _candidates(_validation_rows(cfg), price, imp, cutoff)
    table = score(cand.assign(stage="validation", period="validation"))
    write_table(
        table,
        p.metrics / "improve_validation.csv",
        "Improvement candidates on the validation period (2016-2019), development models",
    )
    sim = _guarded_simulation(cfg, cand, price, imp, cutoff)
    write_table(
        sim,
        p.metrics / "improve_validation_simulation.csv",
        "Volatility guard on the validation period (frozen forecast-driven rule)",
    )
    t = table.set_index(["model", "horizon"])
    horizons = sorted(int(h) for h in cand["horizon"].unique())
    members = list(imp["ensemble"]["members"])
    ens_ok = all(
        t.loc[("ensemble", h), "mae"] < min(t.loc[(m, h), "mae"] for m in members) for h in horizons
    )
    adopt_for = [
        m
        for m in [*IMPROVE_MODELS, "ensemble"]
        if all(
            abs(t.loc[(f"{m}+volrange", h), "coverage"] - 0.8)
            <= abs(t.loc[(m, h), "coverage"] - 0.8) + 1e-12
            for h in horizons
        )
    ]
    s = sim.set_index("rule")["mean_revenue_per_t"]
    return {
        "ensemble": {
            "adopt": bool(ens_ok),
            "members": members,
            "validation_mae": {h: round(float(t.loc[("ensemble", h), "mae"]), 2) for h in horizons},
            "members_validation_mae": {
                m: {h: round(float(t.loc[(m, h), "mae"]), 2) for h in horizons} for m in members
            },
        },
        "volatility_scaled_intervals": {"adopt_for": adopt_for},
        "volatility_guard": {
            "adopt": bool(s["forecast_driven+guard"] >= s["forecast_driven"] - 1e-9),
            "validation_revenue": {k: round(float(v), 2) for k, v in s.items()},
        },
    }


def stage_improve_freeze(cfg: ProjectConfig, overwrite: bool = False) -> dict[str, Any]:
    """Write configs/frozen_improve.yaml from the validation decisions (commit before final)."""
    path = cfg.paths.configs / "frozen_improve.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; it must not change after the final run")
    decisions = stage_improve_validate(cfg)
    record = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "note": "Chosen on 2016-2019 validation forecasts by the rules in configs/improve.yaml, "
        "before any candidate was scored on the test period.",
        "decisions": decisions,
        "improve_config_sha256": sha256_of(cfg.paths.configs / "improve.yaml"),
        "main_frozen_sha256": sha256_of(cfg.paths.frozen),
    }
    path.write_text(
        "# Frozen improvement addendum (D23). Commit BEFORE `cocoasense improve final`.\n"
        + yaml.safe_dump(record, sort_keys=False),
        encoding="utf-8",
    )
    return record


def stage_improve_final(cfg: ProjectConfig) -> pd.DataFrame:
    """One logged run: the adopted candidates on the test period, next to the originals."""
    p = cfg.paths
    path = p.configs / "frozen_improve.yaml"
    _require(path, "improve freeze")
    decisions = yaml.safe_load(path.read_text(encoding="utf-8"))["decisions"]
    imp = _improve_cfg(cfg)
    price = load_monthly(cfg)["price_usd_t"]
    cutoff = cfg.evaluation.ts("final_train_end")
    test = pd.read_parquet(p.forecasts / "test_predictions.parquet")
    cand = _candidates(test[test["experiment"] == "main"], price, imp, cutoff)
    keep = set(IMPROVE_MODELS)
    if decisions["ensemble"]["adopt"]:
        keep.add("ensemble")
    for m in decisions["volatility_scaled_intervals"]["adopt_for"]:
        if m in keep:
            keep.add(f"{m}+volrange")
    cand = cand[cand["model"].isin(keep)]
    long = assign_periods(cand.assign(stage="test"), cfg.evaluation.period_bounds())
    table = score(long)
    write_table(
        table,
        p.metrics / "improve_test_metrics.csv",
        "Improvement addendum on the test period: adopted candidates next to the originals",
    )
    sim = _guarded_simulation(cfg, cand, price, imp, cutoff)
    if not decisions["volatility_guard"]["adopt"]:
        sim = sim[sim["rule"] != "forecast_driven+guard"]
    write_table(
        sim,
        p.metrics / "improve_test_simulation.csv",
        "Selling rule on the test period, with the volatility guard if adopted",
    )
    log_final_run(
        p.artifacts / "final_runs.log",
        p.root,
        path,
        {"run": "improvement_addendum", "models": sorted(keep), "n_rows": len(cand)},
    )
    return table


# ------------------------------------------------------ live explanations

EXPLAINED = ("random_forest", "xgboost")


def stage_explain(cfg: ProjectConfig) -> pd.DataFrame:
    """Per-input contributions behind the live tree-model forecasts (learning track).

    The models are refitted on all prices known now, exactly as the learning track does for its
    latest forecasts, so the breakdown explains the forecasts shown on the Live Forecast page.
    """
    p = cfg.paths
    monthly = load_monthly(cfg)
    factories = learning_models(cfg)
    last = pd.Timestamp(monthly.index.max())
    parts = []
    for name in EXPLAINED:
        if name not in factories:
            continue
        model = factories[name]()
        model.fit(monthly)
        for h in cfg.features.horizons:
            table = contributions(model, monthly, h)  # type: ignore[arg-type]
            parts.append(table.assign(model=name, horizon=h, decision_date=last))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    live = p.forecasts / LEARNING
    if live.exists():
        stored = pd.read_parquet(live)
        stored = stored[(stored["decision_date"] == last) & stored["model"].isin(EXPLAINED)]
        price_now = float(monthly["price_usd_t"].iloc[-1])
        for (m, h), g in out.groupby(["model", "horizon"]):
            row = stored[(stored["model"] == m) & (stored["horizon"] == h)]
            if not row.empty:
                implied = price_now * float(np.exp(g["prediction"].iloc[0]))
                gap = abs(implied / float(row["y_pred"].iloc[0]) - 1)
                if gap > 0.005:
                    logger.warning(
                        "%s h=%s explanation differs from live forecast by %.2f%%", m, h, gap * 100
                    )
    cols = ["model", "horizon", "decision_date", "feature", "label", "value", "contribution"]
    write_table(
        out[[*cols, "base", "prediction"]],
        p.metrics / "live_explanations.csv",
        f"Why the live forecasts are what they are (data through {last:%b %Y}): contribution of "
        "each input to the forecast log return",
    )
    return out

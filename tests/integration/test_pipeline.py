"""End-to-end run on a tiny SYNTHETIC fixture (small grids, shifted timeline)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cocoasense import pipeline
from cocoasense.cli import app
from cocoasense.config import ProjectConfig
from cocoasense.evaluation.walk_forward import TestPeriodLockedError


@pytest.mark.slow
def test_full_pipeline(raw_fixture_files: ProjectConfig) -> None:
    cfg = raw_fixture_files
    p = cfg.paths
    pipeline.stage_ingest(cfg, downloader=lambda *a: (_ for _ in ()).throw(AssertionError()))
    monthly = pipeline.stage_validate(cfg)
    assert monthly.index.max() == pd.Timestamp("2019-12-01")
    pipeline.stage_features(cfg)
    assert (p.processed / "features.parquet").exists()

    with pytest.raises(pipeline.MissingArtifactError):
        pipeline.stage_evaluate_final(cfg)  # no frozen config yet

    val = pipeline.stage_train(cfg)
    assert set(val["stage"]) == {"validation"}
    assert val["target_date"].max() <= pd.Timestamp("2013-12-01")
    assert (p.metrics / "validation_leaderboard_h3.md").exists()
    assert (p.metrics / "validation_fx_experiment.csv").exists()

    frozen = pipeline.stage_freeze(cfg)
    assert frozen["simulation"]["a"] in cfg.simulation.a_grid
    with pytest.raises(FileExistsError):
        pipeline.stage_freeze(cfg)
    saved = yaml.safe_load(p.frozen.read_text(encoding="utf-8"))
    assert saved["models"]["arima"]["use_fx"] is False

    test = pipeline.stage_evaluate_final(cfg)
    main = test[test["experiment"] == "main"]
    assert main["target_date"].min() == pd.Timestamp("2014-01-01")
    assert set(main["model"]) == {"naive", "arima", "random_forest", "xgboost"}
    log = (p.artifacts / "final_runs.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 1 and json.loads(log[0])["frozen_config_sha256"]
    metrics = pd.read_csv(p.metrics / "test_metrics.csv")
    assert {"full", "pre_surge", "surge", "post_peak"} <= set(metrics["period"])

    # LSTM stretch model: own validation -> freeze -> final cycle, same cut-off
    with pytest.raises(pipeline.MissingArtifactError):
        pipeline.stage_lstm_evaluate_final(cfg)  # no frozen LSTM config yet
    lstm_val = pipeline.stage_lstm_train(cfg)
    assert set(lstm_val["model"]) == {"lstm@1982", "lstm@1991"}
    assert lstm_val["target_date"].max() <= pd.Timestamp("2013-12-01")
    fl = pipeline.stage_lstm_freeze(cfg)
    assert fl["model"]["validation_label"] in {"lstm@1982", "lstm@1991"}
    assert fl["main_frozen_sha256"]
    with pytest.raises(FileExistsError):
        pipeline.stage_lstm_freeze(cfg)
    lstm_test = pipeline.stage_lstm_evaluate_final(cfg)
    assert lstm_test["target_date"].min() == pd.Timestamp("2014-01-01")
    log = (p.artifacts / "final_runs.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 2 and json.loads(log[1])["run"] == "lstm_addendum"
    with_lstm = pd.read_csv(p.metrics / "test_metrics_with_lstm.csv")
    assert set(with_lstm["model"]) == {"naive", "arima", "random_forest", "xgboost", "lstm"}
    assert pd.read_csv(p.metrics / "test_metrics.csv").equals(metrics)  # originals untouched

    # Improvement addendum: validate -> freeze -> one logged final run (D23)
    decisions = pipeline.stage_improve_validate(cfg)
    assert set(decisions) == {"ensemble", "volatility_scaled_intervals", "volatility_guard"}
    pipeline.stage_improve_freeze(cfg)
    with pytest.raises(FileExistsError):
        pipeline.stage_improve_freeze(cfg)
    improved = pipeline.stage_improve_final(cfg)
    assert {"naive", "arima", "random_forest", "xgboost"} <= set(improved["model"])
    log = (p.artifacts / "final_runs.log").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(log[-1])["run"] == "improvement_addendum"
    assert pd.read_csv(p.metrics / "test_metrics.csv").equals(metrics)  # originals untouched

    results = pipeline.stage_simulate(cfg)
    assert set(results["strategy"]) == {"sell_forward", "wait", "fixed_50_50", "forecast_driven"}
    sens = pd.read_csv(p.simulation / "sensitivity.csv")
    assert "lstm" in set(sens.loc[sens["parameter"] == "driver_model", "value"])
    for name in ("strategy_metrics.csv", "stress_scenarios.csv", "sensitivity.csv"):
        assert (p.simulation / name).exists()
    meta = json.loads((p.simulation / "simulation_meta.json").read_text(encoding="utf-8"))
    assert meta["cedi_view"]["of"] == results["decision_date"].nunique()
    cedi = pd.read_csv(p.simulation / "cedi_strategy_metrics.csv")
    assert (cedi.loc[cedi["period"] == "full", "n_dates"] == meta["cedi_view"]["deliveries"]).all()
    assert (p.simulation / "cedi_annual.csv").exists()

    # Learning track and insights: monthly refit, confidence, scorecards, live forecasts
    learning = pipeline.stage_learn(cfg)
    assert set(learning["model"]) == {"naive", "arima", "xgboost"}
    assert (learning["fit_end"] == learning["decision_date"]).all()
    live = learning[learning["y_true"].isna()]
    assert (live["target_date"] > pd.Timestamp("2019-12-01")).all() and len(live) == 3 * 4
    again = pipeline.stage_learn(cfg)  # incremental: nothing new to fit, same rows
    pd.testing.assert_frame_equal(again, learning)
    rows = pipeline.stage_insights(cfg)
    assert set(rows["track"]) == {"frozen", "learning"}
    assert rows["confidence"].isin(["High", "Medium", "Low", "Not enough history"]).all()
    card = pd.read_csv(p.metrics / "scorecard.csv")
    assert {"within_5pct_pct", "within_10pct_pct", "direction_correct_pct"} <= set(card.columns)
    for name in ("confidence_check.csv", "probability_calibration.csv", "learning_vs_frozen.csv"):
        assert (p.metrics / name).exists()
    expl = pipeline.stage_explain(cfg)
    assert set(expl["model"]) == {"xgboost"}  # Random Forest is not in this fixture's track
    assert (p.metrics / "live_explanations.csv").exists()
    from cocoasense.report_pdf import build_report

    assert build_report(cfg)[:4] == b"%PDF"
    live_table = pd.read_csv(p.metrics / "live_forecasts.csv")
    assert live_table["explanation"].str.contains("expects").all()
    rec = pipeline.stage_update(cfg, refresh=False)
    assert rec["data_through"] == "2019-12" == rec["data_through_before"]

    pipeline.stage_report(cfg)
    assert (p.figures / "test_actual_vs_pred_h3.png").exists()
    assert (p.figures / "simulation_strategy_outcomes.png").exists()
    assert (p.figures / "test_actual_vs_pred_with_lstm_h1.png").exists()
    assert (p.figures / "simulation_cedi_revenue.png").exists()
    misses = pd.read_csv(p.metrics / "test_largest_misses.csv")
    assert set(misses["horizon"]) == {1, 3} and len(misses) == 20
    assert (
        misses.groupby("horizon")["error_pct"]
        .apply(lambda s: s.abs().is_monotonic_decreasing)
        .all()
    )


def test_cli_help_and_locked_final(project: ProjectConfig) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "evaluate" in res.output
    res = runner.invoke(app, ["evaluate", "--root", str(project.paths.root)])
    assert res.exit_code != 0  # no validation predictions yet
    assert isinstance(res.exception, pipeline.MissingArtifactError)
    for cmd in (["lstm", "evaluate"], ["lstm", "evaluate", "--final"], ["lstm", "train"]):
        res = runner.invoke(app, [*cmd, "--root", str(project.paths.root)])
        assert isinstance(res.exception, pipeline.MissingArtifactError), cmd


def test_cli_stages_on_fixture(raw_fixture_files: ProjectConfig) -> None:
    runner = CliRunner()
    root = str(raw_fixture_files.paths.root)
    for cmd in (["ingest"], ["validate"], ["features"], ["report"]):
        res = runner.invoke(app, [*cmd, "--root", root])
        assert res.exit_code == 0, res.output


def test_cli_final_requires_frozen_config(raw_fixture_files: ProjectConfig) -> None:
    runner = CliRunner()
    root = str(raw_fixture_files.paths.root)
    for cmd in (["ingest"], ["validate"]):
        runner.invoke(app, [*cmd, "--root", root])
    res = runner.invoke(app, ["evaluate", "--final", "--root", root])
    assert isinstance(res.exception, pipeline.MissingArtifactError)
    assert not isinstance(res.exception, TestPeriodLockedError)

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

    results = pipeline.stage_simulate(cfg)
    assert set(results["strategy"]) == {"sell_forward", "wait", "fixed_50_50", "forecast_driven"}
    for name in ("strategy_metrics.csv", "stress_scenarios.csv", "sensitivity.csv"):
        assert (p.simulation / name).exists()

    pipeline.stage_report(cfg)
    assert (p.figures / "test_actual_vs_pred_h3.png").exists()
    assert (p.figures / "simulation_strategy_outcomes.png").exists()
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

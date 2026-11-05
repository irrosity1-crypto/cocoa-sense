"""Dashboard smoke tests with Streamlit's AppTest (SYNTHETIC artifacts)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cocoasense import pipeline
from cocoasense.config import ProjectConfig

DASH = Path(__file__).resolve().parents[2] / "src" / "cocoasense" / "dashboard"
PAGES = [DASH / "Overview.py", *sorted((DASH / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_without_artifacts(
    page: Path, project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COCOASENSE_ROOT", str(project.paths.root))
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception
    assert at.warning  # disclaimer banner on every page
    assert any("cocoasense" in i.value for i in at.info)  # run instructions, not an error


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the SYNTHETIC pipeline once to produce artifacts."""
    import shutil

    import yaml

    from cocoasense.config import load_config
    from tests.conftest import (
        REPO_CONFIGS,
        SMALL_OVERRIDES,
        synthetic_history,
        write_fred,
        write_imf,
        write_pinksheet,
    )

    root = tmp_path_factory.mktemp("dash")
    shutil.copytree(REPO_CONFIGS, root / "configs", ignore=shutil.ignore_patterns("frozen*.yaml"))
    for name, override in SMALL_OVERRIDES.items():
        path = root / "configs" / name
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        content.update(override)
        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    cfg = load_config(root)
    cfg.paths.ensure()
    h = synthetic_history()
    write_pinksheet(cfg.paths.manual / "CMO-Historical-Data-Monthly.xlsx", h["price_usd_t"])
    write_fred(cfg.paths.manual / "PCOCOUSDM.csv", h["price_usd_t"].loc["1992":])
    write_imf(cfg.paths.manual / "ER_GHA_XDC_USD_PA_RT_M.xml", h["ghs_per_usd"])
    pipeline.stage_ingest(cfg)
    pipeline.stage_validate(cfg)
    pipeline.stage_train(cfg)
    pipeline.stage_freeze(cfg)
    pipeline.stage_evaluate_final(cfg)
    pipeline.stage_simulate(cfg)
    pipeline.stage_learn(cfg)
    pipeline.stage_insights(cfg)
    return root


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_with_artifacts(page: Path, built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCOASENSE_ROOT", str(built))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception
    assert at.warning


@pytest.mark.slow
def test_forecast_explorer_hides_future_until_revealed(
    built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COCOASENSE_ROOT", str(built))
    at = AppTest.from_file(
        str(DASH / "pages" / "1_Forecast_Explorer.py"), default_timeout=120
    ).run()
    table = at.dataframe[0].value
    assert "Actual US$/t" not in table.columns
    at.toggle[0].set_value(True).run()
    assert "Actual US$/t" in at.dataframe[0].value.columns


@pytest.mark.slow
def test_strategy_simulator_cedi_view(built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCOASENSE_ROOT", str(built))
    at = AppTest.from_file(
        str(DASH / "pages" / "3_Strategy_Simulator.py"), default_timeout=120
    ).run()
    assert "Mean US$/t" in at.dataframe[0].value.columns
    at.radio(key="currency").set_value("Ghana cedis (GHS)").run()
    assert not at.exception
    assert "Mean GHS/t" in at.dataframe[0].value.columns
    assert any("not extrapolated" in c.value for c in at.caption)

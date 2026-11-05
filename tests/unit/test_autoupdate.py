"""Unattended updates: change-only downloads, offline handling, schedule (SYNTHETIC files)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import requests

from cocoasense import pipeline
from cocoasense.autoupdate import new_data_message
from cocoasense.config import ProjectConfig
from cocoasense.desktop import TASK_NAME, schedule_script, toast_script
from cocoasense.ingestion.download import DownloadError
from cocoasense.ingestion.manifest import read_manifest
from cocoasense.ingestion.sources import check_source
from tests.conftest import synthetic_history, write_fred

FRED = "fred_pcocousdm"


def _copying(src_file: Path) -> Callable[[str, Path, Path], None]:
    """Fake downloader that 'downloads' a local file."""

    def fake(url: str, dest: Path, manual: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, dest)

    return fake


def _offline(url: str, dest: Path, manual: Path) -> None:
    raise DownloadError("no internet")


def test_check_source_keeps_file_unless_values_change(
    raw_fixture_files: ProjectConfig, tmp_path: Path
) -> None:
    cfg = raw_fixture_files
    pipeline.stage_ingest(cfg)
    src = cfg.data.sources[FRED]
    dest = cfg.paths.raw / src.file
    before = dest.read_bytes()

    same = tmp_path / "same.csv"
    shutil.copy(dest, same)
    same.write_text(same.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # bytes differ
    assert check_source(FRED, src, cfg, _copying(same)) is False
    assert dest.read_bytes() == before  # values identical: file untouched

    longer = synthetic_history(end="2020-03")["price_usd_t"].loc["1992":]
    newer = tmp_path / "new.csv"
    write_fred(newer, longer)
    assert check_source(FRED, src, cfg, _copying(newer)) is True
    assert read_manifest(cfg.paths.manifest)[src.file]["date_max"] == "2020-03-01"
    archived = list((cfg.paths.raw / "archive").glob(f"2*/{src.file}"))
    assert len(archived) == 1 and archived[0].read_bytes() == before

    with pytest.raises(DownloadError):
        check_source(FRED, src, cfg, _offline)


def test_auto_update_offline_and_unbuilt(raw_fixture_files: ProjectConfig) -> None:
    cfg = raw_fixture_files
    pipeline.stage_ingest(cfg)
    pipeline.stage_validate(cfg)
    rec = pipeline.stage_auto_update(cfg, downloader=_offline, fx_getter=_fx_down)
    assert rec["status"].startswith("project not built")  # no frozen models in a bare fixture
    assert rec["not_checked"] == list(cfg.data.sources)
    log = (cfg.paths.artifacts / "update_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["mode"] == "auto"


def test_auto_update_retrains_only_on_new_data(
    raw_fixture_files: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = raw_fixture_files
    pipeline.stage_ingest(cfg)
    pipeline.stage_validate(cfg)
    cfg.paths.forecasts.mkdir(parents=True, exist_ok=True)
    for name in (pipeline.LEARNING, pipeline.INSIGHTS):  # pretend forecasts already exist
        pd.DataFrame({"x": [1]}).to_parquet(cfg.paths.forecasts / name)
    called: list[str] = []
    monkeypatch.setattr(pipeline, "stage_learn", lambda c: called.append("learn"))
    monkeypatch.setattr(pipeline, "stage_insights", lambda c: called.append("insights"))

    rec = pipeline.stage_auto_update(cfg, downloader=_offline, fx_getter=_fx_down)
    assert rec["status"] == "offline" and called == []

    def same_file(url: str, dest: Path, manual: Path) -> None:
        match = next(s for s in cfg.data.sources.values() if Path(s.file).name == dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg.paths.raw / match.file, dest)

    rec = pipeline.stage_auto_update(cfg, downloader=same_file, fx_getter=_fx_down)
    assert rec["status"] == "no new data" and rec["changed"] == [] and called == []

    longer = synthetic_history(end="2020-03")["price_usd_t"].loc["1992":]

    def new_fred(url: str, dest: Path, manual: Path) -> None:
        if dest.suffix == ".csv":
            write_fred(dest, longer)
        else:
            same_file(url, dest, manual)

    rec = pipeline.stage_auto_update(cfg, downloader=new_fred, fx_getter=_fx_down)
    assert rec["changed"] == [FRED]
    assert rec["status"] == "retrained on new data" and called == ["learn", "insights"]


def test_schedule_script() -> None:
    script = schedule_script(Path("C:/py/pythonw.exe"), Path("C:/repo"), "08:30")
    assert f"-TaskName '{TASK_NAME}'" in script
    assert "-m cocoasense.autoupdate" in script
    assert "-RunOnlyIfNetworkAvailable" in script and "-StartWhenAvailable" in script
    assert "-AtLogOn" in script and "-Daily -At '08:30'" in script


def _fx_down(url: str) -> dict[str, object]:
    raise requests.ConnectionError("offline")


def _fx_up(url: str) -> dict[str, object]:
    return {"date": "2019-12-20", "usd": {"ghs": 5.6}}


def test_stage_fx_offline_then_online(raw_fixture_files: ProjectConfig) -> None:
    cfg = raw_fixture_files
    pipeline.stage_ingest(cfg)
    pipeline.stage_validate(cfg)
    assert pipeline.stage_fx(cfg, _fx_down) == {"status": "offline"}
    rec = pipeline.stage_fx(cfg, _fx_up)
    assert rec["status"] == "ok" and rec["latest_rate"] == 5.6
    assert (cfg.paths.artifacts / "fx" / pipeline.FX_DAILY).exists()


def test_new_data_message(raw_fixture_files: ProjectConfig) -> None:
    cfg = raw_fixture_files
    pipeline.stage_ingest(cfg)
    pipeline.stage_validate(cfg)
    title, text = new_data_message(cfg, {"fx": {"status": "ok", "latest_rate": 10.0}})
    assert "new cocoa price" in title and "December 2019" in text and "GH₵" in text
    _, plain = new_data_message(cfg, {"fx": {"status": "offline"}})
    assert "GH₵" not in plain and plain.endswith("Forecasts have been updated.")


def test_toast_script_escapes_text() -> None:
    script = toast_script("Price <up> & more", "it's 5,000")
    assert "&lt;up&gt; &amp; more" in script
    assert "it''s 5,000" in script  # single quote doubled for PowerShell
    assert "CreateToastNotifier" in script

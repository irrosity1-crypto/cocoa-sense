from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import requests

from cocoasense.config import ProjectConfig
from cocoasense.ingestion import download as dl
from cocoasense.ingestion.manifest import (
    ManifestEntry,
    read_manifest,
    sha256_of,
    verify,
    write_entry,
)
from cocoasense.ingestion.sources import ingest_all
from tests.conftest import synthetic_history, write_fred


def test_manifest_roundtrip_and_verify(tmp_path: Path) -> None:
    f = tmp_path / "raw" / "a.csv"
    f.parent.mkdir()
    f.write_text("x\n1\n")
    e = ManifestEntry(
        "a.csv",
        "src",
        "http://x",
        "download",
        "2026-01-01T00:00:00+00:00",
        sha256_of(f),
        f.stat().st_size,
        1,
        "2000-01-01",
        "2000-01-01",
    )
    m = tmp_path / "raw" / "manifest.json"
    write_entry(m, e)
    assert read_manifest(m)["a.csv"]["sha256"] == e.sha256
    assert verify(m, tmp_path / "raw") == []
    f.write_text("changed")
    assert "differs" in verify(m, tmp_path / "raw")[0]
    f.unlink()
    assert "missing" in verify(m, tmp_path / "raw")[0]


def test_ingest_uses_manual_files_and_records_manifest(raw_fixture_files: ProjectConfig) -> None:
    cfg = raw_fixture_files

    def no_network(url: str, dest: Path, manual: Path) -> None:
        raise AssertionError("should not download when manual files exist")

    entries = ingest_all(cfg, downloader=no_network)
    assert {e.origin for e in entries} == {"manual"}
    man = read_manifest(cfg.paths.manifest)
    pink = man["worldbank/CMO-Historical-Data-Monthly.xlsx"]
    assert pink["rows"] == len(synthetic_history())
    assert pink["date_min"] == "1975-01-01"


def test_ingest_downloads_and_refresh_archives(project: ProjectConfig, tmp_path: Path) -> None:
    src_dir = tmp_path / "fake_web"
    h = synthetic_history()
    write_fred(src_dir / "fred.csv", h["price_usd_t"])
    from tests.conftest import write_imf, write_pinksheet

    write_pinksheet(src_dir / "pink.xlsx", h["price_usd_t"])
    write_imf(src_dir / "imf.xml", h["ghs_per_usd"])
    by_suffix = {".csv": "fred.csv", ".xlsx": "pink.xlsx", ".xml": "imf.xml"}
    calls: list[str] = []

    def fake_download(url: str, dest: Path, manual: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / by_suffix[dest.suffix], dest)

    ingest_all(project, downloader=fake_download)
    assert len(calls) == 3
    ingest_all(project, downloader=fake_download)  # reuse, no new downloads
    assert len(calls) == 3
    ingest_all(project, refresh=True, downloader=fake_download)
    assert len(calls) == 6
    assert any((project.paths.raw / "archive").rglob("PCOCOUSDM.csv"))


def test_download_error_asks_for_manual_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise requests.ConnectionError("blocked by proxy")

    monkeypatch.setattr(dl.requests, "get", boom)
    with pytest.raises(dl.DownloadError, match="manual"):
        dl.download("http://example.invalid/x.csv", tmp_path / "x.csv", tmp_path / "manual")
    assert not (tmp_path / "x.csv").exists()


def test_download_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        content = b"a,b\n1,2\n"

        def raise_for_status(self) -> None:
            return None

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Resp())
    dl.download("http://example.invalid/x.csv", tmp_path / "d" / "x.csv", tmp_path)
    assert (tmp_path / "d" / "x.csv").read_bytes() == Resp.content

"""Obtain each configured raw file (download or manual drop-in) and record it in the manifest.

Rules:
* Files under ``data/raw/`` are never edited.
* An existing raw file is reused unless ``refresh=True``; a refresh archives the old copy first.
* A file placed in ``data/raw/manual/`` with the same name as the source file takes priority
  over downloading and is recorded with ``origin="manual"``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cocoasense.config import ProjectConfig, SourceConfig
from cocoasense.ingestion.download import download
from cocoasense.ingestion.manifest import ManifestEntry, read_manifest, sha256_of, write_entry
from cocoasense.validation.parsers import PARSERS

logger = logging.getLogger(__name__)

Downloader = Callable[[str, Path, Path], None]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _archive(path: Path, raw_dir: Path) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = raw_dir / "archive" / stamp / path.relative_to(raw_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    logger.info("archived %s -> %s", path, target)


def obtain_source(
    key: str,
    src: SourceConfig,
    cfg: ProjectConfig,
    refresh: bool = False,
    downloader: Downloader = download,
) -> ManifestEntry:
    """Make sure the raw file for ``key`` exists and its manifest entry is current."""
    paths = cfg.paths
    dest = paths.raw / src.file
    manual = paths.manual / Path(src.file).name
    existing = read_manifest(paths.manifest).get(src.file)

    origin = existing["origin"] if existing else "download"
    retrieved_at = existing["retrieved_at"] if existing else _now()

    if manual.exists() and (not dest.exists() or sha256_of(manual) != sha256_of(dest)):
        if dest.exists():
            _archive(dest, paths.raw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manual, dest)
        origin, retrieved_at = "manual", _now()
        logger.info("using manual file %s for %s", manual, key)
    elif refresh or not dest.exists():
        if dest.exists():
            _archive(dest, paths.raw)
        downloader(src.url, dest, paths.manual)
        origin, retrieved_at = "download", _now()
    else:
        logger.info("reusing raw file %s", dest)

    series = PARSERS[key](dest)
    entry = ManifestEntry(
        path=src.file,
        source=key,
        url=src.url,
        origin=origin,
        retrieved_at=retrieved_at,
        sha256=sha256_of(dest),
        bytes=dest.stat().st_size,
        rows=int(series.notna().sum()),
        date_min=_iso(series.first_valid_index()),
        date_max=_iso(series.last_valid_index()),
    )
    if existing and existing["sha256"] != entry.sha256 and origin == existing["origin"]:
        logger.warning("%s changed on disk since the manifest was written", src.file)
    write_entry(paths.manifest, entry)
    return entry


def _iso(ts: object) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d") if ts is not None else ""


def ingest_all(
    cfg: ProjectConfig, refresh: bool = False, downloader: Downloader = download
) -> list[ManifestEntry]:
    """Obtain every configured source."""
    cfg.paths.ensure()
    return [
        obtain_source(key, src, cfg, refresh=refresh, downloader=downloader)
        for key, src in cfg.data.sources.items()
    ]

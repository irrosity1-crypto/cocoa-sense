"""``data/raw/manifest.json``: provenance record for every raw file.

Each entry stores source key, URL, retrieval time, SHA-256, size, row count and date range,
so that any result can be traced back to the exact bytes it was computed from.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    """Provenance for one raw file (path is relative to ``data/raw``)."""

    path: str
    source: str
    url: str
    origin: str  # "download" or "manual"
    retrieved_at: str  # ISO-8601 UTC
    sha256: str
    bytes: int
    rows: int
    date_min: str
    date_max: str


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read the manifest, returning ``{relative_path: entry_dict}`` (empty if absent)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        content = json.load(fh)
    files = content.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"malformed manifest: {path}")
    return files


def write_entry(path: Path, entry: ManifestEntry) -> None:
    """Insert or replace one entry and rewrite the manifest deterministically."""
    files = read_manifest(path)
    files[entry.path] = asdict(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({"files": dict(sorted(files.items()))}, fh, indent=2)
        fh.write("\n")
    logger.info("manifest updated: %s (sha256 %s…)", entry.path, entry.sha256[:12])


def verify(path: Path, raw_dir: Path) -> list[str]:
    """Return a list of problems: missing files or hashes that no longer match."""
    problems = []
    for rel, entry in read_manifest(path).items():
        f = raw_dir / rel
        if not f.exists():
            problems.append(f"{rel}: listed in manifest but missing on disk")
        elif sha256_of(f) != entry["sha256"]:
            problems.append(f"{rel}: SHA-256 differs from manifest (file changed?)")
    return problems

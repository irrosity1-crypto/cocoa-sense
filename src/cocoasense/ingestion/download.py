"""Plain HTTP download with a clear manual fallback message. No workarounds for blocks."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "cocoasense-ai/0.1 (university final-year project; contact via GitHub)"


class DownloadError(RuntimeError):
    """Raised when a source cannot be fetched; the message tells the user what to do."""


def download(url: str, dest: Path, manual_dir: Path, timeout: float = 60.0) -> None:
    """Download ``url`` to ``dest`` atomically, or raise ``DownloadError``.

    If the network, a proxy or the server refuses the request we stop and ask the user to place
    the file in ``data/raw/manual/`` rather than trying to get around the block.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}) as resp:
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"Could not download {url} ({exc}).\n"
            f"Download it in a browser and save it as {manual_dir / dest.name}, "
            "then rerun `cocoasense ingest`."
        ) from exc
    tmp.replace(dest)
    logger.info("downloaded %s -> %s (%d bytes)", url, dest, dest.stat().st_size)

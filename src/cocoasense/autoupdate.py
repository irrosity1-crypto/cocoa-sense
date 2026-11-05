"""Entry point for the scheduled automatic update (``pythonw -m cocoasense.autoupdate``).

Runs without a console, so everything is logged to auto_update.log in the CocoaSense app folder
(``%LOCALAPPDATA%/CocoaSense``).
"""

from __future__ import annotations

import logging

from cocoasense import pipeline
from cocoasense.config import load_config
from cocoasense.desktop import app_dir


def main() -> None:  # pragma: no cover - run by the Windows scheduler
    logging.basicConfig(
        filename=app_dir() / "auto_update.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        encoding="utf-8",
    )
    log = logging.getLogger("cocoasense.autoupdate")
    try:
        record = pipeline.stage_auto_update(load_config())
        log.info("auto-update finished: %s", record)
    except Exception:
        log.exception("auto-update failed")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()

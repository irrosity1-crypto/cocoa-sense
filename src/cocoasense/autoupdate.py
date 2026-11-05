"""Entry point for the scheduled automatic update (``pythonw -m cocoasense.autoupdate``).

Runs without a console, so everything is logged to auto_update.log in the CocoaSense app folder
(``%LOCALAPPDATA%/CocoaSense``).
"""

from __future__ import annotations

import logging
from typing import Any

from cocoasense import pipeline
from cocoasense.config import ProjectConfig, load_config
from cocoasense.desktop import app_dir, notify


def new_data_message(cfg: ProjectConfig, record: dict[str, Any]) -> tuple[str, str]:
    """Title and text for the 'new price published' notification."""
    monthly = pipeline.load_monthly(cfg)
    price = monthly["price_usd_t"].dropna()
    text = f"{price.index[-1]:%B %Y} average: US${price.iloc[-1]:,.0f}/t"
    fx = record.get("fx") or {}
    if fx.get("status") == "ok":
        text += f" (about GH₵{price.iloc[-1] * fx['latest_rate']:,.0f}/t)"
    return "CocoaSense: new cocoa price", text + ". Forecasts have been updated."


def main() -> None:  # pragma: no cover - run by the Windows scheduler
    logging.basicConfig(
        filename=app_dir() / "auto_update.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        encoding="utf-8",
    )
    log = logging.getLogger("cocoasense.autoupdate")
    try:
        cfg = load_config()
        record = pipeline.stage_auto_update(cfg)
        log.info("auto-update finished: %s", record)
        if record["status"] == "retrained on new data":
            notify(*new_data_message(cfg, record))
    except Exception:
        log.exception("auto-update failed")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()

"""Live US$ to cedi rate: mirrors, fallback, backfill, combination (SYNTHETIC responses)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from cocoasense.ingestion.fx_live import (
    CLOUDFLARE,
    JSDELIVR,
    FxUnavailableError,
    combined_monthly,
    fetch_rate,
    latest,
    read_daily,
    sample_dates,
    update_fx,
)


def fake_api(rates: dict[str, float], latest_day: str, down: tuple[str, ...] = ()) -> Any:
    """Getter answering like the API: ``rates`` by date tag; mirrors listed in ``down`` fail."""

    def get(url: str) -> dict[str, Any]:
        for name in down:
            if name in url:
                raise requests.ConnectionError(f"{name} unreachable")
        if "@latest" in url or "//latest." in url:
            tag = latest_day
        elif "jsdelivr" in url:
            tag = url.split("@")[-1][:10]
        else:
            tag = url.split("//")[1][:10]
        if tag not in rates:
            raise requests.HTTPError("404")
        return {"date": tag, "usd": {"ghs": rates[tag]}}

    return get


def test_fetch_rate_latest_and_fallback() -> None:
    api = fake_api({"2026-11-05": 11.65}, "2026-11-05")
    assert fetch_rate(None, api) == (date(2026, 11, 5), 11.65, "currency-api (jsDelivr)")
    down = fake_api({"2026-11-05": 11.65}, "2026-11-05", down=("jsdelivr",))
    assert fetch_rate(None, down)[2] == "currency-api (Cloudflare)"
    with pytest.raises(FxUnavailableError):
        fetch_rate(None, fake_api({}, "2026-11-05", down=("jsdelivr", "pages.dev")))
    with pytest.raises(FxUnavailableError, match="implausible"):
        fetch_rate(None, fake_api({"2026-11-05": -3.0}, "2026-11-05"))
    assert "{tag}" in JSDELIVR and "{tag}" in CLOUDFLARE


def test_sample_dates() -> None:
    days = sample_dates(pd.Timestamp("2026-01-01"), date(2026, 3, 10))
    assert days == [date(2026, 2, d) for d in (1, 8, 15, 22)] + [date(2026, 3, 1), date(2026, 3, 8)]


def test_update_fx_backfills_and_is_incremental(tmp_path: Path) -> None:
    path = tmp_path / "fx" / "daily.csv"
    imf = pd.Series([10.5, 10.8], index=pd.to_datetime(["2025-12-01", "2026-01-01"]))
    rates = {f"2026-02-{d:02d}": 10.9 + d / 100 for d in (1, 8, 15, 22)}
    rates |= {"2026-03-01": 11.0, "2026-03-03": 11.1}
    api = fake_api(rates, "2026-03-03")
    rec = update_fx(path, imf, today=date(2026, 3, 3), getter=api)
    assert rec == {
        "latest_date": "2026-03-03",
        "latest_rate": 11.1,
        "source": "currency-api (jsDelivr)",
        "added": 6,
    }
    daily = read_daily(path)
    assert len(daily) == 6 and daily["date"].is_monotonic_increasing
    again = update_fx(path, imf, today=date(2026, 3, 3), getter=api)
    assert again["added"] == 0 and len(read_daily(path)) == 6

    monthly = combined_monthly(imf, daily)
    assert monthly.loc["2026-01-01", "source"] == "IMF monthly average"
    assert monthly.loc["2026-02-01", "ghs_per_usd"] == pytest.approx(10.9 + 46 / 400)
    assert monthly.loc["2026-03-01", "source"] == "currency-api daily average"
    assert latest(daily) == (pd.Timestamp("2026-03-03"), 11.1, "currency-api (jsDelivr)")


def test_update_fx_offline_and_partial(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    imf = pd.Series([10.8], index=pd.to_datetime(["2026-01-01"]))
    with pytest.raises(FxUnavailableError):
        update_fx(path, imf, today=date(2026, 2, 10), getter=fake_api({}, "2026-02-10"))
    assert not path.exists()
    # The latest rate works but most backfill days are missing: they are skipped, not invented.
    api = fake_api({"2026-02-10": 11.0, "2026-02-01": 10.9}, "2026-02-10")
    rec = update_fx(path, imf, today=date(2026, 2, 10), getter=api)
    assert rec["added"] == 2
    assert set(read_daily(path)["date"].dt.day) == {1, 10}
    empty = read_daily(tmp_path / "none.csv")
    assert latest(empty) is None
    assert combined_monthly(imf, empty).shape == (1, 2)

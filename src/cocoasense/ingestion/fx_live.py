"""Live US$ to Ghana cedi rate from a free, keyless, public-domain API (decision D21).

Source: fawazahmed0/exchange-api ("currency-api"), CC0-1.0, no API key, no rate limits, daily
rates with dated history. Two mirrors are tried in turn (jsDelivr, then Cloudflare Pages).

The rates are for display and conversion only: no forecasting model uses them, so the frozen
evaluation and the learning track are unaffected.

* ``update_fx`` appends the latest rate to ``artifacts/fx/usd_ghs_daily.csv`` and backfills the
  months after the last IMF monthly observation with sampled days (1st, 8th, 15th, 22nd).
* ``combined_monthly`` gives one GHS-per-US$ value per month: the IMF monthly average where it
  exists, otherwise the average of the stored daily API rates, with the source recorded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

JSDELIVR = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{tag}/v1/currencies/usd.min.json"
CLOUDFLARE = "https://{tag}.currency-api.pages.dev/v1/currencies/usd.min.json"
MIRRORS = (("currency-api (jsDelivr)", JSDELIVR), ("currency-api (Cloudflare)", CLOUDFLARE))
SAMPLE_DAYS = (1, 8, 15, 22)
COLUMNS = ["date", "ghs_per_usd", "source"]

Getter = Callable[[str], dict[str, Any]]


class FxUnavailableError(RuntimeError):
    """No mirror answered (offline or blocked); the next run retries."""


def _get_json(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "cocoasense-ai (student project)"})
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def fetch_rate(day: date | None = None, getter: Getter = _get_json) -> tuple[date, float, str]:
    """GHS per US$ for ``day`` (``None`` = latest): (rate date, rate, source name)."""
    tag = "latest" if day is None else day.isoformat()
    errors = []
    for name, template in MIRRORS:
        try:
            data = getter(template.format(tag=tag))
            rate = float(data["usd"]["ghs"])
            if not 0.0001 < rate < 1000:
                raise ValueError(f"implausible rate {rate}")
            return date.fromisoformat(data["date"]), rate, name
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    raise FxUnavailableError("; ".join(errors))


def read_daily(path: Path) -> pd.DataFrame:
    """Stored daily rates (empty frame if none yet)."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df[COLUMNS]


def sample_dates(after_month: pd.Timestamp, today: date) -> list[date]:
    """Sampled days in every month after ``after_month`` up to ``today``."""
    out = []
    month = pd.Timestamp(after_month).to_period("M") + 1
    while month.start_time.date() <= today:
        for d in SAMPLE_DAYS:
            day = date(month.year, month.month, d)
            if day <= today:
                out.append(day)
        month += 1
    return out


def update_fx(
    path: Path, imf: pd.Series, today: date | None = None, getter: Getter = _get_json
) -> dict[str, Any]:
    """Fetch the latest rate and backfill sampled days after the IMF series ends.

    Raises ``FxUnavailableError`` only if the latest rate cannot be fetched; a missing backfill
    day is skipped and retried on the next run.
    """
    today = today or datetime.now(UTC).date()
    daily = read_daily(path)
    have = {pd.Timestamp(d).date() for d in daily["date"]}
    rows = []
    latest_day, latest_rate, source = fetch_rate(None, getter)
    if latest_day not in have:
        rows.append({"date": latest_day, "ghs_per_usd": latest_rate, "source": source})
        have.add(latest_day)
    imf_last = imf.dropna().index.max()
    if pd.notna(imf_last):
        for day in sample_dates(pd.Timestamp(imf_last), today):
            if day in have:
                continue
            try:
                got_day, rate, src = fetch_rate(day, getter)
            except FxUnavailableError as exc:
                logger.warning("no rate for %s: %s", day, exc)
                continue
            if got_day not in have:
                rows.append({"date": got_day, "ghs_per_usd": rate, "source": src})
                have.add(got_day)
    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        new["date"] = pd.to_datetime(new["date"])
        daily = new if daily.empty else pd.concat([daily, new], ignore_index=True)
        daily = daily.sort_values("date").drop_duplicates("date", keep="first")
        path.parent.mkdir(parents=True, exist_ok=True)
        daily.assign(date=pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")).to_csv(
            path, index=False
        )
    return {
        "latest_date": latest_day.isoformat(),
        "latest_rate": latest_rate,
        "source": source,
        "added": len(rows),
    }


def combined_monthly(imf: pd.Series, daily: pd.DataFrame) -> pd.DataFrame:
    """One rate per month: IMF monthly average, else the mean of stored daily API rates."""
    out = pd.DataFrame({"ghs_per_usd": imf.dropna(), "source": "IMF monthly average"})
    if not daily.empty:
        months = pd.to_datetime(daily["date"]).dt.to_period("M").dt.to_timestamp()
        api = daily.groupby(months)["ghs_per_usd"].mean()
        api = api[~api.index.isin(out.index)]
        extra = pd.DataFrame({"ghs_per_usd": api, "source": "currency-api daily average"})
        out = pd.concat([out, extra])
    out.index = pd.DatetimeIndex(out.index, name="month")
    return out.sort_index()


def latest(daily: pd.DataFrame) -> tuple[pd.Timestamp, float, str] | None:
    """Most recent stored daily rate."""
    if daily.empty:
        return None
    row = daily.sort_values("date").iloc[-1]
    return pd.Timestamp(row["date"]), float(row["ghs_per_usd"]), str(row["source"])

"""Data-quality checks. Errors stop the pipeline; warnings are reported but not acted on.

Nothing is silently "fixed": outliers are listed, not removed, and gaps are reported, not filled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cocoasense.config import ValidationThresholds

logger = logging.getLogger(__name__)

Level = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Issue:
    """One finding from a check."""

    series: str
    level: Level
    check: str
    message: str


@dataclass
class QualityReport:
    """All findings plus a short summary per series."""

    issues: list[Issue] = field(default_factory=list)
    summaries: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    def add(self, series: str, level: Level, check: str, message: str) -> None:
        self.issues.append(Issue(series, level, check, message))

    def to_markdown(self) -> str:
        lines = ["# Data-quality report", ""]
        lines += ["## Series summary", "", "| series | rows | first | last | min | max |"]
        lines += ["|---|---|---|---|---|---|"]
        for name, s in self.summaries.items():
            lines.append(
                f"| {name} | {s['rows']} | {s['first']} | {s['last']} | {s['min']} | {s['max']} |"
            )
        lines += ["", "## Findings", ""]
        if not self.issues:
            lines.append("No findings.")
        for level in ("error", "warning", "info"):
            items = [i for i in self.issues if i.level == level]
            if items:
                lines += [f"### {level.title()}s ({len(items)})", ""]
                lines += [f"- **{i.series}** · {i.check}: {i.message}" for i in items]
                lines.append("")
        return "\n".join(lines) + "\n"

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data_quality.md").write_text(self.to_markdown(), encoding="utf-8")
        payload = {"issues": [asdict(i) for i in self.issues], "summaries": self.summaries}
        (directory / "data_quality.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )


class DataQualityError(RuntimeError):
    """Raised when at least one error-level issue is found."""


def summarise(s: pd.Series) -> dict[str, object]:
    """Row count, date range and value range of a series."""
    valid = s.dropna()
    return {
        "rows": int(valid.size),
        "first": valid.index.min().strftime("%Y-%m") if valid.size else "",
        "last": valid.index.max().strftime("%Y-%m") if valid.size else "",
        "min": round(float(valid.min()), 4) if valid.size else None,
        "max": round(float(valid.max()), 4) if valid.size else None,
    }


def check_monthly_index(s: pd.Series, name: str, report: QualityReport) -> None:
    """Month-start index, sorted, no duplicates, gaps reported."""
    idx = pd.DatetimeIndex(s.index)
    if idx.has_duplicates:
        dups = idx[idx.duplicated()].strftime("%Y-%m").tolist()
        report.add(name, "error", "duplicates", f"duplicate months: {dups[:10]}")
    if not idx.is_monotonic_increasing:
        report.add(name, "error", "order", "index is not sorted by date")
    if not (idx == idx.to_period("M").to_timestamp()).all():
        report.add(name, "error", "month_start", "index values are not month-start dates")
    valid = s.dropna()
    if valid.empty:
        report.add(name, "error", "empty", "series has no valid values")
        return
    full = pd.date_range(valid.index.min(), valid.index.max(), freq="MS")
    missing = full.difference(valid.index)
    if len(missing):
        report.add(
            name,
            "error",
            "gaps",
            f"{len(missing)} missing months inside the range, e.g. "
            f"{missing[:5].strftime('%Y-%m').tolist()}",
        )
    else:
        report.add(name, "info", "gaps", "no missing months between first and last observation")


def check_range(
    s: pd.Series, name: str, lo: float, hi: float, unit: str, report: QualityReport
) -> None:
    """Values must be positive and inside a plausible range."""
    valid = s.dropna()
    if (valid <= 0).any():
        report.add(name, "error", "positive", f"{int((valid <= 0).sum())} non-positive values")
    out = valid[(valid < lo) | (valid > hi)]
    if len(out):
        report.add(
            name,
            "error",
            "range",
            f"{len(out)} values outside [{lo}, {hi}] {unit}, e.g. {out.head(3).round(4).to_dict()}",
        )


def check_jumps(s: pd.Series, name: str, threshold: float, report: QualityReport) -> None:
    """List large monthly log changes (kept in the data, reported for transparency)."""
    r = np.log(s.dropna()).diff().dropna()
    big = r[r.abs() > threshold]
    if len(big):
        listing = ", ".join(f"{d:%Y-%m} ({v:+.2f})" for d, v in big.items())
        report.add(name, "warning", "large_moves", f"monthly |log change| > {threshold}: {listing}")


def check_redenomination(fx: pd.Series, report: QualityReport) -> None:
    """The 2007 cedi redenomination (10,000 old = 1 new) would show as a ~9.2 log-point jump.

    We require the series to be continuous (expressed in new cedis throughout).
    """
    r = np.log(fx.dropna()).diff().abs()
    if (r > np.log(100)).any():
        when = r[r > np.log(100)].index.strftime("%Y-%m").tolist()
        report.add("ghs_per_usd", "error", "redenomination", f"break detected at {when}")
    else:
        window = fx.loc["2007-05":"2007-09"].round(4).to_dict()
        report.add(
            "ghs_per_usd",
            "info",
            "redenomination",
            "no break around July 2007; series is in new cedis throughout. "
            f"May–Sep 2007: { {k.strftime('%Y-%m'): v for k, v in window.items()} }",
        )


def crosscheck(
    primary: pd.Series, other: pd.Series, t: ValidationThresholds, report: QualityReport
) -> pd.Series:
    """Compare the primary price with an independent series over their overlap."""
    joined = pd.concat([primary, other], axis=1, keys=["primary", "other"]).dropna()
    pct = (joined["primary"] - joined["other"]).abs() / joined["other"] * 100
    mean_pct = float(pct.mean())
    msg = (
        f"{len(joined)} overlapping months; mean |diff| {mean_pct:.2f}%, "
        f"median {pct.median():.2f}%, correlation {joined.corr().iloc[0, 1]:.4f}"
    )
    level: Level = "error" if mean_pct > t.crosscheck_mean_tolerance_pct else "info"
    report.add("price_usd_t", level, "crosscheck", msg)
    flagged = pct[pct > t.crosscheck_month_flag_pct]
    if len(flagged):
        listing = ", ".join(f"{d:%Y-%m} ({v:.1f}%)" for d, v in flagged.items())
        report.add(
            "price_usd_t",
            "warning",
            "crosscheck_months",
            f"months differing by more than {t.crosscheck_month_flag_pct}%: {listing}",
        )
    return pct


def validate(
    price: pd.Series,
    fx: pd.Series,
    crosscheck_price: pd.Series | None,
    t: ValidationThresholds,
) -> QualityReport:
    """Run every check and return the report (does not raise)."""
    report = QualityReport()
    for name, s in (("price_usd_t", price), ("ghs_per_usd", fx)):
        report.summaries[name] = summarise(s)
        check_monthly_index(s, name, report)
    check_range(price, "price_usd_t", t.price_min_usd_t, t.price_max_usd_t, "USD/t", report)
    check_range(fx, "ghs_per_usd", t.fx_min_ghs_per_usd, t.fx_max_ghs_per_usd, "GHS/USD", report)
    check_jumps(price, "price_usd_t", t.max_abs_monthly_log_change_warn, report)
    check_jumps(fx, "ghs_per_usd", t.max_abs_monthly_log_change_warn, report)
    check_redenomination(fx, report)
    if crosscheck_price is not None:
        report.summaries["crosscheck_price_usd_t"] = summarise(crosscheck_price)
        crosscheck(price, crosscheck_price, t, report)
    return report

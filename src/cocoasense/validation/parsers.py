"""Parse each raw file into a monthly ``pd.Series`` indexed by month-start dates.

Every parser documents the unit it returns. Prices are returned in US dollars per metric tonne
(``price_usd_t``); the exchange rate in Ghana cedis per US dollar (``ghs_per_usd``).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

_PERIOD = re.compile(r"^\d{4}M\d{2}$")


class ParseError(ValueError):
    """Raised when a raw file does not have the expected structure."""


def _monthly(values: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").astype(float)
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    s.index.name = "date"
    s.name = name
    return s.sort_index()


def parse_pinksheet(path: Path) -> pd.Series:
    """World Bank Pink Sheet monthly workbook -> cocoa price in USD per tonne.

    The workbook reports cocoa in US dollars per kilogram; we multiply by 1,000.
    """
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None, dtype=object)
    header_row = None
    for i in range(min(15, len(raw))):
        if "Cocoa" in [str(v).strip() for v in raw.iloc[i].tolist()]:
            header_row = i
            break
    if header_row is None:
        raise ParseError(f"{path}: no 'Cocoa' column header found in 'Monthly Prices'")
    headers = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    col = headers.index("Cocoa")
    unit = str(raw.iloc[header_row + 1, col]).strip()
    if unit != "($/kg)":
        raise ParseError(f"{path}: expected cocoa unit '($/kg)', found '{unit}'")
    body = raw.iloc[header_row + 2 :, [0, col]]
    body = body[body.iloc[:, 0].astype(str).str.match(_PERIOD)]
    if body.empty:
        raise ParseError(f"{path}: no monthly rows like '1960M01' found")
    dates = pd.to_datetime(body.iloc[:, 0].astype(str).str.replace("M", "-"), format="%Y-%m")
    values = pd.Series(body.iloc[:, 1].to_numpy(), index=dates)
    return _monthly(values, "price_usd_t") * 1000.0


def parse_fred(path: Path) -> pd.Series:
    """FRED CSV (``observation_date``/``DATE`` + series column) -> USD per tonne."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in {"observation_date", "date"}), None)
    if date_col is None or len(df.columns) != 2:
        raise ParseError(f"{path}: expected two columns (date, value), got {list(df.columns)}")
    value_col = next(c for c in df.columns if c != date_col)
    values = df[value_col].replace(".", np.nan)
    return _monthly(pd.Series(values.to_numpy(), index=pd.to_datetime(df[date_col])), "price_usd_t")


def parse_imf_er(path: Path) -> pd.Series:
    """IMF ER SDMX-ML (structure-specific) -> Ghana cedis per US dollar, period average.

    Indicator ``XDC_USD`` is "domestic currency per US dollar", so no inversion is needed.
    """
    root = ET.parse(path).getroot()
    series = [el for el in root.iter() if el.tag.split("}")[-1] == "Series"]
    if len(series) != 1:
        raise ParseError(f"{path}: expected exactly one <Series>, found {len(series)}")
    attrs = series[0].attrib
    expected = {"COUNTRY": "GHA", "INDICATOR": "XDC_USD", "FREQUENCY": "M"}
    for key, val in expected.items():
        if attrs.get(key) != val:
            raise ParseError(f"{path}: series {key}={attrs.get(key)!r}, expected {val!r}")
    dates, values = [], []
    for obs in series[0]:
        if obs.tag.split("}")[-1] != "Obs":
            continue
        period = obs.attrib["TIME_PERIOD"]  # e.g. "1990-M01"
        dates.append(pd.Timestamp(period.replace("-M", "-") + "-01"))
        values.append(obs.attrib.get("OBS_VALUE"))
    if not dates:
        raise ParseError(f"{path}: no observations")
    return _monthly(pd.Series(values, index=dates), "ghs_per_usd")


PARSERS: dict[str, Callable[[Path], pd.Series]] = {
    "worldbank_pinksheet": parse_pinksheet,
    "fred_pcocousdm": parse_fred,
    "imf_er_ghs_usd": parse_imf_er,
}

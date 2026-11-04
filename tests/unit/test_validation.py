"""Validator tests use deliberately broken SYNTHETIC data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cocoasense.config import load_config
from cocoasense.validation.checks import validate
from cocoasense.validation.parsers import ParseError, parse_fred, parse_imf_er, parse_pinksheet
from tests.conftest import synthetic_history, write_fred, write_imf, write_pinksheet

T = load_config().data.validation


def _checks(report: object) -> set[str]:
    return {i.check for i in report.errors}  # type: ignore[attr-defined]


def test_pinksheet_parser_converts_kg_to_tonne(tmp_path: Path) -> None:
    h = synthetic_history()
    write_pinksheet(tmp_path / "p.xlsx", h["price_usd_t"])
    s = parse_pinksheet(tmp_path / "p.xlsx")
    assert s.name == "price_usd_t"
    assert len(s) == len(h)
    np.testing.assert_allclose(s.to_numpy(), h["price_usd_t"].to_numpy(), rtol=1e-6)


def test_pinksheet_parser_rejects_wrong_unit(tmp_path: Path) -> None:
    path = tmp_path / "p.xlsx"
    rows = [[None, "Cocoa"], [None, "($/mt)"], ["2000M01", 1.0]]
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Monthly Prices", header=False, index=False)
    with pytest.raises(ParseError, match="unit"):
        parse_pinksheet(path)


def test_fred_parser_handles_missing_dot(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_text("observation_date,PCOCOUSDM\n2000-01-01,900\n2000-02-01,.\n")
    s = parse_fred(path)
    assert s.isna().sum() == 1 and s.iloc[0] == 900
    write_fred(tmp_path / "g.csv", synthetic_history()["price_usd_t"])
    assert parse_fred(tmp_path / "g.csv").notna().all()


def test_imf_parser_checks_series_identity(tmp_path: Path) -> None:
    fx = synthetic_history()["ghs_per_usd"]
    write_imf(tmp_path / "ok.xml", fx)
    s = parse_imf_er(tmp_path / "ok.xml")
    assert s.name == "ghs_per_usd" and s.index.min() == pd.Timestamp("1990-01-01")
    write_imf(tmp_path / "bad.xml", fx, country="CIV")
    with pytest.raises(ParseError, match="COUNTRY"):
        parse_imf_er(tmp_path / "bad.xml")


def test_clean_synthetic_data_passes() -> None:
    h = synthetic_history()
    report = validate(h["price_usd_t"], h["ghs_per_usd"].dropna(), h["price_usd_t"], T)
    assert report.errors == []


def test_broken_data_is_caught() -> None:
    h = synthetic_history()
    price = h["price_usd_t"].copy()
    price.iloc[10] = -5.0  # negative
    price.iloc[20] = 50_000.0  # implausible
    price = price.drop(price.index[30])  # gap
    price = pd.concat([price, price.iloc[[5]]]).sort_index()  # duplicate
    fx = h["ghs_per_usd"].dropna().copy()
    fx.loc["2007-07-01":] *= 10_000  # unhandled redenomination break
    report = validate(price, fx, None, T)
    assert {"positive", "range", "gaps", "duplicates", "redenomination"} <= _checks(report)


def test_crosscheck_fails_on_large_disagreement() -> None:
    h = synthetic_history()
    report = validate(h["price_usd_t"], h["ghs_per_usd"].dropna(), h["price_usd_t"] * 1.5, T)
    assert "crosscheck" in _checks(report)


def test_report_saved(tmp_path: Path) -> None:
    h = synthetic_history()
    price = h["price_usd_t"].copy()
    price.iloc[100] *= 2  # large jump -> warning only
    report = validate(price, h["ghs_per_usd"].dropna(), None, T)
    report.save(tmp_path)
    text = (tmp_path / "data_quality.md").read_text(encoding="utf-8")
    assert "large_moves" in text and (tmp_path / "data_quality.json").exists()

"""Parsers against real payloads recorded with `app.cli record-fixtures` (skipped if absent)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from app.providers.nse import parse_bhavcopy, parse_constituents, parse_fii_dii, parse_holidays
from tests.conftest import FIXTURES

REC = FIXTURES / "recorded"
pytestmark = pytest.mark.skipif(not REC.exists(), reason="no recorded fixtures yet")


def _rows(name: str) -> list[dict[str, str]]:
    text = (REC / name).read_text(encoding="utf-8-sig")
    return [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]


def _need(name: str) -> None:
    if not (REC / name).exists():
        pytest.skip(f"{name} not recorded")


def test_recorded_fii_dii() -> None:
    _need("nse_fiidii.json")
    rows = parse_fii_dii(json.loads((REC / "nse_fiidii.json").read_text()))
    assert {r.category for r in rows} == {"FII", "DII"}


def test_recorded_holidays() -> None:
    _need("nse_holidays.json")
    assert parse_holidays(json.loads((REC / "nse_holidays.json").read_text()))


def test_recorded_index_list() -> None:
    _need("nse_nifty50list.csv")
    assert len(parse_constituents(_rows("nse_nifty50list.csv"))) == 50


def test_recorded_bhavcopy() -> None:
    _need("nse_bhavcopy_full.csv")
    rows = parse_bhavcopy(_rows("nse_bhavcopy_full.csv"))
    assert len(rows) > 1000
    assert any(r.delivery_pct is not None for r in rows)

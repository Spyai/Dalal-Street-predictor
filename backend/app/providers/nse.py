"""NSE public data adapter (FII/DII, holiday calendar, index constituents, bhavcopy).

Goes through PoliteClient: honest user agent, robots.txt, <= 1 req/s, stops on blocks.
NSE's JSON API (www.nseindia.com/api/*) expects the session cookies its homepage sets,
so we load the homepage once per session before calling it. That is ordinary session
handling, not a bypass: if NSE refuses us, we stop and report "source unavailable".
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from typing import Any

from app.providers.base import (
    BhavRecord,
    ConstituentRecord,
    FlowRecord,
    HolidayRecord,
    SourceUnavailable,
)
from app.providers.http import PoliteClient

SOURCE = "nse"
WWW = "https://www.nseindia.com"
ARCHIVES = "https://nsearchives.nseindia.com"

INDEX_LIST_URLS = {
    "nifty50": f"{ARCHIVES}/content/indices/ind_nifty50list.csv",
    "nifty100": f"{ARCHIVES}/content/indices/ind_nifty100list.csv",
    "nifty200": f"{ARCHIVES}/content/indices/ind_nifty200list.csv",
    "nifty500": f"{ARCHIVES}/content/indices/ind_nifty500list.csv",
}
HINT = (
    "NSE public endpoints can block automated clients. Dalal will not work around that. "
    "Kite Connect covers prices; for flows/holidays, drop NSE's CSV into /imports."
)


class NSEProvider:
    name = SOURCE

    def __init__(self, client: PoliteClient) -> None:
        self._http = client
        self._warmed = False

    # ---------------------------------------------------------------- helpers

    def _api_json(self, path: str) -> Any:
        if not self._warmed:
            self._http.get(WWW + "/", expect_html=True)
            self._warmed = True
        resp = self._http.get(
            WWW + path, headers={"Accept": "application/json", "Referer": WWW + "/"}
        )
        try:
            return resp.json()
        except ValueError as exc:
            raise SourceUnavailable(SOURCE, f"non-JSON response from {path}", HINT) from exc

    def _csv(self, url: str) -> list[dict[str, str]]:
        text = self._http.get(url).text
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
        if not rows:
            raise SourceUnavailable(SOURCE, f"empty CSV from {url}", HINT)
        return rows

    # ------------------------------------------------------------- endpoints

    def fii_dii(self) -> list[FlowRecord]:
        """Latest provisional FII/FPI and DII cash-market activity (INR crore)."""
        payload = self._api_json("/api/fiidiiTradeReact")
        return parse_fii_dii(payload)

    def holidays(self) -> list[HolidayRecord]:
        payload = self._api_json("/api/holiday-master?type=trading")
        return parse_holidays(payload)

    def index_constituents(self, index: str) -> list[ConstituentRecord]:
        url = INDEX_LIST_URLS.get(index.lower())
        if url is None:
            raise SourceUnavailable(SOURCE, f"unknown index list {index}")
        return parse_constituents(self._csv(url))

    def bhavcopy(self, trade_date: date) -> list[BhavRecord]:
        """Full bhavcopy with delivery quantity/percentage for one trading day."""
        url = f"{ARCHIVES}/products/content/sec_bhavdata_full_{trade_date:%d%m%Y}.csv"
        return parse_bhavcopy(self._csv(url))


# --------------------------------------------------------------------- parsers
# Pure functions so they can be tested against recorded payloads.


def _num(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if s in {"", "-", "NA", "N/A", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _nse_date(s: str) -> date:
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised NSE date {s!r}")


def parse_fii_dii(payload: Any) -> list[FlowRecord]:
    if not isinstance(payload, list) or not payload:
        raise SourceUnavailable(SOURCE, "unexpected FII/DII payload shape", HINT)
    out: list[FlowRecord] = []
    for row in payload:
        cat_raw = str(row.get("category", "")).upper()
        if "FII" in cat_raw or "FPI" in cat_raw:
            category = "FII"
        elif "DII" in cat_raw:
            category = "DII"
        else:
            continue
        out.append(
            FlowRecord(
                trade_date=_nse_date(str(row["date"])),
                category=category,
                buy_value_cr=_num(row.get("buyValue")),
                sell_value_cr=_num(row.get("sellValue")),
                net_value_cr=_num(row.get("netValue")),
                provisional=True,
            )
        )
    if not out:
        raise SourceUnavailable(SOURCE, "FII/DII payload had no FII or DII rows", HINT)
    return out


def parse_holidays(payload: Any) -> list[HolidayRecord]:
    if not isinstance(payload, dict) or "CM" not in payload:
        raise SourceUnavailable(SOURCE, "holiday payload missing CM (equity) segment", HINT)
    out: list[HolidayRecord] = []
    for row in payload["CM"]:
        out.append(
            HolidayRecord(
                holiday_date=_nse_date(str(row["tradingDate"])),
                description=(row.get("description") or "").strip() or None,
                segment="CM",
            )
        )
    return out


def parse_constituents(rows: list[dict[str, str]]) -> list[ConstituentRecord]:
    return [
        ConstituentRecord(
            symbol=r["Symbol"].upper(),
            name=r.get("Company Name") or None,
            industry=r.get("Industry") or None,
            series=r.get("Series") or None,
            isin=r.get("ISIN Code") or None,
        )
        for r in rows
        if r.get("Symbol")
    ]


def parse_bhavcopy(rows: list[dict[str, str]]) -> list[BhavRecord]:
    out: list[BhavRecord] = []
    for r in rows:
        o, h, lo, c = (
            _num(r.get(k)) for k in ("OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE")
        )
        if None in (o, h, lo, c):
            continue
        out.append(
            BhavRecord(
                symbol=r["SYMBOL"].upper(),
                series=r.get("SERIES", ""),
                trade_date=_nse_date(r["DATE1"]),
                open=o,  # type: ignore[arg-type]
                high=h,  # type: ignore[arg-type]
                low=lo,  # type: ignore[arg-type]
                close=c,  # type: ignore[arg-type]
                prev_close=_num(r.get("PREV_CLOSE")),
                volume=_num(r.get("TTL_TRD_QNTY")),
                turnover_lacs=_num(r.get("TURNOVER_LACS")),
                delivery_qty=_num(r.get("DELIV_QTY")),
                delivery_pct=_num(r.get("DELIV_PER")),
            )
        )
    return out

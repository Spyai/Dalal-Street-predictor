"""Daily-bar ingestion priority, bhavcopy ingest, backfill resumability, global cues."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DailyBar, GlobalCueSnapshot, Instrument
from app.providers.base import (
    Bar,
    BhavRecord,
    Candle,
    Holding,
    InstrumentRecord,
    Quote,
    SourceUnavailable,
)
from app.services import global_cues, instruments
from app.services.backfill import run_backfill
from app.services.market_data import ingest_bhavcopy, upsert_daily_bars

D = date(2026, 10, 29)


def _inst(db: Session, symbol: str = "RELIANCE") -> Instrument:
    i = instruments.get_or_create_equity(db, symbol, "NSE")
    db.commit()
    return i


def _bar(d: date, close: float) -> Bar:
    return Bar(d, close, close + 1, close - 1, close, 1000)


def _get(db: Session, inst_id: int, d: date) -> DailyBar:
    db.expire_all()
    row = db.get(DailyBar, (inst_id, d))
    assert row is not None
    return row


def test_higher_priority_source_wins(db: Session) -> None:
    inst = _inst(db)
    ingest_bhavcopy(db, [BhavRecord("RELIANCE", "EQ", D, 1, 2, 0.5, 1.5, 1.4, 10, 1.0, 5, 50.0)])
    db.commit()
    upsert_daily_bars(db, inst.id, [_bar(D, 999)], "yfinance")
    db.commit()
    row = _get(db, inst.id, D)
    assert row.source == "nse_bhavcopy" and row.close == 1.5 and row.delivery_pct == 50.0

    upsert_daily_bars(db, inst.id, [_bar(D + timedelta(days=1), 10)], "yfinance")
    upsert_daily_bars(db, inst.id, [_bar(D + timedelta(days=1), 11)], "kite")
    db.commit()
    assert _get(db, inst.id, D + timedelta(days=1)).source == "kite"


def test_bhavcopy_ingest_filters_series_and_unknowns(db: Session) -> None:
    _inst(db, "RELIANCE")
    recs = [
        BhavRecord("RELIANCE", "EQ", D, 1, 2, 0.5, 1.5, None, 10, None, 5, 50.0),
        BhavRecord("RELIANCE", "N1", D, 1, 2, 0.5, 1.5, None, 10, None, None, None),
        BhavRecord("UNLISTEDX", "EQ", D, 1, 2, 0.5, 1.5, None, 10, None, None, None),
    ]
    stored, unknown = ingest_bhavcopy(db, recs)
    assert (stored, unknown) == (1, 1)


class FakeProvider:
    def __init__(self, name: str, have: set[str], fail: bool = False) -> None:
        self.name = name
        self.have = have
        self.fail = fail
        self.requests: list[tuple[str, date, date]] = []

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]:
        self.requests.append((symbol, start, end))
        if self.fail:
            raise SourceUnavailable(self.name, "down")
        if symbol not in self.have:
            return []
        out, d = [], start
        while d <= end:
            if d.weekday() < 5:
                out.append(_bar(d, 100))
            d += timedelta(days=1)
        return out

    def intraday_bars(self, *a: Any) -> list[Candle]:
        return []

    def quotes(self, instruments: list[tuple[str, str]]) -> dict[tuple[str, str], Quote]:
        return {}

    def instruments(self, exchange: str) -> list[InstrumentRecord]:
        return []

    def holdings(self) -> list[Holding]:
        return []


def test_backfill_falls_back_and_resumes(db: Session) -> None:
    a, b, c = _inst(db, "AAA"), _inst(db, "BBB"), _inst(db, "CCC")
    kite = FakeProvider("kite", have={"AAA"}, fail=False)
    yf = FakeProvider("yfinance", have={"BBB"})
    start, end = date(2026, 9, 1), date(2026, 10, 29)
    rep = run_backfill([a.id, b.id, c.id], start, end, [kite, yf])
    assert rep.filled_by_source == {"kite": 1, "yfinance": 1}
    assert set(rep.missing) == {"NSE:CCC"}

    # Second run: AAA and BBB are complete, so only CCC is requested again.
    kite2, yf2 = FakeProvider("kite", have=set()), FakeProvider("yfinance", have=set())
    rep2 = run_backfill([a.id, b.id, c.id], start, end, [kite2, yf2])
    assert rep2.already_complete == 2
    assert {r[0] for r in kite2.requests} == {"CCC"}

    # Extending the end date fetches only the new tail.
    kite3 = FakeProvider("kite", have={"AAA"})
    run_backfill([a.id], start, end + timedelta(days=1), [kite3])
    assert kite3.requests == [("AAA", end + timedelta(days=1), end + timedelta(days=1))]


def test_backfill_records_provider_failure(db: Session) -> None:
    a = _inst(db, "AAA")
    rep = run_backfill(
        [a.id], date(2026, 10, 1), date(2026, 10, 29), [FakeProvider("kite", set(), fail=True)]
    )
    assert "down" in rep.missing["NSE:AAA"]


class CueProvider(FakeProvider):
    def __init__(self, quotes: dict[tuple[str, str], Quote] | None, fail: bool = False) -> None:
        super().__init__("yfinance", set())
        self._q = quotes or {}
        self.fail = fail

    def quotes(self, instruments: list[tuple[str, str]]) -> dict[tuple[str, str], Quote]:
        if self.fail:
            raise SourceUnavailable("yfinance", "blocked")
        return {k: v for k, v in self._q.items() if k in instruments}


def test_global_cues_snapshot_and_staleness(db: Session) -> None:
    now = datetime(2026, 10, 30, 2, 0, tzinfo=UTC)  # 07:30 IST
    fresh = Quote("^GSPC", "GLOBAL", 5000, 4950, now - timedelta(hours=6))
    old = Quote("CL=F", "GLOBAL", 80, 79, now - timedelta(hours=10))
    global_cues.refresh(
        db, CueProvider({("^GSPC", "GLOBAL"): fresh, ("CL=F", "GLOBAL"): old}), now=now
    )
    db.commit()
    views = {v.key: v for v in global_cues.latest(db, now)}
    assert views["sp500"].status == "ok" and views["sp500"].change_pct == pytest.approx(
        1.0101, 1e-3
    )
    assert views["crude_wti"].status == "stale", "commodities older than 2h are stale"
    assert views["nikkei"].status == "unavailable"
    assert views["gift_nifty"].status == "unavailable" and "does not scrape" in (
        views["gift_nifty"].note or ""
    )
    assert views["us10y"].last is None

    # A later failed refresh must not erase the last good reading, and must mark it stale.
    later = now + timedelta(minutes=10)
    global_cues.refresh(db, CueProvider(None, fail=True), now=later)
    db.commit()
    v = {x.key: x for x in global_cues.latest(db, later)}["sp500"]
    assert v.last == 5000 and v.status == "stale" and "latest refresh failed" in (v.note or "")
    n = len(db.scalars(select(GlobalCueSnapshot).where(GlobalCueSnapshot.key == "sp500")).all())
    assert n == 2, "every attempt is kept as a snapshot"

"""Daily-bar ingestion priority, bhavcopy ingest, backfill resumability, global cues."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import IST
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
from app.providers.yfinance_provider import YFinanceProvider, capture_yf_errors, request_failures
from app.services import global_cues, instruments
from app.services.backfill import run_backfill
from app.services.market_data import (
    ensure_recent,
    fetch_daily,
    ingest_bhavcopy,
    last_final_global_day,
    upsert_daily_bars,
)

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


# ------------------------------------------------------------------ provisional bars (B)


def test_same_source_overwrites_its_own_earlier_bar(db: Session) -> None:
    """A bar stored while its market was open (a provisional print) is replaced by the final
    one from the same source; a lower-priority source still cannot touch it."""
    inst = _inst(db, "^GSPC")
    upsert_daily_bars(db, inst.id, [_bar(D, 5000)], "yfinance")
    db.commit()
    upsert_daily_bars(db, inst.id, [_bar(D, 5050)], "yfinance")
    db.commit()
    row = _get(db, inst.id, D)
    assert row.close == 5050 and row.source == "yfinance"
    upsert_daily_bars(db, inst.id, [_bar(D, 1)], "unknown_source")
    db.commit()
    assert _get(db, inst.id, D).close == 5050


class ClosesProvider(FakeProvider):
    """Returns the bars it is given (date -> close) and records every request."""

    def __init__(self, closes: dict[date, float]) -> None:
        super().__init__("yfinance", set())
        self.closes = closes

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]:
        self.requests.append((symbol, start, end))
        return [_bar(d, c) for d, c in sorted(self.closes.items()) if start <= d <= end]


def test_ensure_recent_refetch_window_replaces_a_provisional_bar(db: Session) -> None:
    inst = _inst(db, "ES=F")
    for i in range(5):  # history through D, with D's bar a mid-session print
        upsert_daily_bars(db, inst.id, [_bar(D - timedelta(days=i), 100 + i)], "yfinance")
    db.commit()
    final = ClosesProvider({D - timedelta(days=2): 102, D - timedelta(days=1): 101, D: 123.0})

    # Default: a bar for D exists, nothing is requested, the provisional print stays.
    assert ensure_recent(db, inst, D, [final], years=1) == (None, [])
    assert final.requests == [] and _get(db, inst.id, D).close == 100

    # With a re-fetch window the last three calendar days are requested again and the
    # same-source upsert replaces the provisional D bar with the final one.
    source, errors = ensure_recent(db, inst, D, [final], years=1, refetch_days=3)
    db.commit()
    assert (source, errors) == ("yfinance", [])
    assert final.requests == [("ES=F", D - timedelta(days=2), D)]
    assert _get(db, inst.id, D).close == 123.0

    # Behind by more than the window: the request starts where the history ends.
    later = D + timedelta(days=5)
    final.closes[later] = 130.0
    ensure_recent(db, inst, later, [final], years=1, refetch_days=3)
    assert final.requests[-1] == ("ES=F", D + timedelta(days=1), later)

    # No history at all: the usual multi-year request, the window changes nothing.
    fresh = _inst(db, "NQ=F")
    ensure_recent(db, fresh, D, [final], years=1, refetch_days=3)
    assert final.requests[-1] == ("NQ=F", D - timedelta(days=365 + 7), D)


def test_last_final_global_day_is_the_previous_utc_day() -> None:
    assert last_final_global_day(datetime(2026, 10, 30, 14, 0, tzinfo=UTC)) == date(2026, 10, 29)
    ist_1930 = datetime(2026, 10, 30, 19, 30, tzinfo=IST)  # 14:00 UTC: US session open
    assert last_final_global_day(ist_1930) == date(2026, 10, 30 - 1)
    ist_0730 = datetime(2026, 10, 30, 7, 30, tzinfo=IST)  # 02:00 UTC: US closed hours ago
    assert last_final_global_day(ist_0730) == date(2026, 10, 29)
    ist_0400 = datetime(2026, 10, 30, 4, 0, tzinfo=IST)  # still 29 Oct in UTC
    assert last_final_global_day(ist_0400) == date(2026, 10, 28)
    with pytest.raises(ValueError, match="timezone"):
        last_final_global_day(datetime(2026, 10, 30, 14, 0))


# ------------------------------------------------------------------ "no rows" vs failure (C)


def _logging_downloader(messages: list[str], frame: pd.DataFrame | None = None) -> Any:
    """A stand-in for yfinance.download that complains the way yfinance does (ERROR records
    on the `yfinance` logger) and returns `frame` (empty by default)."""

    def dl(**kw: Any) -> pd.DataFrame | None:
        log = logging.getLogger("yfinance")
        for m in messages:
            log.error(m)
        return pd.DataFrame() if frame is None else frame

    return dl


NO_ROWS = [
    "\n1 Failed download:",
    "['^N225']: possibly delisted; no price data found  (1d 2026-10-29 -> 2026-10-30)",
]
RATE_LIMITED = [
    "\n1 Failed download:",
    "['^N225']: YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')",
]


def test_yfinance_empty_range_is_no_rows_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr("app.services.market_data.mark_error", lambda s, e: errors.append(s))
    monkeypatch.setattr("app.services.market_data.mark_ok", lambda *a, **k: None)

    closed = YFinanceProvider(downloader=_logging_downloader(NO_ROWS))
    assert closed.daily_bars("^N225", "GLOBAL", D, D) == []
    inst = Instrument(id=1, symbol="^N225", exchange="GLOBAL")
    n, source, reasons = fetch_daily(None, inst, D, D, [closed])  # type: ignore[arg-type]
    assert (n, source) == (0, None)
    assert reasons == ["yfinance: no bars for GLOBAL:^N225 2026-10-29..2026-10-29"]
    assert errors == [], "a closed market is not a source error"

    limited = YFinanceProvider(downloader=_logging_downloader(RATE_LIMITED))
    with pytest.raises(SourceUnavailable, match="Rate limited"):
        limited.daily_bars("^N225", "GLOBAL", D, D)
    n, source, reasons = fetch_daily(None, inst, D, D, [limited])  # type: ignore[arg-type]
    assert (n, source) == (0, None) and "Rate limited" in reasons[0]
    assert errors == ["yfinance"], "a refused request is"

    # An empty frame with no explanation at all is still treated as unavailable, and a
    # 10-day quote window with nothing in it is never "no rows".
    silent = YFinanceProvider(downloader=_logging_downloader([]))
    with pytest.raises(SourceUnavailable, match="blocked, rate-limited or delisted"):
        silent.daily_bars("^N225", "GLOBAL", D, D)
    with pytest.raises(SourceUnavailable):
        closed.quotes([("^N225", "GLOBAL")])


def test_yfinance_error_classification() -> None:
    assert request_failures(NO_ROWS) == []
    assert (
        request_failures(
            ["2 Failed downloads:", "['A']: Data doesn't exist for startDate = 1, endDate = 2"]
        )
        == []
    )
    assert request_failures(RATE_LIMITED) == [RATE_LIMITED[1]]
    assert request_failures(["['A']: possibly delisted; no timezone found"]) == [
        "['A']: possibly delisted; no timezone found"
    ]
    assert request_failures(["['A']: ConnectionError(...)", "", "['B']: no price data found"]) == [
        "['A']: ConnectionError(...)"
    ]
    with capture_yf_errors() as got:
        logging.getLogger("yfinance").error("boom")
        logging.getLogger("yfinance").warning("not captured")
    assert got == ["boom"]

"""NSE trading calendar (equity cash market).

Holidays come from NSE's holiday master, or a CSV you drop into /imports. Dalal never
invents holiday dates. If a year has no holiday data, weekdays are assumed to be
trading days and every caller is told the calendar is unverified for that year.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import IST
from app.models import MarketHoliday
from app.providers.base import ExchangeDataProvider, HolidayRecord

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN = time(9, 0)


@dataclass(frozen=True)
class DayInfo:
    day: date
    is_trading_day: bool
    reason: str | None  # "weekend" | "NSE holiday: Diwali" | None
    calendar_verified: bool  # False if no holiday data for this year


@dataclass(frozen=True)
class MarketClock:
    now_ist: datetime
    session: str  # pre_open | open | closed | holiday | weekend
    today: DayInfo
    next_trading_day: date
    previous_trading_day: date


class TradingCalendar:
    """Loaded once from the DB; cheap to query repeatedly."""

    def __init__(self, holidays: dict[date, str | None], covered_years: set[int]) -> None:
        self._holidays = holidays
        self._years = covered_years

    @classmethod
    def load(cls, db: Session) -> TradingCalendar:
        rows = db.execute(
            select(MarketHoliday.holiday_date, MarketHoliday.description).where(
                MarketHoliday.segment == "CM"
            )
        ).all()
        holidays = {d: desc for d, desc in rows}
        return cls(holidays, {d.year for d in holidays})

    def info(self, d: date) -> DayInfo:
        verified = d.year in self._years
        if d.weekday() >= 5:
            return DayInfo(d, False, "weekend", verified)
        if d in self._holidays:
            desc = self._holidays[d] or "holiday"
            return DayInfo(d, False, f"NSE holiday: {desc}", verified)
        return DayInfo(d, True, None, verified)

    def is_trading_day(self, d: date) -> bool:
        return self.info(d).is_trading_day

    def next_trading_day(self, d: date) -> date:
        cur = d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur

    def previous_trading_day(self, d: date) -> date:
        cur = d - timedelta(days=1)
        while not self.is_trading_day(cur):
            cur -= timedelta(days=1)
        return cur

    def last_completed_session(self, now: datetime) -> date:
        """Most recent trading day whose close has passed (the day EOD data should exist for)."""
        local = now.astimezone(IST)
        today = local.date()
        if self.is_trading_day(today) and local.time() >= MARKET_CLOSE:
            return today
        return self.previous_trading_day(today)

    def clock(self, now: datetime | None = None) -> MarketClock:
        now_ist = (now or datetime.now(UTC)).astimezone(IST)
        today = self.info(now_ist.date())
        t = now_ist.time()
        if not today.is_trading_day:
            session = "weekend" if today.reason == "weekend" else "holiday"
        elif PRE_OPEN <= t < MARKET_OPEN:
            session = "pre_open"
        elif MARKET_OPEN <= t < MARKET_CLOSE:
            session = "open"
        else:
            session = "closed"
        return MarketClock(
            now_ist=now_ist,
            session=session,
            today=today,
            next_trading_day=self.next_trading_day(now_ist.date()),
            previous_trading_day=self.previous_trading_day(now_ist.date()),
        )


def store_holidays(db: Session, records: list[HolidayRecord], source: str) -> int:
    if not records:
        return 0
    stmt = insert(MarketHoliday).values(
        [
            {
                "holiday_date": r.holiday_date,
                "segment": r.segment,
                "description": r.description,
                "source": source,
            }
            for r in records
        ]
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_holiday_date_segment",
        set_={
            "description": stmt.excluded.description,
            "source": stmt.excluded.source,
            "fetched_at": func.now(),
        },
    )
    db.execute(stmt)
    return len(records)


def refresh_from_exchange(db: Session, provider: ExchangeDataProvider) -> int:
    return store_holidays(db, provider.holidays(), source=provider.name)


def import_holiday_csv(db: Session, path: Path) -> int:
    """CSV with columns: date (YYYY-MM-DD or DD-Mon-YYYY), description."""
    from app.providers.nse import _nse_date

    records: list[HolidayRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            norm = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            raw = norm.get("date") or norm.get("tradingdate") or norm.get("holiday_date")
            if not raw:
                continue
            records.append(
                HolidayRecord(holiday_date=_nse_date(raw), description=norm.get("description"))
            )
    return store_holidays(db, records, source=f"import:{path.name}")


def coverage(db: Session) -> dict[int, int]:
    rows = db.execute(
        select(func.extract("year", MarketHoliday.holiday_date), func.count())
        .where(MarketHoliday.segment == "CM")
        .group_by(func.extract("year", MarketHoliday.holiday_date))
    ).all()
    return {int(y): int(n) for y, n in rows}

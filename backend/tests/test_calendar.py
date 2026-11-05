"""Trading calendar logic (pure; no database)."""

from __future__ import annotations

from datetime import date, datetime

from app.config import IST
from app.services.calendar import TradingCalendar

# Synthetic holidays for the test only.
CAL = TradingCalendar(
    {date(2026, 11, 10): "Synthetic Holiday B", date(2026, 1, 26): "Republic Day"}, {2026}
)


def test_weekend_and_holiday_reasons() -> None:
    assert CAL.info(date(2026, 10, 31)).reason == "weekend"  # Saturday
    info = CAL.info(date(2026, 11, 10))
    assert not info.is_trading_day
    assert info.reason == "NSE holiday: Synthetic Holiday B"
    assert CAL.info(date(2026, 10, 30)).is_trading_day


def test_next_and_previous_skip_weekends_and_holidays() -> None:
    # Fri 6 Nov -> next is Mon 9 Nov; Mon 9 Nov -> next skips the Tue holiday -> Wed 11 Nov.
    assert CAL.next_trading_day(date(2026, 11, 6)) == date(2026, 11, 9)
    assert CAL.next_trading_day(date(2026, 11, 9)) == date(2026, 11, 11)
    assert CAL.previous_trading_day(date(2026, 11, 11)) == date(2026, 11, 9)
    assert CAL.previous_trading_day(date(2026, 11, 2)) == date(2026, 10, 30)


def test_unverified_year_is_flagged() -> None:
    info = CAL.info(date(2027, 3, 3))
    assert info.is_trading_day and not info.calendar_verified


def test_last_completed_session() -> None:
    before_close = datetime(2026, 10, 30, 15, 0, tzinfo=IST)
    after_close = datetime(2026, 10, 30, 15, 31, tzinfo=IST)
    saturday = datetime(2026, 10, 31, 10, 0, tzinfo=IST)
    assert CAL.last_completed_session(before_close) == date(2026, 10, 29)
    assert CAL.last_completed_session(after_close) == date(2026, 10, 30)
    assert CAL.last_completed_session(saturday) == date(2026, 10, 30)


def test_clock_sessions() -> None:
    assert CAL.clock(datetime(2026, 10, 30, 9, 5, tzinfo=IST)).session == "pre_open"
    assert CAL.clock(datetime(2026, 10, 30, 9, 15, tzinfo=IST)).session == "open"
    assert CAL.clock(datetime(2026, 10, 30, 15, 30, tzinfo=IST)).session == "closed"
    assert CAL.clock(datetime(2026, 11, 10, 11, 0, tzinfo=IST)).session == "holiday"
    assert CAL.clock(datetime(2026, 11, 1, 11, 0, tzinfo=IST)).session == "weekend"
    # UTC input is converted to IST: 04:00 UTC = 09:30 IST.
    from datetime import UTC

    assert CAL.clock(datetime(2026, 10, 30, 4, 0, tzinfo=UTC)).session == "open"

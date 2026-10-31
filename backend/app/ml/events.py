"""Scheduled-event flags for the market-call features (brief §4.3 "event flags").

Dates come from `<settings.imports_dir>/events.csv` (columns `date,kind`; kinds
`budget`, `rbi_policy`, `fomc`, `us_cpi`; dates ISO `YYYY-MM-DD` or `DD-Mon-YYYY`).
Without that file the only dates Dalal knows are the Union Budget days it is certain of,
and `events_loaded` is 0 so the model (and the reader of a feature snapshot) can tell
that the RBI/FOMC/CPI flags are unknown rather than genuinely off. No RBI, FOMC or CPI
date is ever written from memory.

Which session a flag lands on (point-in-time, calendar-aware):
- Indian-hours events (budget, RBI policy) hit the first session whose close can react:
  the session D itself, or D when the event fell on a non-session day since the previous
  session (a Saturday budget shows on the following Monday). Window `(prev, D]`.
- US after-hours events (FOMC decision at 14:00 ET, CPI at 08:30 ET) land after the
  Indian close, so they are "overnight" for the next Indian session. Window `[prev, D)`.
`prev` is the previous trading session when the caller knows it (the feature pipeline
passes it from the NSE calendar); otherwise the previous weekday.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from app.config import Settings, get_settings
from app.logging import get_logger

log = get_logger(__name__)

EVENTS_FILE = "events.csv"
EVENT_KINDS: tuple[str, ...] = ("budget", "rbi_policy", "fomc", "us_cpi")
EVENT_KEYS: tuple[str, ...] = (
    "budget_day",
    "rbi_policy_day",
    "fomc_overnight",
    "us_cpi_overnight",
    "events_loaded",
)
# Union Budget presentation days (Indian market hours). 2025 and 2026 fell on a
# weekend; the NSE held a special session, but the feature pipeline treats weekends as
# non-sessions, so these flag the following Monday instead.
KNOWN_BUDGET_DAYS: frozenset[date] = frozenset(
    {
        date(2021, 2, 1),
        date(2022, 2, 1),
        date(2023, 2, 1),
        date(2024, 2, 1),  # interim (election year)
        date(2024, 7, 23),  # full budget after the election
        date(2025, 2, 1),
        date(2026, 2, 1),
    }
)


@dataclass(frozen=True)
class EventCalendar:
    """Event dates by kind, plus whether they came from events.csv."""

    dates: dict[str, frozenset[date]] = field(default_factory=dict)
    loaded: bool = False
    path: Path | None = None

    def of(self, kind: str) -> frozenset[date]:
        return self.dates.get(kind, frozenset())

    def flags(self, d: date, previous_session: date | None = None) -> dict[str, float]:
        """Feature values (0.0/1.0) for session `d`; see the module docstring for timing."""
        prev = previous_session if previous_session is not None else previous_weekday(d)
        if prev >= d:
            raise ValueError(f"previous_session {prev} must be before the session {d}")
        return {
            "budget_day": _any_in(self.of("budget"), prev, d, lo_open=True, hi_open=False),
            "rbi_policy_day": _any_in(self.of("rbi_policy"), prev, d, lo_open=True, hi_open=False),
            "fomc_overnight": _any_in(self.of("fomc"), prev, d, lo_open=False, hi_open=True),
            "us_cpi_overnight": _any_in(self.of("us_cpi"), prev, d, lo_open=False, hi_open=True),
            "events_loaded": 1.0 if self.loaded else 0.0,
        }


def _any_in(dates: frozenset[date], lo: date, hi: date, *, lo_open: bool, hi_open: bool) -> float:
    for ev in dates:
        after_lo = ev > lo if lo_open else ev >= lo
        before_hi = ev < hi if hi_open else ev <= hi
        if after_lo and before_hi:
            return 1.0
    return 0.0


def previous_weekday(d: date) -> date:
    cur = d - timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def parse_event_date(raw: str) -> date:
    raw = raw.strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return datetime.strptime(raw, "%d-%b-%Y").date()


def load_events(settings: Settings | None = None) -> EventCalendar:
    """Read events.csv if present (unioned with the known budget days); never guess."""
    s = settings or get_settings()
    path = Path(s.imports_dir) / EVENTS_FILE
    dates: dict[str, set[date]] = {k: set() for k in EVENT_KINDS}
    dates["budget"] |= KNOWN_BUDGET_DAYS
    if not path.exists():
        return EventCalendar({k: frozenset(v) for k, v in dates.items()}, False, None)
    skipped = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            norm = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            raw_date, kind = norm.get("date", ""), norm.get("kind", "").lower()
            if not raw_date or kind not in dates:
                skipped += 1
                continue
            try:
                dates[kind].add(parse_event_date(raw_date))
            except ValueError:
                skipped += 1
    if skipped:
        log.warning("events_csv_rows_skipped", path=str(path), skipped=skipped)
    return EventCalendar({k: frozenset(v) for k, v in dates.items()}, True, path)


def event_flags(
    d: date,
    settings: Settings | None = None,
    *,
    previous_session: date | None = None,
    events: EventCalendar | None = None,
) -> dict[str, float]:
    """Event feature flags for session `d` (keys: EVENT_KEYS).

    `previous_session` is the last trading day before `d` (default: previous weekday).
    Pass a preloaded `events` calendar when calling this for many sessions.
    """
    cal = events if events is not None else load_events(settings)
    return cal.flags(d, previous_session)

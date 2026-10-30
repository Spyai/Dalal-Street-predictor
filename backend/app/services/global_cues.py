"""Overnight/global cues: refresh, snapshot (kept forever), and read with staleness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import IST
from app.models import GlobalCueSnapshot
from app.providers.base import MarketDataProvider, ProviderError, Quote
from app.services.instruments import GLOBAL_CUES, INDICES, RefSeries
from app.services.source_status import mark_error, mark_ok

# A reading older than this is shown as "stale (as of X)".
STALE_AFTER = {
    "US": timedelta(hours=30),
    "US futures": timedelta(hours=2),
    "Asia": timedelta(hours=30),
    "Commodities": timedelta(hours=2),
    "FX": timedelta(hours=2),
    "Rates": timedelta(hours=30),
    "India": timedelta(hours=30),
}
# Weekends: markets are shut, so allow readings from Friday's close.
WEEKEND_GRACE = timedelta(hours=66)

GIFT_NIFTY = RefSeries("gift_nifty", "GIFT NIFTY", "GLOBAL", "GIFT Nifty", None, "India")
GIFT_NIFTY_NOTE = (
    "Not available from Kite Connect or yfinance. Dalal does not scrape it. "
    "Shown as unavailable rather than guessed."
)


@dataclass(frozen=True)
class CueView:
    key: str
    label: str
    group: str
    unit: str
    last: float | None
    prev_close: float | None
    change_pct: float | None
    change_bps: float | None
    as_of: datetime | None
    status: str  # ok | stale | unavailable
    source: str | None
    note: str | None


def _status(ref: RefSeries, as_of: datetime | None, now: datetime) -> str:
    if as_of is None:
        return "stale"
    limit = STALE_AFTER.get(ref.group, timedelta(hours=30))
    local = now.astimezone(IST)
    if local.weekday() >= 5 or (local.weekday() == 0 and local.hour < 12):
        limit = max(limit, WEEKEND_GRACE)
    return "ok" if now - as_of <= limit else "stale"


def refresh(
    db: Session,
    yf: MarketDataProvider,
    broker: MarketDataProvider | None = None,
    now: datetime | None = None,
) -> list[GlobalCueSnapshot]:
    """Fetch every cue and append a snapshot row per cue (including unavailable ones)."""
    now = now or datetime.now(UTC)
    snaps: list[GlobalCueSnapshot] = []
    quotes: dict[str, tuple[Quote, str]] = {}

    # Indian indices: broker first.
    if broker is not None:
        try:
            got = broker.quotes([(r.symbol, r.exchange) for r in INDICES])
            for r in INDICES:
                if (r.symbol, r.exchange) in got:
                    quotes[r.key] = (got[(r.symbol, r.exchange)], broker.name)
            mark_ok(broker.name)
        except ProviderError as exc:
            mark_error(broker.name, exc)

    yf_refs = [r for r in INDICES if r.key not in quotes] + GLOBAL_CUES
    yf_error: str | None = None
    try:
        got = yf.quotes([(r.symbol, r.exchange) for r in yf_refs])
        for r in yf_refs:
            if (r.symbol, r.exchange) in got:
                quotes[r.key] = (got[(r.symbol, r.exchange)], yf.name)
        latest = max((q.as_of for q, _ in quotes.values() if q.as_of), default=None)
        mark_ok(yf.name, latest)
    except ProviderError as exc:
        mark_error(yf.name, exc)
        yf_error = f"yfinance: {exc.reason}"

    for ref in INDICES + GLOBAL_CUES:
        if ref.key in quotes:
            q, src = quotes[ref.key]
            snaps.append(
                GlobalCueSnapshot(
                    key=ref.key,
                    ticker=ref.yf_ticker or ref.symbol,
                    last=q.last,
                    prev_close=q.prev_close,
                    change_pct=q.change_pct,
                    as_of=q.as_of,
                    source=src,
                    status=_status(ref, q.as_of, now),
                    note=None,
                )
            )
        else:
            snaps.append(
                GlobalCueSnapshot(
                    key=ref.key,
                    ticker=ref.yf_ticker or ref.symbol,
                    source="none",
                    status="unavailable",
                    note=yf_error or "no data returned",
                )
            )
    snaps.append(
        GlobalCueSnapshot(
            key=GIFT_NIFTY.key,
            ticker="-",
            source="none",
            status="unavailable",
            note=GIFT_NIFTY_NOTE,
        )
    )
    db.add_all(snaps)
    db.flush()
    return snaps


def latest(db: Session, now: datetime | None = None) -> list[CueView]:
    """Most recent snapshot per cue; staleness recomputed against `now`.

    If the latest attempt failed but an earlier one succeeded, the earlier reading is
    shown as stale with its as-of time, never as current.
    """
    now = now or datetime.now(UTC)
    views: list[CueView] = []
    for ref in INDICES + GLOBAL_CUES + [GIFT_NIFTY]:
        rows = db.scalars(
            select(GlobalCueSnapshot)
            .where(GlobalCueSnapshot.key == ref.key)
            .order_by(GlobalCueSnapshot.fetched_at.desc(), GlobalCueSnapshot.id.desc())
            .limit(1)
        ).all()
        last_any = rows[0] if rows else None
        good = last_any
        if last_any is not None and last_any.last is None:
            good = db.scalar(
                select(GlobalCueSnapshot)
                .where(GlobalCueSnapshot.key == ref.key, GlobalCueSnapshot.last.is_not(None))
                .order_by(GlobalCueSnapshot.fetched_at.desc())
                .limit(1)
            )
        if good is None or good.last is None:
            note = last_any.note if last_any else "never fetched"
            views.append(
                CueView(
                    ref.key,
                    ref.label,
                    ref.group,
                    ref.unit,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "unavailable",
                    None,
                    note,
                )
            )
            continue
        status = _status(ref, good.as_of, now)
        note = None
        if good is not last_any and last_any is not None:
            status = "stale"
            note = f"latest refresh failed: {last_any.note}"
        bps = None
        if ref.unit == "yield_pct" and good.prev_close is not None:
            bps = round((good.last - good.prev_close) * 100, 1)
        views.append(
            CueView(
                key=ref.key,
                label=ref.label,
                group=ref.group,
                unit=ref.unit,
                last=good.last,
                prev_close=good.prev_close,
                change_pct=good.change_pct,
                change_bps=bps,
                as_of=good.as_of,
                status=status,
                source=good.source,
                note=note,
            )
        )
    return views

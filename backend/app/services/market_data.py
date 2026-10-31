"""Daily bars: ingest with source priority, fetch with provider fallback, read for charts.

Upsert semantics: a higher-priority source replaces a lower one, and the *same* source
replaces its own earlier row (`>=`), so a bar first stored while its market was still
open (a provisional print) is overwritten by the final one on the next fetch. Global
cues trade in other time zones: `last_final_global_day` says through which UTC day their
bars are final, and `ensure_recent(refetch_days=...)` re-requests the last few days so a
provisional bar never becomes the permanent close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import case, func, literal, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.logging import get_logger
from app.models import DailyBar, Instrument
from app.providers.base import Bar, BhavRecord, MarketDataProvider, NotSupported, ProviderError
from app.services.source_status import mark_error, mark_ok

log = get_logger(__name__)

# Higher wins when two sources supply the same day. Exchange bhavcopy is the record.
SOURCE_PRIORITY = {"nse_bhavcopy": 30, "kite": 20, "yfinance": 10}
BHAV_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}


def _priority(col: Any) -> Any:
    return case(
        *[(col == src, literal(p)) for src, p in SOURCE_PRIORITY.items()],
        else_=literal(0),
    )


def upsert_daily_bars(db: Session, instrument_id: int, bars: list[Bar], source: str) -> int:
    if not bars:
        return 0
    rows = [
        {
            "instrument_id": instrument_id,
            "trade_date": b.trade_date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "source": source,
        }
        for b in bars
    ]
    return _upsert_rows(db, rows)


def _upsert_rows(db: Session, rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for i in range(0, len(rows), 2000):
        chunk = rows[i : i + 2000]
        stmt = insert(DailyBar).values(chunk)
        ex = stmt.excluded
        stmt = stmt.on_conflict_do_update(
            index_elements=[DailyBar.instrument_id, DailyBar.trade_date],
            set_={
                "open": ex.open,
                "high": ex.high,
                "low": ex.low,
                "close": ex.close,
                "volume": ex.volume,
                "delivery_qty": func.coalesce(ex.delivery_qty, DailyBar.delivery_qty),
                "delivery_pct": func.coalesce(ex.delivery_pct, DailyBar.delivery_pct),
                "source": ex.source,
                "ingested_at": func.now(),
            },
            where=_priority(ex.source) >= _priority(DailyBar.source),
        )
        db.execute(stmt)
        total += len(chunk)
    return total


def ingest_bhavcopy(db: Session, records: list[BhavRecord]) -> tuple[int, int]:
    """Upsert NSE bhavcopy rows for known NSE instruments. Returns (stored, unknown_symbols)."""
    ids: dict[str, int] = {
        sym: inst_id
        for sym, inst_id in db.execute(
            select(Instrument.symbol, Instrument.id).where(Instrument.exchange == "NSE")
        ).all()
    }
    rows: list[dict[str, object]] = []
    unknown = 0
    for r in records:
        if r.series not in BHAV_SERIES:
            continue
        inst_id = ids.get(r.symbol)
        if inst_id is None:
            unknown += 1
            continue
        rows.append(
            {
                "instrument_id": inst_id,
                "trade_date": r.trade_date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "delivery_qty": r.delivery_qty,
                "delivery_pct": r.delivery_pct,
                "source": "nse_bhavcopy",
            }
        )
    return _upsert_rows(db, rows), unknown


def latest_bar_date(db: Session, instrument_id: int) -> date | None:
    return db.scalar(
        select(func.max(DailyBar.trade_date)).where(DailyBar.instrument_id == instrument_id)
    )


def last_final_global_day(now: datetime) -> date:
    """The last UTC calendar day whose GLOBAL-exchange daily bars are final.

    US cash and futures sessions, crude and FX all close before the UTC day ends, so a
    bar dated the current UTC day may still be a partial print; yesterday's is complete.
    The run day itself is never fetched for GLOBAL instruments.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(UTC).date() - timedelta(days=1)


def fetch_daily(
    db: Session,
    inst: Instrument,
    start: date,
    end: date,
    chain: list[MarketDataProvider],
) -> tuple[int, str | None, list[str]]:
    """Try providers in order; store the first non-empty result.

    Returns (rows_stored, source_used, errors). Errors are per-provider reasons so the
    caller can show exactly why data is missing. A provider that answered but had no
    rows for the range (a market holiday) is noted as "no bars" and is *not* marked as
    errored; only a request that failed (`ProviderError`) is.
    """
    errors: list[str] = []
    for provider in chain:
        try:
            bars = provider.daily_bars(inst.symbol, inst.exchange, start, end)
        except NotSupported as exc:
            errors.append(str(exc))
            continue
        except ProviderError as exc:
            mark_error(provider.name, exc)
            errors.append(str(exc))
            continue
        if bars:
            n = upsert_daily_bars(db, inst.id, bars, provider.name)
            mark_ok(provider.name, datetime.combine(bars[-1].trade_date, datetime.min.time(), UTC))
            return n, provider.name, errors
        errors.append(f"{provider.name}: no bars for {inst.exchange}:{inst.symbol} {start}..{end}")
    return 0, None, errors


def ensure_recent(
    db: Session,
    inst: Instrument,
    expected_latest: date,
    chain: list[MarketDataProvider],
    years: int,
    *,
    refetch_days: int = 0,
) -> tuple[str | None, list[str]]:
    """Make sure an instrument has history up to `expected_latest`. Used on page views.

    With `refetch_days` = 0 (the default) nothing is requested once a bar for
    `expected_latest` exists. With `refetch_days` = n the last n calendar days ending at
    `expected_latest` are requested again even when they are present, so a bar stored
    while its market was still open is replaced by the final one (same-source upsert).
    Returns (source that supplied bars or None, per-provider reasons for what is missing).
    """
    latest = latest_bar_date(db, inst.id)
    if latest is not None and latest >= expected_latest and refetch_days <= 0:
        return None, []
    if latest is None:
        start = expected_latest - timedelta(days=365 * years + 7)
    else:
        start = latest + timedelta(days=1)
        if refetch_days > 0:
            start = min(start, expected_latest - timedelta(days=refetch_days - 1))
    _, source, errors = fetch_daily(db, inst, start, expected_latest, chain)
    return source, errors


def daily_frame(db: Session, instrument_id: int, start: date | None = None) -> pd.DataFrame:
    q = select(
        DailyBar.trade_date,
        DailyBar.open,
        DailyBar.high,
        DailyBar.low,
        DailyBar.close,
        DailyBar.volume,
        DailyBar.delivery_pct,
        DailyBar.source,
    ).where(DailyBar.instrument_id == instrument_id)
    if start is not None:
        q = q.where(DailyBar.trade_date >= start)
    rows = db.execute(q.order_by(DailyBar.trade_date)).all()
    df = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume", "delivery_pct", "source"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df

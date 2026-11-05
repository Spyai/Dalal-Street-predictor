"""Historical daily-bar backfill with provider fallback, batching and resumability.

Safe to re-run: each instrument only fetches the ranges it is missing. Progress is
committed per batch, so an interrupted run keeps what it loaded.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select

from app.db import session_scope
from app.logging import get_logger
from app.models import DailyBar, Instrument
from app.providers.base import MarketDataProvider, NotSupported, ProviderError
from app.services.market_data import upsert_daily_bars
from app.services.source_status import mark_error, mark_ok

log = get_logger(__name__)
BATCH = 50
EDGE_SLACK = timedelta(days=7)  # listing gaps/holidays at the start of a range


@dataclass
class BackfillReport:
    requested: int = 0
    already_complete: int = 0
    rows_by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    filled_by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    missing: dict[str, str] = field(default_factory=dict)  # "NSE:XYZ" -> last error

    def summary(self) -> str:
        parts = [f"{self.requested} instruments", f"{self.already_complete} already complete"]
        parts += [
            f"{n} via {s} ({self.rows_by_source[s]} rows)" for s, n in self.filled_by_source.items()
        ]
        if self.missing:
            parts.append(f"{len(self.missing)} with no data")
        return ", ".join(parts)


@dataclass(frozen=True)
class _Need:
    inst_id: int
    symbol: str
    exchange: str
    start: date
    end: date


def _needs(instrument_ids: list[int], start: date, end: date) -> tuple[list[_Need], int]:
    with session_scope() as db:
        rows = db.execute(
            select(
                Instrument.id,
                Instrument.symbol,
                Instrument.exchange,
                func.min(DailyBar.trade_date),
                func.max(DailyBar.trade_date),
            )
            .outerjoin(DailyBar, DailyBar.instrument_id == Instrument.id)
            .where(Instrument.id.in_(instrument_ids))
            .group_by(Instrument.id)
        ).all()
    needs: list[_Need] = []
    complete = 0
    for inst_id, sym, ex, lo, hi in rows:
        if lo is None:
            needs.append(_Need(inst_id, sym, ex, start, end))
            continue
        added = False
        if lo > start + EDGE_SLACK:
            needs.append(_Need(inst_id, sym, ex, start, lo - timedelta(days=1)))
            added = True
        if hi < end:
            needs.append(_Need(inst_id, sym, ex, hi + timedelta(days=1), end))
            added = True
        if not added:
            complete += 1
    return needs, complete


def run_backfill(
    instrument_ids: list[int],
    start: date,
    end: date,
    chain: list[MarketDataProvider],
    progress: Callable[[str], None] | None = None,
) -> BackfillReport:
    report = BackfillReport(requested=len(instrument_ids))
    needs, report.already_complete = _needs(instrument_ids, start, end)
    say = progress or (lambda msg: log.info("backfill", msg=msg))
    say(f"{len(needs)} ranges to fetch across {len(instrument_ids)} instruments")

    remaining = needs
    for provider in chain:
        if not remaining:
            break
        still: list[_Need] = []
        batched = hasattr(provider, "daily_bars_many")
        groups: dict[tuple[date, date], list[_Need]] = defaultdict(list)
        for n in remaining:
            groups[(n.start, n.end)].append(n)
        for (g_start, g_end), items in groups.items():
            step = BATCH if batched else 1
            for i in range(0, len(items), step):
                chunk = items[i : i + step]
                try:
                    if batched:
                        got = provider.daily_bars_many(  # type: ignore[attr-defined]
                            [(n.symbol, n.exchange) for n in chunk], g_start, g_end
                        )
                    else:
                        n0 = chunk[0]
                        got = {
                            (n0.symbol, n0.exchange): provider.daily_bars(
                                n0.symbol, n0.exchange, g_start, g_end
                            )
                        }
                except NotSupported as exc:
                    for n in chunk:
                        report.missing[f"{n.exchange}:{n.symbol}"] = str(exc)
                    still.extend(chunk)
                    continue
                except ProviderError as exc:
                    mark_error(provider.name, exc)
                    for n in chunk:
                        report.missing[f"{n.exchange}:{n.symbol}"] = str(exc)
                    still.extend(chunk)
                    continue
                with session_scope() as db:
                    for n in chunk:
                        bars = got.get((n.symbol, n.exchange)) or []
                        if not bars:
                            still.append(n)
                            report.missing.setdefault(
                                f"{n.exchange}:{n.symbol}", f"{provider.name}: no bars"
                            )
                            continue
                        rows = upsert_daily_bars(db, n.inst_id, bars, provider.name)
                        report.rows_by_source[provider.name] += rows
                        report.filled_by_source[provider.name] += 1
                        report.missing.pop(f"{n.exchange}:{n.symbol}", None)
                done = min(i + step, len(items))
                say(f"{provider.name}: {done}/{len(items)} for {g_start}..{g_end}")
        if report.filled_by_source.get(provider.name):
            mark_ok(provider.name, datetime.now(UTC))
        remaining = still
    return report

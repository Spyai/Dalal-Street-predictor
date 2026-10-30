"""FII/DII cash-market flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import FiiDiiFlow
from app.providers.base import ExchangeDataProvider, FlowRecord
from app.services.calendar import TradingCalendar

# NSE publishes provisional FII/DII numbers in the evening, typically after 17:30 IST.
PUBLISH_HOUR_IST = 19


@dataclass(frozen=True)
class FlowDay:
    trade_date: date
    fii_net_cr: float | None
    dii_net_cr: float | None
    fii_buy_cr: float | None
    fii_sell_cr: float | None
    dii_buy_cr: float | None
    dii_sell_cr: float | None
    provisional: bool
    source: str


def store(db: Session, records: list[FlowRecord], source: str) -> int:
    if not records:
        return 0
    stmt = insert(FiiDiiFlow).values(
        [
            {
                "trade_date": r.trade_date,
                "category": r.category,
                "segment": "cash",
                "buy_value_cr": r.buy_value_cr,
                "sell_value_cr": r.sell_value_cr,
                "net_value_cr": r.net_value_cr,
                "provisional": r.provisional,
                "source": source,
            }
            for r in records
        ]
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_fii_dii_day_cat_seg",
        set_={
            "buy_value_cr": stmt.excluded.buy_value_cr,
            "sell_value_cr": stmt.excluded.sell_value_cr,
            "net_value_cr": stmt.excluded.net_value_cr,
            "provisional": stmt.excluded.provisional,
            "source": stmt.excluded.source,
            "fetched_at": func.now(),
        },
    )
    db.execute(stmt)
    return len(records)


def refresh(db: Session, provider: ExchangeDataProvider) -> int:
    return store(db, provider.fii_dii(), source=provider.name)


def recent(db: Session, days: int = 10) -> list[FlowDay]:
    dates = db.scalars(
        select(FiiDiiFlow.trade_date)
        .where(FiiDiiFlow.segment == "cash")
        .group_by(FiiDiiFlow.trade_date)
        .order_by(FiiDiiFlow.trade_date.desc())
        .limit(days)
    ).all()
    if not dates:
        return []
    rows = db.scalars(
        select(FiiDiiFlow).where(FiiDiiFlow.trade_date.in_(dates), FiiDiiFlow.segment == "cash")
    ).all()
    by_day: dict[date, dict[str, FiiDiiFlow]] = {}
    for r in rows:
        by_day.setdefault(r.trade_date, {})[r.category] = r
    out: list[FlowDay] = []
    for d in sorted(by_day, reverse=True):
        fii, dii = by_day[d].get("FII"), by_day[d].get("DII")
        any_row = fii or dii
        assert any_row is not None
        out.append(
            FlowDay(
                trade_date=d,
                fii_net_cr=fii.net_value_cr if fii else None,
                dii_net_cr=dii.net_value_cr if dii else None,
                fii_buy_cr=fii.buy_value_cr if fii else None,
                fii_sell_cr=fii.sell_value_cr if fii else None,
                dii_buy_cr=dii.buy_value_cr if dii else None,
                dii_sell_cr=dii.sell_value_cr if dii else None,
                provisional=any(r.provisional for r in by_day[d].values()),
                source=any_row.source,
            )
        )
    return out


def expected_latest(cal: TradingCalendar, now: datetime | None = None) -> date:
    """The trading day whose FII/DII numbers should already be published."""
    from app.config import IST

    local = (now or datetime.now(UTC)).astimezone(IST)
    today = local.date()
    if cal.is_trading_day(today) and local.hour >= PUBLISH_HOUR_IST:
        return today
    return cal.previous_trading_day(today)

"""Instrument master: reference series, the trading universe, Kite instrument tokens."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Instrument
from app.providers.base import ConstituentRecord, ExchangeDataProvider, MarketDataProvider
from app.providers.nse import parse_constituents


@dataclass(frozen=True)
class RefSeries:
    key: str
    symbol: str
    exchange: str  # INDEX | GLOBAL
    label: str
    yf_ticker: str | None
    group: str
    unit: str = "level"  # level | price | fx | yield_pct


# Indian indices (prefer Kite; yfinance fallback) and overnight global cues (yfinance).
INDICES: list[RefSeries] = [
    RefSeries("nifty50", "NIFTY 50", "INDEX", "Nifty 50", "^NSEI", "India"),
    RefSeries("banknifty", "NIFTY BANK", "INDEX", "Bank Nifty", "^NSEBANK", "India"),
    RefSeries("india_vix", "INDIA VIX", "INDEX", "India VIX", "^INDIAVIX", "India"),
]
GLOBAL_CUES: list[RefSeries] = [
    RefSeries("sp500", "^GSPC", "GLOBAL", "S&P 500", "^GSPC", "US"),
    RefSeries("nasdaq", "^IXIC", "GLOBAL", "Nasdaq", "^IXIC", "US"),
    RefSeries("es_fut", "ES=F", "GLOBAL", "S&P 500 fut", "ES=F", "US futures"),
    RefSeries("nq_fut", "NQ=F", "GLOBAL", "Nasdaq fut", "NQ=F", "US futures"),
    RefSeries("nikkei", "^N225", "GLOBAL", "Nikkei 225", "^N225", "Asia"),
    RefSeries("hang_seng", "^HSI", "GLOBAL", "Hang Seng", "^HSI", "Asia"),
    RefSeries("crude_wti", "CL=F", "GLOBAL", "WTI crude", "CL=F", "Commodities", "price"),
    RefSeries("crude_brent", "BZ=F", "GLOBAL", "Brent crude", "BZ=F", "Commodities", "price"),
    RefSeries("usdinr", "USDINR=X", "GLOBAL", "USD/INR", "USDINR=X", "FX", "fx"),
    RefSeries("dxy", "DX-Y.NYB", "GLOBAL", "Dollar index", "DX-Y.NYB", "FX"),
    RefSeries("us10y", "^TNX", "GLOBAL", "US 10Y yield", "^TNX", "Rates", "yield_pct"),
]
ALL_REF = INDICES + GLOBAL_CUES


def ensure_reference_instruments(db: Session) -> None:
    for ref in ALL_REF:
        stmt = insert(Instrument).values(
            symbol=ref.symbol,
            exchange=ref.exchange,
            name=ref.label,
            yf_ticker=ref.yf_ticker,
            is_index=True,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_instrument_exchange_symbol",
            set_={"name": ref.label, "yf_ticker": ref.yf_ticker, "is_index": True},
        )
        db.execute(stmt)


def get_instrument(db: Session, symbol: str, exchange: str) -> Instrument | None:
    return db.scalar(
        select(Instrument).where(
            Instrument.symbol == symbol.upper(), Instrument.exchange == exchange.upper()
        )
    )


def get_or_create_equity(db: Session, symbol: str, exchange: str) -> Instrument:
    inst = get_instrument(db, symbol, exchange)
    if inst is None:
        inst = Instrument(symbol=symbol.upper(), exchange=exchange.upper(), is_index=False)
        db.add(inst)
        db.flush()
    return inst


def search(db: Session, q: str, limit: int = 15) -> list[Instrument]:
    q = q.strip().upper()
    if not q:
        return []
    return list(
        db.scalars(
            select(Instrument)
            .where(
                Instrument.exchange.in_(["NSE", "BSE", "INDEX"]),
                or_(Instrument.symbol.startswith(q), func.upper(Instrument.name).contains(q)),
            )
            .order_by(
                (Instrument.symbol == q).desc(),
                Instrument.in_universe.desc(),
                (Instrument.exchange == "NSE").desc(),
                Instrument.symbol,
            )
            .limit(limit)
        )
    )


def _import_constituents(universe: str) -> list[ConstituentRecord] | None:
    path = Path(get_settings().imports_dir) / f"ind_{universe.lower()}list.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            {(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)
        ]
    return parse_constituents(rows)


def sync_universe(
    db: Session, exchange_provider: ExchangeDataProvider | None, universe: str | None = None
) -> tuple[int, str]:
    """Mark universe members (+ watchlist). Returns (count, source)."""
    settings = get_settings()
    universe = (universe or settings.universe).lower()
    members: list[ConstituentRecord] = []
    source = "watchlist"
    if universe != "watchlist":
        imported = _import_constituents(universe)
        if imported:
            members, source = imported, f"import:ind_{universe}list.csv"
        elif exchange_provider is not None:
            members, source = exchange_provider.index_constituents(universe), exchange_provider.name
    symbols = {m.symbol for m in members} | set(settings.watchlist_symbols)
    by_symbol = {m.symbol: m for m in members}

    db.execute(update(Instrument).where(Instrument.exchange == "NSE").values(in_universe=False))
    for sym in symbols:
        m = by_symbol.get(sym)
        values = {"symbol": sym, "exchange": "NSE", "in_universe": True, "is_index": False}
        if m is not None:
            values |= {"name": m.name, "industry": m.industry, "series": m.series, "isin": m.isin}
        stmt = insert(Instrument).values(**values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_instrument_exchange_symbol",
            set_={k: v for k, v in values.items() if k not in {"symbol", "exchange"}},
        )
        db.execute(stmt)
    return len(symbols), source


def sync_broker_instruments(db: Session, broker: MarketDataProvider) -> int:
    """Store broker instrument tokens for NSE/BSE equities and indices."""
    count = 0
    for exchange in ("NSE", "BSE"):
        for rec in broker.instruments(exchange):
            values = {
                "symbol": rec.symbol.upper(),
                "exchange": rec.exchange,
                "kite_token": rec.kite_token,
                "lot_size": rec.lot_size,
                "is_index": rec.is_index,
            }
            stmt = insert(Instrument).values(name=rec.name, **values)
            set_: dict[str, object] = {"kite_token": rec.kite_token, "lot_size": rec.lot_size}
            if rec.name:
                set_["name"] = func.coalesce(Instrument.name, rec.name)
            stmt = stmt.on_conflict_do_update(constraint="uq_instrument_exchange_symbol", set_=set_)
            db.execute(stmt)
            count += 1
    return count


def universe_instruments(db: Session) -> list[Instrument]:
    return list(
        db.scalars(
            select(Instrument)
            .where(Instrument.in_universe.is_(True), Instrument.exchange == "NSE")
            .order_by(Instrument.symbol)
        )
    )

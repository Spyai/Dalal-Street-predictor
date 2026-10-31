"""Provider adapter interfaces.

Business logic depends only on these protocols and record types, so a broker or
data source can be swapped without touching services, jobs or the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------- errors


class ProviderError(Exception):
    """Base class. `hint` tells the user what would fix or replace the source."""

    def __init__(self, source: str, reason: str, hint: str | None = None) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason
        self.hint = hint


class NotConfigured(ProviderError):
    """Credentials or keys for this source are missing."""


class AuthExpired(ProviderError):
    """Credentials exist but the session/token is missing or expired (e.g. daily Kite login)."""


class SourceUnavailable(ProviderError):
    """The source refused, blocked, broke, or returned unusable data. Stop and report."""


class NotSupported(ProviderError):
    """This provider does not offer the requested data."""


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Bar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass(frozen=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass(frozen=True)
class Quote:
    symbol: str
    exchange: str
    last: float
    prev_close: float | None
    as_of: datetime | None
    volume: float | None = None

    @property
    def change_pct(self) -> float | None:
        if self.prev_close in (None, 0):
            return None
        return (self.last / self.prev_close - 1.0) * 100.0


@dataclass(frozen=True)
class InstrumentRecord:
    symbol: str
    exchange: str  # NSE | BSE | INDEX
    name: str | None
    kite_token: int | None = None
    isin: str | None = None
    series: str | None = None
    lot_size: int | None = None
    is_index: bool = False


@dataclass(frozen=True)
class ConstituentRecord:
    symbol: str
    name: str | None
    industry: str | None
    series: str | None
    isin: str | None


@dataclass(frozen=True)
class FlowRecord:
    trade_date: date
    category: str  # FII | DII
    buy_value_cr: float | None
    sell_value_cr: float | None
    net_value_cr: float | None
    provisional: bool = True


@dataclass(frozen=True)
class HolidayRecord:
    holiday_date: date
    description: str | None
    segment: str = "CM"


@dataclass(frozen=True)
class BhavRecord:
    symbol: str
    series: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    prev_close: float | None
    volume: float | None
    turnover_lacs: float | None
    delivery_qty: float | None
    delivery_pct: float | None


@dataclass(frozen=True)
class Holding:
    symbol: str
    exchange: str
    isin: str | None
    quantity: float
    average_price: float
    last_price: float | None


@dataclass(frozen=True)
class NewsItem:
    source: str
    url: str
    title: str
    published_at: datetime | None
    summary: str | None = None
    symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FundamentalsSnapshot:
    symbol: str
    as_of: date | None
    values: dict[str, float | None]
    source: str


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class MarketDataProvider(Protocol):
    """Prices: daily/intraday candles, quotes, instrument master, read-only holdings."""

    name: str

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]: ...

    def intraday_bars(
        self, symbol: str, exchange: str, start: datetime, end: datetime, interval: str
    ) -> list[Candle]: ...

    def quotes(self, instruments: list[tuple[str, str]]) -> dict[tuple[str, str], Quote]: ...

    def instruments(self, exchange: str) -> list[InstrumentRecord]: ...

    def holdings(self) -> list[Holding]: ...


@runtime_checkable
class ExchangeDataProvider(Protocol):
    """Exchange-published datasets: flows, holiday calendar, index lists, bhavcopy."""

    name: str

    def fii_dii(self) -> list[FlowRecord]: ...

    def holidays(self) -> list[HolidayRecord]: ...

    def index_constituents(self, index: str) -> list[ConstituentRecord]: ...

    def bhavcopy(self, trade_date: date) -> list[BhavRecord]: ...


@runtime_checkable
class NewsProvider(Protocol):
    """Headlines and announcements. Implemented in Phase 5 (RSS, NSE/BSE announcements)."""

    name: str

    def fetch(self, since: datetime) -> list[NewsItem]: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Ratios, results, shareholding. Implemented in Phase 5 (Screener CSV imports, yfinance)."""

    name: str

    def fundamentals(self, symbol: str) -> FundamentalsSnapshot | None: ...

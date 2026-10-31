"""Zerodha Kite Connect adapter. READ-ONLY: this module never calls any order endpoint.

Kite access tokens expire daily (~06:00 IST). Token storage and the login flow live in
app.services.kite_auth; this adapter only takes an authenticated client.

Rate limits respected (Kite docs): historical 3 req/s, quote 1 req/s.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from app.config import IST
from app.providers.base import (
    AuthExpired,
    Bar,
    Candle,
    Holding,
    InstrumentRecord,
    NotSupported,
    Quote,
    SourceUnavailable,
)

SOURCE = "kite"

# Max days per historical request, per interval (Kite limits).
MAX_DAYS = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}
INDEX_SEGMENT = "INDICES"
QUOTE_BATCH = 500


class KiteClient(Protocol):
    """Subset of kiteconnect.KiteConnect this adapter uses. No order methods, by design."""

    def historical_data(
        self, instrument_token: int, from_date: Any, to_date: Any, interval: str, **kw: Any
    ) -> list[dict[str, Any]]: ...

    def quote(self, *instruments: Any) -> dict[str, dict[str, Any]]: ...

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]: ...

    def holdings(self) -> list[dict[str, Any]]: ...


class _RateLimiter:
    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._interval - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def _translate(exc: Exception) -> Exception:
    """Map kiteconnect exceptions to provider errors."""
    name = exc.__class__.__name__
    msg = str(exc)
    if name == "TokenException":
        return AuthExpired(SOURCE, "Kite session expired or invalid", "Log in to Kite again")
    if name == "PermissionException":
        return SourceUnavailable(
            SOURCE,
            f"permission denied: {msg}",
            "Your Kite Connect plan may not include this data (historical/market data "
            "needs the paid Connect plan).",
        )
    if name in {"NetworkException", "DataException", "GeneralException", "InputException"}:
        return SourceUnavailable(SOURCE, f"{name}: {msg}")
    return SourceUnavailable(SOURCE, f"unexpected error: {name}: {msg}")


class KiteProvider:
    name = SOURCE

    def __init__(
        self,
        client: KiteClient,
        token_lookup: Callable[[str, str], int | None],
    ) -> None:
        """`token_lookup(symbol, exchange)` resolves Kite instrument tokens (from the DB)."""
        self._kite = client
        self._token_lookup = token_lookup
        self._hist_limiter = _RateLimiter(3)
        self._quote_limiter = _RateLimiter(1)

    def _call(self, limiter: _RateLimiter, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        limiter.wait()
        try:
            return fn(*a, **kw)
        except Exception as exc:
            raise _translate(exc) from exc

    def _token(self, symbol: str, exchange: str) -> int:
        tok = self._token_lookup(symbol, exchange)
        if tok is None:
            raise NotSupported(SOURCE, f"no Kite instrument token for {exchange}:{symbol}")
        return tok

    def _history(
        self, token: int, start: datetime, end: datetime, interval: str
    ) -> list[dict[str, Any]]:
        span = timedelta(days=MAX_DAYS[interval])
        rows: list[dict[str, Any]] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + span - timedelta(seconds=1), end)
            rows.extend(
                self._call(
                    self._hist_limiter,
                    self._kite.historical_data,
                    token,
                    cursor,
                    chunk_end,
                    interval,
                )
                or []
            )
            cursor = chunk_end + timedelta(seconds=1)
        return rows

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]:
        token = self._token(symbol, exchange)
        s = datetime(start.year, start.month, start.day, tzinfo=IST)
        e = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=IST)
        bars: dict[date, Bar] = {}
        for row in self._history(token, s, e, "day"):
            d = _as_ist(row["date"]).date()
            bars[d] = Bar(
                trade_date=d,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if row.get("volume") is not None else None,
            )
        return [bars[d] for d in sorted(bars)]

    def intraday_bars(
        self, symbol: str, exchange: str, start: datetime, end: datetime, interval: str
    ) -> list[Candle]:
        if interval not in MAX_DAYS or interval == "day":
            raise NotSupported(SOURCE, f"unsupported intraday interval {interval}")
        token = self._token(symbol, exchange)
        return [
            Candle(
                ts=_as_ist(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if row.get("volume") is not None else None,
            )
            for row in self._history(token, start, end, interval)
        ]

    def quotes(self, instruments: list[tuple[str, str]]) -> dict[tuple[str, str], Quote]:
        keys = {_kite_key(sym, ex): (sym, ex) for sym, ex in instruments}
        out: dict[tuple[str, str], Quote] = {}
        key_list = list(keys)
        for i in range(0, len(key_list), QUOTE_BATCH):
            batch = key_list[i : i + QUOTE_BATCH]
            data = self._call(self._quote_limiter, self._kite.quote, batch) or {}
            for k, q in data.items():
                if k not in keys or q.get("last_price") is None:
                    continue
                sym, ex = keys[k]
                ohlc = q.get("ohlc") or {}
                ts = q.get("last_trade_time") or q.get("timestamp")
                out[(sym, ex)] = Quote(
                    symbol=sym,
                    exchange=ex,
                    last=float(q["last_price"]),
                    prev_close=float(ohlc["close"]) if ohlc.get("close") else None,
                    as_of=_as_ist(ts) if ts else None,
                    volume=float(q["volume"]) if q.get("volume") is not None else None,
                )
        return out

    def instruments(self, exchange: str) -> list[InstrumentRecord]:
        rows = self._call(self._quote_limiter, self._kite.instruments, exchange) or []
        out: list[InstrumentRecord] = []
        for r in rows:
            seg = r.get("segment", "")
            is_index = seg == INDEX_SEGMENT or r.get("instrument_type") == "INDEX"
            if not is_index and r.get("instrument_type") != "EQ":
                continue
            out.append(
                InstrumentRecord(
                    symbol=str(r["tradingsymbol"]),
                    exchange="INDEX" if is_index else exchange,
                    name=r.get("name") or None,
                    kite_token=int(r["instrument_token"]),
                    lot_size=int(r["lot_size"]) if r.get("lot_size") else None,
                    is_index=is_index,
                )
            )
        return out

    def holdings(self) -> list[Holding]:
        rows = self._call(self._quote_limiter, self._kite.holdings) or []
        return [
            Holding(
                symbol=r["tradingsymbol"],
                exchange=r.get("exchange", "NSE"),
                isin=r.get("isin"),
                quantity=float(r.get("quantity", 0)) + float(r.get("t1_quantity", 0)),
                average_price=float(r.get("average_price", 0)),
                last_price=float(r["last_price"]) if r.get("last_price") is not None else None,
            )
            for r in rows
        ]


def _kite_key(symbol: str, exchange: str) -> str:
    # Indices are quoted on the NSE/BSE exchange prefix in Kite, e.g. "NSE:NIFTY 50".
    if exchange == "INDEX":
        return f"BSE:{symbol}" if symbol == "SENSEX" else f"NSE:{symbol}"
    return f"{exchange}:{symbol}"


def _as_ist(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=IST) if value.tzinfo is None else value.astimezone(IST)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=IST)
    parsed = datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=IST) if parsed.tzinfo is None else parsed.astimezone(IST)

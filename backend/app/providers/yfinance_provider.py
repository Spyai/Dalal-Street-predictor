"""yfinance adapter: EOD history fallback for NSE/BSE (.NS/.BO) and the global-cue source.

yfinance wraps Yahoo's unofficial endpoints. Treat it as best-effort: every failure
surfaces as SourceUnavailable and is shown as such, never papered over.

"No rows" is not a failure. `yfinance.download` never raises: it logs each ticker's
problem on the `yfinance` logger and returns an empty frame both when Yahoo refused or
rate-limited the request and when the range simply holds no session (a foreign market
holiday). `_fetch` captures those log records while the download runs: an empty frame
whose only complaints are "no price data found" for the range is returned as *no rows*
(`daily_bars` -> `[]`), while a rate limit, HTTP or connection error, or an empty frame
with no explanation at all, raises SourceUnavailable.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from app.providers.base import (
    Bar,
    Candle,
    Holding,
    InstrumentRecord,
    NotSupported,
    Quote,
    SourceUnavailable,
)

SOURCE = "yfinance"

# Index symbols as stored in Dalal -> Yahoo tickers.
INDEX_TICKERS: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "SENSEX": "^BSESN",
}

Downloader = Callable[..., pd.DataFrame | None]

YF_LOGGER = "yfinance"
# yfinance's wording when Yahoo answered but had no session in the requested range
# (YFPricesMissingError, or Yahoo's own explanation). Anything else it logs at ERROR
# level is a failed request.
NO_ROWS_MARKERS: tuple[str, ...] = ("no price data found", "data doesn't exist", "no data found")


@contextmanager
def capture_yf_errors() -> Iterator[list[str]]:
    """Collect what yfinance logs at ERROR level (its only error channel) during a call."""
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage().strip())

    logger = logging.getLogger(YF_LOGGER)
    handler = _Collect(level=logging.ERROR)
    previous_level = logger.level
    if logger.getEffectiveLevel() > logging.ERROR:
        logger.setLevel(logging.ERROR)  # a silenced logger would hide the failure from us
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def is_no_rows_message(message: str) -> bool:
    """True for yfinance's "nothing in this range" complaint, False for a real failure."""
    text = message.lower()
    return any(marker in text for marker in NO_ROWS_MARKERS)


_FAILED_HEADER = re.compile(r"^\d+ failed downloads?:$", re.IGNORECASE)


def request_failures(messages: list[str]) -> list[str]:
    """The captured messages that describe a failed request (not merely an empty range).

    yfinance prefixes the per-ticker lines with a "N Failed download(s):" header; that
    header carries no reason of its own and is dropped.
    """
    stripped = [m.strip() for m in messages]
    return [m for m in stripped if m and not _FAILED_HEADER.match(m) and not is_no_rows_message(m)]


def yf_ticker(symbol: str, exchange: str) -> str:
    if exchange == "NSE":
        return f"{symbol}.NS"
    if exchange == "BSE":
        return f"{symbol}.BO"
    if exchange == "INDEX":
        if symbol not in INDEX_TICKERS:
            raise NotSupported(SOURCE, f"no Yahoo ticker mapped for index {symbol}")
        return INDEX_TICKERS[symbol]
    return symbol  # GLOBAL: already a Yahoo ticker


def _default_downloader(**kwargs: Any) -> pd.DataFrame | None:
    import yfinance as yf

    if os.environ.get("VERCEL"):
        # Serverless filesystems are read-only except /tmp; yfinance caches timezones.
        yf.set_tz_cache_location("/tmp/yfinance-cache")
    return yf.download(progress=False, **kwargs)


def split_by_ticker(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Normalise yfinance output (single or multi-ticker, either column layout)."""
    if not isinstance(df.columns, pd.MultiIndex):
        return {tickers[0]: df} if len(tickers) == 1 else {}
    out: dict[str, pd.DataFrame] = {}
    lvl0 = set(df.columns.get_level_values(0))
    lvl1 = set(df.columns.get_level_values(1))
    for t in tickers:
        if t in lvl0:
            out[t] = pd.DataFrame(df[t])
        elif t in lvl1:
            out[t] = pd.DataFrame(df.xs(t, axis=1, level=1))
    return out


def _frame_to_bars(df: pd.DataFrame) -> list[Bar]:
    bars: list[Bar] = []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)  # type: ignore[arg-type]
        vol = row.get("Volume")
        bars.append(
            Bar(
                trade_date=ts.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=None if vol is None or pd.isna(vol) else float(vol),
            )
        )
    return bars


class YFinanceProvider:
    name = SOURCE

    def __init__(self, downloader: Downloader | None = None) -> None:
        self._download = downloader or _default_downloader

    def _fetch(self, *, allow_no_rows: bool = False, **kwargs: Any) -> pd.DataFrame:
        """One download. Raises SourceUnavailable for a failed request; with
        `allow_no_rows` an empty answer that yfinance explains as "no price data" for the
        range comes back as an empty frame instead (the market was closed)."""
        with capture_yf_errors() as reported:
            try:
                df = self._download(**kwargs)
            except Exception as exc:  # yfinance raises a wide variety of things
                raise SourceUnavailable(SOURCE, f"download failed: {exc}") from exc
        failures = request_failures(reported)
        if df is not None and not df.empty:
            return df
        tickers = kwargs.get("tickers")
        what = (
            f"{len(tickers)} tickers" if isinstance(tickers, list) and len(tickers) > 2 else tickers
        )
        if failures:
            raise SourceUnavailable(SOURCE, f"download failed for {what}: {failures[0]}")
        if allow_no_rows and reported:
            return pd.DataFrame()  # Yahoo answered: no session in this range
        raise SourceUnavailable(
            SOURCE, f"Yahoo returned no data for {what} (blocked, rate-limited or delisted)"
        )

    def _fetch_many(
        self, tickers: list[str], *, allow_no_rows: bool = False, **kwargs: Any
    ) -> dict[str, pd.DataFrame]:
        df = self._fetch(
            tickers=tickers,
            group_by="ticker",
            multi_level_index=True,
            allow_no_rows=allow_no_rows,
            **kwargs,
        )
        if df.empty:
            return {}
        frames = split_by_ticker(df, tickers)
        return {t: f.dropna(how="all") for t, f in frames.items() if not f.dropna(how="all").empty}

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]:
        return self.daily_bars_many([(symbol, exchange)], start, end).get((symbol, exchange), [])

    def daily_bars_many(
        self, instruments: list[tuple[str, str]], start: date, end: date
    ) -> dict[tuple[str, str], list[Bar]]:
        """Batched daily history. Close is split-adjusted by Yahoo, not dividend-adjusted.

        A ticker with no session in the range (a holiday on its exchange) is simply absent
        from the result; only a failed request raises.
        """
        tickers = {yf_ticker(s, e): (s, e) for s, e in instruments}
        frames = self._fetch_many(
            list(tickers),
            allow_no_rows=True,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        return {
            tickers[t]: [b for b in _frame_to_bars(f) if start <= b.trade_date <= end]
            for t, f in frames.items()
        }

    def intraday_bars(
        self, symbol: str, exchange: str, start: datetime, end: datetime, interval: str
    ) -> list[Candle]:
        raise NotSupported(SOURCE, "intraday candles come from the broker API only")

    def quotes(self, instruments: list[tuple[str, str]]) -> dict[tuple[str, str], Quote]:
        """Latest price + previous session close per ticker, in two batched calls.

        A 10-day daily download gives session closes; a 1-day 5-minute download (with
        pre/post market) gives the latest print and its timestamp when available.
        """
        tickers = {yf_ticker(s, e): (s, e) for s, e in instruments}
        daily = self._fetch_many(list(tickers), period="10d", interval="1d", auto_adjust=False)
        try:
            intra = self._fetch_many(
                list(tickers), period="1d", interval="5m", prepost=True, auto_adjust=False
            )
        except SourceUnavailable:
            intra = {}  # daily close is still a valid (older) reading; as_of says how old
        out: dict[tuple[str, str], Quote] = {}
        for t, key in tickers.items():
            frame = daily.get(t)
            if frame is None or "Close" not in frame:
                continue
            closes = frame["Close"].dropna().astype(float)
            if closes.empty:
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
            as_of = _to_utc(pd.Timestamp(closes.index[-1]))
            ifr = intra.get(t)
            if ifr is not None and "Close" in ifr:
                ic = ifr["Close"].dropna()
                if not ic.empty:
                    its = _to_utc(pd.Timestamp(ic.index[-1]))
                    if its is not None and (as_of is None or its >= as_of):
                        last, as_of = float(ic.iloc[-1]), its
            out[key] = Quote(
                symbol=key[0], exchange=key[1], last=last, prev_close=prev, as_of=as_of
            )
        return out

    def instruments(self, exchange: str) -> list[InstrumentRecord]:
        raise NotSupported(SOURCE, "no instrument master; use the broker or NSE equity list")

    def holdings(self) -> list[Holding]:
        raise NotSupported(SOURCE, "holdings require the broker API")


def _to_utc(ts: pd.Timestamp) -> datetime | None:
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        # Daily bars carry a bare date: treat as that day's 00:00 UTC (conservative age).
        return datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
    return ts.tz_convert("UTC").to_pydatetime()

"""yfinance adapter against DataFrames shaped like yfinance 1.x output."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from app.providers.base import NotSupported, SourceUnavailable
from app.providers.yfinance_provider import YFinanceProvider, split_by_ticker, yf_ticker

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _ohlc(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    n = len(index)
    close = [base + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Adj Close": close,
            "Volume": [1000.0] * n,
        },
        index=index,
    )


def _multi(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """group_by='ticker' layout: columns (Ticker, Price)."""
    return pd.concat(frames, axis=1)


def test_ticker_mapping() -> None:
    assert yf_ticker("RELIANCE", "NSE") == "RELIANCE.NS"
    assert yf_ticker("500325", "BSE") == "500325.BO"
    assert yf_ticker("NIFTY 50", "INDEX") == "^NSEI"
    assert yf_ticker("^GSPC", "GLOBAL") == "^GSPC"
    with pytest.raises(NotSupported):
        yf_ticker("NIFTY MIDCAP 150", "INDEX")


def test_split_by_ticker_both_layouts() -> None:
    idx = pd.date_range("2026-10-26", periods=3, freq="B")
    a, b = _ohlc(idx, 100), _ohlc(idx, 200)
    by_ticker = _multi({"A": a, "B": b})
    assert set(split_by_ticker(by_ticker, ["A", "B"])) == {"A", "B"}
    by_column = by_ticker.swaplevel(axis=1)
    out = split_by_ticker(by_column, ["A", "B"])
    assert out["B"]["Close"].iloc[0] == 200
    assert list(split_by_ticker(a, ["A"])) == ["A"]


def test_daily_bars_many_batched_and_filtered() -> None:
    calls: list[dict[str, Any]] = []
    idx = pd.date_range("2026-10-20", periods=8, freq="B")

    def dl(**kw: Any) -> pd.DataFrame:
        calls.append(kw)
        return _multi({"RELIANCE.NS": _ohlc(idx, 1400), "INFY.NS": _ohlc(idx, 1500)})

    p = YFinanceProvider(downloader=dl)
    out = p.daily_bars_many(
        [("RELIANCE", "NSE"), ("INFY", "NSE")], date(2026, 10, 22), date(2026, 10, 28)
    )
    assert len(calls) == 1, "one batched request for all tickers"
    assert calls[0]["auto_adjust"] is False
    rel = out[("RELIANCE", "NSE")]
    assert rel[0].trade_date == date(2026, 10, 22) and rel[-1].trade_date == date(2026, 10, 28)
    assert rel[0].close == 1402


def test_quotes_prefers_newer_intraday_print() -> None:
    daily_idx = pd.date_range("2026-10-26", periods=4, freq="B")
    intra_idx = pd.date_range("2026-10-29 13:00", periods=3, freq="5min", tz="America/New_York")

    def dl(**kw: Any) -> pd.DataFrame:
        if kw["interval"] == "1d":
            return _multi({"^GSPC": _ohlc(daily_idx, 5000), "ES=F": _ohlc(daily_idx, 5010)})
        return _multi({"ES=F": _ohlc(intra_idx, 5050)})

    q = YFinanceProvider(downloader=dl).quotes([("^GSPC", "GLOBAL"), ("ES=F", "GLOBAL")])
    spx, es = q[("^GSPC", "GLOBAL")], q[("ES=F", "GLOBAL")]
    assert spx.last == 5003 and spx.prev_close == 5002
    assert es.last == 5052, "latest 5-minute print beats the daily close"
    assert es.prev_close == 5012
    assert es.as_of is not None and es.as_of.tzinfo is not None
    assert es.change_pct == pytest.approx((5052 / 5012 - 1) * 100)


def test_quotes_survive_intraday_failure() -> None:
    idx = pd.date_range("2026-10-26", periods=3, freq="B")

    def dl(**kw: Any) -> pd.DataFrame:
        if kw["interval"] == "5m":
            raise RuntimeError("rate limited")
        return _ohlc(idx, 80).pipe(lambda d: _multi({"CL=F": d}))

    q = YFinanceProvider(downloader=dl).quotes([("CL=F", "GLOBAL")])
    assert q[("CL=F", "GLOBAL")].last == 82


def test_empty_and_errors_raise_source_unavailable() -> None:
    with pytest.raises(SourceUnavailable):
        YFinanceProvider(downloader=lambda **kw: pd.DataFrame()).daily_bars(
            "X", "NSE", date(2026, 1, 1), date(2026, 1, 5)
        )

    def boom(**kw: Any) -> pd.DataFrame:
        raise ValueError("Yahoo said no")

    with pytest.raises(SourceUnavailable, match="Yahoo said no"):
        YFinanceProvider(downloader=boom).quotes([("^GSPC", "GLOBAL")])


def test_no_intraday_or_holdings() -> None:
    p = YFinanceProvider(downloader=lambda **kw: pd.DataFrame())
    with pytest.raises(NotSupported):
        p.holdings()

"""Kite adapter against responses shaped like kiteconnect 5.x output."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timedelta
from typing import Any

import pytest

import app.providers.kite as kite_module
from app.config import IST
from app.providers.base import AuthExpired, NotSupported, SourceUnavailable
from app.providers.kite import KiteProvider


class TokenException(Exception):
    pass


class PermissionException(Exception):
    pass


class FakeKite:
    def __init__(self) -> None:
        self.hist_calls: list[tuple[int, datetime, datetime, str]] = []
        self.raise_on: Exception | None = None

    def historical_data(
        self, token: int, f: datetime, t: datetime, interval: str, **_: Any
    ) -> list[dict[str, Any]]:
        if self.raise_on:
            raise self.raise_on
        self.hist_calls.append((token, f, t, interval))
        rows = []
        d = f.date()
        while d <= t.date():
            if d.weekday() < 5:
                rows.append(
                    {
                        "date": datetime(d.year, d.month, d.day, tzinfo=IST),
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 100,
                    }
                )
            d += timedelta(days=1)
        return rows

    def quote(self, *instruments: Any) -> dict[str, dict[str, Any]]:
        keys = instruments[0]
        return {
            k: {
                "last_price": 22000.5,
                "ohlc": {"close": 21900.0},
                "volume": None,
                "timestamp": datetime(2026, 10, 30, 9, 20, tzinfo=IST),
            }
            for k in keys
        }

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "instrument_token": 738561,
                "tradingsymbol": "RELIANCE",
                "name": "RELIANCE INDUSTRIES",
                "instrument_type": "EQ",
                "segment": "NSE",
                "lot_size": 1,
                "exchange": "NSE",
            },
            {
                "instrument_token": 256265,
                "tradingsymbol": "NIFTY 50",
                "name": "NIFTY 50",
                "instrument_type": "EQ",
                "segment": "INDICES",
                "lot_size": 0,
                "exchange": "NSE",
            },
            {
                "instrument_token": 1,
                "tradingsymbol": "RELIANCE-BE",
                "name": "X",
                "instrument_type": "FUT",
                "segment": "NFO-FUT",
                "lot_size": 250,
                "exchange": "NSE",
            },
        ]

    def holdings(self) -> list[dict[str, Any]]:
        return [
            {
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "isin": "INE009A01021",
                "quantity": 10,
                "t1_quantity": 2,
                "average_price": 1400.0,
                "last_price": 1500.0,
            }
        ]


def _provider(fake: FakeKite) -> KiteProvider:
    tokens = {("RELIANCE", "NSE"): 738561, ("NIFTY 50", "INDEX"): 256265}
    return KiteProvider(fake, lambda s, e: tokens.get((s, e)))


def test_daily_history_is_chunked_within_kite_limit() -> None:
    fake = FakeKite()
    bars = _provider(fake).daily_bars("RELIANCE", "NSE", date(2020, 1, 1), date(2026, 10, 29))
    assert len(fake.hist_calls) >= 2
    for _, f, t, interval in fake.hist_calls:
        assert interval == "day"
        assert (t - f) <= timedelta(days=2000)
    # Chunks are contiguous and non-overlapping.
    for (_, _, t1, _), (_, f2, _, _) in zip(fake.hist_calls, fake.hist_calls[1:], strict=False):
        assert f2 - t1 == timedelta(seconds=1)
    assert bars[0].trade_date == date(2020, 1, 1)
    assert bars[-1].trade_date == date(2026, 10, 29)
    assert len({b.trade_date for b in bars}) == len(bars)


def test_unknown_symbol_not_supported() -> None:
    with pytest.raises(NotSupported):
        _provider(FakeKite()).daily_bars("NOPE", "NSE", date(2026, 1, 1), date(2026, 1, 2))


def test_quotes_map_prev_close_from_ohlc() -> None:
    q = _provider(FakeKite()).quotes([("NIFTY 50", "INDEX")])
    nifty = q[("NIFTY 50", "INDEX")]
    assert nifty.last == 22000.5 and nifty.prev_close == 21900.0
    assert nifty.change_pct == pytest.approx((22000.5 / 21900 - 1) * 100)


def test_instruments_keep_equities_and_indices_only() -> None:
    recs = _provider(FakeKite()).instruments("NSE")
    assert {(r.symbol, r.exchange, r.is_index) for r in recs} == {
        ("RELIANCE", "NSE", False),
        ("NIFTY 50", "INDEX", True),
    }


def test_holdings_include_t1() -> None:
    h = _provider(FakeKite()).holdings()
    assert h[0].quantity == 12


def test_error_translation() -> None:
    fake = FakeKite()
    fake.raise_on = TokenException("Incorrect `api_key` or `access_token`.")
    with pytest.raises(AuthExpired):
        _provider(fake).daily_bars("RELIANCE", "NSE", date(2026, 1, 1), date(2026, 1, 5))
    fake.raise_on = PermissionException("Insufficient permission for that call.")
    with pytest.raises(SourceUnavailable) as ei:
        _provider(fake).daily_bars("RELIANCE", "NSE", date(2026, 1, 1), date(2026, 1, 5))
    assert ei.value.hint and "Connect plan" in ei.value.hint


def test_adapter_never_touches_order_endpoints() -> None:
    """Dalal is read-only. The adapter source must not reference any order/GTT/MF write API."""
    src = inspect.getsource(kite_module)
    forbidden = [
        "place_order",
        "modify_order",
        "cancel_order",
        "exit_order",
        "place_gtt",
        "modify_gtt",
        "delete_gtt",
        "place_mf_order",
        "place_mf_sip",
        "convert_position",
    ]
    hits = [f for f in forbidden if f in src]
    assert not hits, f"order endpoints referenced in kite adapter: {hits}"

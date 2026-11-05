"""Indicator correctness and lookahead safety.

The causality test is the seed of Phase 2's leakage suite: an indicator's value on day t
must be identical whether or not days after t exist in the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import indicators as ind


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    vol = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_sma_known_values() -> None:
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert list(ind.sma(s, 3).dropna()) == [2.0, 3.0, 4.0]


def test_rsi_bounds_and_extremes() -> None:
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(up, 14).dropna().eq(100).all()
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ind.rsi(down, 14).dropna().eq(0).all()


def _wilder_rsi_reference(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Textbook Wilder RSI: SMA seed over the first n changes, then recursive smoothing."""
    d = np.diff(close)
    gains, losses = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(close), np.nan)
    ag, al = gains[:n].mean(), losses[:n].mean()
    out[n] = 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al)
    return out


def test_rsi_converges_to_wilder_reference(ohlcv: pd.DataFrame) -> None:
    ours = ind.rsi(ohlcv["close"], 14).to_numpy()
    ref = _wilder_rsi_reference(ohlcv["close"].to_numpy(), 14)
    # Seeding differs (EWM from the first change vs SMA of 14); after the warm-up both
    # are the same recursion, so they agree to within rounding. Charts load 300 days of
    # warm-up for exactly this reason.
    assert np.nanmax(np.abs(ours[150:] - ref[150:])) < 0.01
    assert np.isnan(ours[:14]).all()


def test_bollinger_band_order(ohlcv: pd.DataFrame) -> None:
    up, mid, lo = ind.bollinger(ohlcv["close"])
    valid = mid.notna()
    assert (up[valid] >= mid[valid]).all() and (mid[valid] >= lo[valid]).all()


CAUSAL = {
    "sma50": lambda d: ind.sma(d["close"], 50),
    "ema20": lambda d: ind.ema(d["close"], 20),
    "rsi14": lambda d: ind.rsi(d["close"], 14),
    "macd": lambda d: ind.macd(d["close"])[0],
    "macd_signal": lambda d: ind.macd(d["close"])[1],
    "bb_upper": lambda d: ind.bollinger(d["close"])[0],
    "atr14": lambda d: ind.atr(d["high"], d["low"], d["close"]),
    "vol_z": lambda d: ind.volume_zscore(d["volume"]),
}


@pytest.mark.parametrize("name", list(CAUSAL))
def test_no_lookahead(name: str, ohlcv: pd.DataFrame) -> None:
    fn = CAUSAL[name]
    full = fn(ohlcv)
    for cut in (120, 250, 399):
        partial = fn(ohlcv.iloc[:cut])
        pd.testing.assert_series_equal(
            partial, full.iloc[:cut], check_names=False, obj=f"{name} truncated at {cut}"
        )


def test_future_shock_does_not_change_past(ohlcv: pd.DataFrame) -> None:
    shocked = ohlcv.copy()
    shocked.iloc[300:, shocked.columns.get_loc("close")] *= 3
    for name, fn in CAUSAL.items():
        a, b = fn(ohlcv).iloc[:300], fn(shocked).iloc[:300]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=name)


def test_support_resistance_levels(ohlcv: pd.DataFrame) -> None:
    levels = ind.support_resistance(ohlcv.tail(250))
    assert levels, "a random walk over a year should produce some levels"
    last = ohlcv["close"].iloc[-1]
    for lv in levels:
        assert lv["kind"] == ("support" if lv["price"] < last else "resistance")
        assert int(lv["touches"]) >= 2
    prices = [lv["price"] for lv in levels]
    assert prices == sorted(prices)

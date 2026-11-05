"""Technical indicators for charts (and, from Phase 2, features).

Every function here is causal: the value at row t uses only rows <= t. Phase 2's
lookahead-leakage tests rely on this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # All gains, no losses -> RSI 100.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def volume_zscore(volume: pd.Series, n: int = 20) -> pd.Series:
    """Today's volume vs the previous n days (excludes today from the baseline)."""
    base = volume.shift(1).rolling(n, min_periods=n)
    return (volume - base.mean()) / base.std(ddof=0).replace(0.0, np.nan)


def support_resistance(
    df: pd.DataFrame, window: int = 5, tolerance_pct: float = 1.5, max_levels: int = 6
) -> list[dict[str, float | int | str]]:
    """Cluster swing highs/lows into price levels.

    A swing high/low is the extreme of a (2*window+1)-bar neighbourhood. Swings within
    `tolerance_pct` of each other merge into one level; strength = number of touches.
    Levels are returned relative to the last close (support below, resistance above).
    For display only: uses future bars to confirm a swing, so never use it as a feature.
    """
    if len(df) < 2 * window + 1:
        return []
    highs, lows = df["high"], df["low"]
    span = 2 * window + 1
    is_high = highs == highs.rolling(span, center=True).max()
    is_low = lows == lows.rolling(span, center=True).min()
    pivots = sorted(list(highs[is_high].values) + list(lows[is_low].values))
    clusters: list[list[float]] = []
    for p in pivots:
        if clusters and abs(p / np.mean(clusters[-1]) - 1) * 100 <= tolerance_pct:
            clusters[-1].append(float(p))
        else:
            clusters.append([float(p)])
    last = float(df["close"].iloc[-1])
    levels = [
        {
            "price": round(float(np.mean(c)), 2),
            "touches": len(c),
            "kind": "support" if np.mean(c) < last else "resistance",
        }
        for c in clusters
        if len(c) >= 2
    ]
    levels.sort(key=lambda lv: abs(float(lv["price"]) / last - 1))
    return sorted(levels[:max_levels], key=lambda lv: float(lv["price"]))

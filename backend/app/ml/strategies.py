"""Transparent, rule-based candidate generators for stock ideas (brief §4.2, §4.3).

Four long-only setups, each a plain boolean rule over the point-in-time stock-feature
frame that `app.ml.ideas_features` builds (one row per stock per session D, every number
computed from data dated <= D). A rule either fires or it does not; nothing here is
fitted. The LightGBM model in `app.ml.ideas_models` then estimates, for every candidate,
P(target before stop), with the strategy itself as a feature.

    breakout_20d        close above the prior 20-session high, volume z >= 1, above the 200-DMA
    momentum_12_1       top decile of 12-month-minus-1-month relative strength vs NIFTY 50,
                        above the 50-DMA
    mean_reversion_rsi  RSI(14) < 30, above the 200-DMA, three consecutive down closes
    pullback_20dma      20-DMA > 50-DMA > 200-DMA, close within 1.5 % of the 20-DMA after a
                        run of >= 5 % over the prior 20 sessions

Each candidate carries the setup name, three to five reasons that quote the numbers the
rule looked at, and the one observation that would invalidate it. A reason is only
written when its number exists: a missing value is never replaced by a default, and a
rule whose inputs are missing simply does not fire (NaN comparisons are False).

The frame is `app.ml.ideas_features.stock_features` (MultiIndex (symbol, session_date),
META_COLUMNS then FEATURE_NAMES) or any flat frame with the same column names; `COLUMNS`
maps each concept a rule reads to the accepted names. A row whose `eligible` flag is
False (the point-in-time universe verdict) never fires, and the momentum decile is ranked
within the eligible rows of each session. The leakage rules for the columns belong to the
feature pipeline. Pure numpy/pandas: the API may import this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.ml.core import finite

STRATEGY_NAMES: tuple[str, ...] = (
    "breakout_20d",
    "momentum_12_1",
    "mean_reversion_rsi",
    "pullback_20dma",
)
# `Idea.setup` (String 32): the setup family the brief names; one per strategy.
SETUPS: dict[str, str] = {
    "breakout_20d": "breakout",
    "momentum_12_1": "momentum",
    "mean_reversion_rsi": "mean_reversion",
    "pullback_20dma": "pullback",
}
SETUP_LABELS: dict[str, str] = {
    "breakout_20d": "20-day breakout",
    "momentum_12_1": "12-1 month momentum",
    "mean_reversion_rsi": "RSI mean reversion in an uptrend",
    "pullback_20dma": "Pullback to the 20-DMA",
}

# Rule thresholds (contract §2), named so the reasons text and the tests quote one source.
BREAKOUT_VOLUME_Z = 1.0
MOMENTUM_TOP_SHARE = 0.10  # top decile of the universe's relative strength
RSI_OVERSOLD = 30.0
DOWN_CLOSES = 3
PULLBACK_BAND_PCT = 1.5
PULLBACK_MIN_RUN_PCT = 5.0

ID_COLUMNS: tuple[str, str] = ("symbol", "session_date")

# Concept -> column names accepted for it, first match wins. The first name is the one
# `app.ml.ideas_features` writes. Definitions (all as of D):
#   close             last close (Rs)                                   [META]
#   atr14_pct         ATR(14) as % of close
#   vol_z20           volume z-score vs the previous 20 sessions (NaN when volume is missing)
#   rsi14             Wilder RSI(14)
#   dist_sma{n}_pct   close / SMA(n) - 1, in %
#   up_streak         signed run of closes: +n after n up closes, -n after n down closes
#                     (the `app.ml.features` convention); an unsigned `down_streak` (n
#                     after n down closes, 0 otherwise) is accepted instead
#   dist_high20_pct   close / max(high over sessions D-20..D-1) - 1, in % (> 0 = breakout)
#   rs_12_1           stock return over sessions D-250..D-20 minus NIFTY 50's, in points
#   rs_12_1_rank      cross-sectional percentile of rs_12_1 within the universe at D, (0, 1]
#                     (optional: computed within the frame per session when absent)
#   ret_20d           20-session return, in % (the "run" of the pullback setup)
#   eligible          the universe rule at D (bool)                      [META]
#   instrument_id     the stock's Instrument.id                          [META]
COLUMNS: dict[str, tuple[str, ...]] = {
    "close": ("close",),
    "atr14_pct": ("atr14_pct", "atr_pct"),
    "vol_z20": ("vol_z20", "volume_z20"),
    "rsi14": ("rsi14", "rsi_14"),
    "dist_sma20_pct": ("dist_sma20_pct",),
    "dist_sma50_pct": ("dist_sma50_pct",),
    "dist_sma200_pct": ("dist_sma200_pct",),
    "up_streak": ("up_streak",),
    "down_streak": ("down_streak",),
    "brk_20d_high_pct": ("dist_high20_pct", "brk_20d_high_pct", "breakout_20d_pct"),
    "rs_12_1": ("rs_12_1", "rs_12_1_pct"),
    "rs_12_1_rank": ("rs_12_1_rank", "rs_12_1_pct_rank"),
    "ret_20d": ("ret_20d",),
    "eligible": ("eligible",),
    "instrument_id": ("instrument_id",),
}
# Plain-English labels for the columns THIS module adds to a frame (the drivers view);
# `app.ml.ideas_features.FEATURE_LABELS` covers the feature pipeline's own names.
IDEA_FEATURE_LABELS: dict[str, str] = {
    "rs_12_1_rank": "Relative-strength percentile in the universe",
    "strategy_breakout_20d": "Setup: 20-day breakout",
    "strategy_momentum_12_1": "Setup: 12-1 month momentum",
    "strategy_mean_reversion_rsi": "Setup: RSI mean reversion",
    "strategy_pullback_20dma": "Setup: pullback to the 20-DMA",
}
ONE_HOT_PREFIX = "strategy_"
RANK_COLUMN = "rs_12_1_rank"
STREAK_COLUMN = "down_streak"


@dataclass(frozen=True)
class Candidate:
    """One stock-session-strategy that passed a rule, with the words that explain it.

    `close` and `atr14_pct` are carried from the frame row so the ranking can price the
    entry zone, stop and targets without a second lookup (None when missing: such a
    candidate can be scored but never priced, and `app.ml.ranking` drops it);
    `instrument_id` when the frame has it (the ledger's `IdeaRecordIn.instrument_id`).
    """

    symbol: str
    session_date: date
    strategy: str
    setup: str
    reasons: list[str]
    invalidation: str
    close: float | None = None
    atr14_pct: float | None = None
    instrument_id: int | None = None


# --------------------------------------------------------------------------- frame access


def normalise_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """A flat frame with `symbol` and `session_date` columns (python dates), whatever the
    caller indexed it by: a (session_date, symbol) MultiIndex, a `session_date` index with
    a `symbol` column, or one row as a Series. Raises ValueError without both identifiers.
    """
    df = frame.to_frame().T if isinstance(frame, pd.Series) else frame
    levels = [n for n in df.index.names if n in ID_COLUMNS and n not in df.columns]
    if levels:  # a (session_date, symbol) MultiIndex, or a session_date index
        df = df.reset_index(level=levels)
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"stock frame needs columns {ID_COLUMNS}; missing {missing}")
    out = df.reset_index(drop=True).copy()
    out["session_date"] = [_as_date(d) for d in out["session_date"]]
    out["symbol"] = out["symbol"].astype(str)
    return out


def _as_date(x: object) -> date:
    if isinstance(x, datetime):  # pd.Timestamp is a datetime subclass
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(str(x)).date()


def resolve_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Concept -> the frame column that holds it (first accepted name present), or None."""
    cols = set(map(str, frame.columns))
    return {
        concept: next((n for n in names if n in cols), None) for concept, names in COLUMNS.items()
    }


def _series(frame: pd.DataFrame, found: Mapping[str, str | None], concept: str) -> pd.Series:
    """The numeric column for `concept`, or an all-NaN series when the frame lacks it."""
    col = found.get(concept)
    if col is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").astype(float)


def eligible_mask(frame: pd.DataFrame, found: Mapping[str, str | None] | None = None) -> pd.Series:
    """The frame's `eligible` flag as booleans (a missing flag counts as False, a frame
    without the column as all True: the caller already applied the universe rule)."""
    found = found or resolve_columns(frame)
    col = found.get("eligible")
    if col is None:
        return pd.Series(True, index=frame.index, dtype=bool)
    raw = frame[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.astype(bool)
    return pd.Series(
        [bool(v) if v is not None and v == v else False for v in raw.to_numpy(dtype=object)],
        index=frame.index,
        dtype=bool,
    )


def down_streak(frame: pd.DataFrame, found: Mapping[str, str | None] | None = None) -> pd.Series:
    """Consecutive lower closes ending at D, derived from the signed `up_streak` (-n after
    n down closes -> n; an up run -> 0), else the frame's own `down_streak`; NaN stays NaN."""
    found = found or resolve_columns(frame)
    if found.get("up_streak") is not None:
        return (-_series(frame, found, "up_streak")).clip(lower=0.0)
    return _series(frame, found, "down_streak")


def rs_rank(frame: pd.DataFrame, found: Mapping[str, str | None] | None = None) -> pd.Series:
    """Cross-sectional percentile of `rs_12_1` within each session of the frame, (0, 1].

    Uses the frame's own `rs_12_1_rank` when it has one (the universe-wide rank the
    feature pipeline computed); otherwise ranks the ELIGIBLE rows present per session,
    which is only the universe rank when the frame holds the whole universe for that
    session. Ineligible rows and a lone row get NaN: one stock cannot be ranked against
    itself, and a stock outside the universe is not part of its decile.
    """
    found = found or resolve_columns(frame)
    if found.get("rs_12_1_rank") is not None:
        return _series(frame, found, "rs_12_1_rank")
    rs = _series(frame, found, "rs_12_1").where(eligible_mask(frame, found))
    if rs.notna().sum() == 0:
        return rs
    counts = rs.groupby(frame["session_date"]).transform("count")
    ranked = rs.groupby(frame["session_date"]).rank(pct=True, method="average")
    return ranked.where(counts > 1)


# --------------------------------------------------------------------------- the rules


def signals(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """One boolean column per strategy (STRATEGY_NAMES order), aligned with the frame rows.

    Vectorised; a NaN in any input of a rule makes that rule False for the row, and so
    does an `eligible` flag that is False.
    """
    df = normalise_frame(frame)
    f = resolve_columns(df)
    close = _series(df, f, "close")
    d20, d50, d200 = (
        _series(df, f, k) for k in ("dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct")
    )
    priced = (close > 0) & eligible_mask(df, f)
    out = pd.DataFrame(index=df.index)
    out["breakout_20d"] = (
        priced
        & (_series(df, f, "brk_20d_high_pct") > 0.0)
        & (_series(df, f, "vol_z20") >= BREAKOUT_VOLUME_Z)
        & (d200 > 0.0)
    )
    out["momentum_12_1"] = priced & (rs_rank(df, f) >= 1.0 - MOMENTUM_TOP_SHARE) & (d50 > 0.0)
    out["mean_reversion_rsi"] = (
        priced
        & (_series(df, f, "rsi14") < RSI_OVERSOLD)
        & (d200 > 0.0)
        & (down_streak(df, f) >= DOWN_CLOSES)
    )
    out["pullback_20dma"] = (
        priced
        & (d20 < d50)  # SMA20 > SMA50  <=>  close/SMA20 < close/SMA50
        & (d50 < d200)  # SMA50 > SMA200
        & (d20.abs() <= PULLBACK_BAND_PCT)
        & (_series(df, f, "ret_20d") >= PULLBACK_MIN_RUN_PCT)
    )
    return out[list(STRATEGY_NAMES)].fillna(False).astype(bool)


def candidate_frame(frame: pd.DataFrame | pd.Series, session: date | None = None) -> pd.DataFrame:
    """The rows that fired, one per (row, strategy), with `strategy` and `setup` added.

    For training: every original column is kept (the stock features at D), the strategy
    is a column, and a stock-session that passes two rules appears twice, once per
    strategy, as the contract pools all strategies into one model. Rows whose `eligible`
    flag is False are dropped first (outside the universe at D: never a candidate), the
    momentum percentile is ranked on the eligible cross-section of each session and kept
    as `rs_12_1_rank`, and the unsigned `down_streak` is added next to a signed
    `up_streak`, so the words, the live row and the training rows all see the same
    numbers. Index: fresh range.
    """
    df = normalise_frame(frame)
    if session is not None:
        df = df[df["session_date"] == session]
    f = resolve_columns(df)
    df = df[eligible_mask(df, f).to_numpy()]
    if df.empty:
        return df.assign(strategy=pd.Series(dtype=str), setup=pd.Series(dtype=str))
    if f["rs_12_1_rank"] is None and f["rs_12_1"] is not None:
        # The universe percentile is ranked on the whole cross-section, before any rule
        # filters it, and kept as a column so the words and the model see the same number.
        df = df.assign(**{RANK_COLUMN: rs_rank(df, f).to_numpy()})
    if f["up_streak"] is None and f["down_streak"] is None:
        pass  # neither spelling: the mean-reversion rule cannot fire (NaN)
    elif f["up_streak"] is not None and f["down_streak"] is None:
        df = df.assign(**{STREAK_COLUMN: down_streak(df, f).to_numpy()})
    sig = signals(df)
    parts: list[pd.DataFrame] = []
    for strategy in STRATEGY_NAMES:
        hit = df[sig[strategy].to_numpy()]
        if hit.empty:
            continue
        parts.append(hit.assign(strategy=strategy, setup=SETUPS[strategy]))
    if not parts:
        return df.iloc[0:0].assign(strategy=pd.Series(dtype=str), setup=pd.Series(dtype=str))
    out = pd.concat(parts, ignore_index=True)
    order = {s: i for i, s in enumerate(STRATEGY_NAMES)}
    out["_order"] = out["strategy"].map(order)
    out = out.sort_values(["session_date", "_order", "symbol"], kind="stable")
    return out.drop(columns="_order").reset_index(drop=True)


def candidates(frame: pd.DataFrame | pd.Series, session: date | None = None) -> list[Candidate]:
    """Every (stock, session, strategy) that passes a rule, with reasons and invalidation.

    `frame` is the stock-feature frame (many sessions, or the live cross-section of one
    session) or a single row. With `session`, only that session's rows are considered;
    the momentum decile is ranked within the eligible rows given, so pass the whole
    universe of the session (see `rs_rank`). Sorted by session, strategy order, symbol,
    so the output is deterministic.
    """
    cf = candidate_frame(frame, session)
    if cf.empty:
        return []
    f = resolve_columns(cf)
    out: list[Candidate] = []
    for _, row in cf.iterrows():
        values = {concept: finite(row[col]) if col else None for concept, col in f.items()}
        strategy = str(row["strategy"])
        iid = values.get("instrument_id")
        out.append(
            Candidate(
                symbol=str(row["symbol"]),
                session_date=_as_date(row["session_date"]),
                strategy=strategy,
                setup=SETUPS[strategy],
                reasons=reasons_for(strategy, values),
                invalidation=invalidation_for(strategy, values),
                close=values.get("close"),
                atr14_pct=values.get("atr14_pct"),
                instrument_id=int(iid) if iid is not None else None,
            )
        )
    return out


# --------------------------------------------------------------------------- the words


def _rs(x: float) -> str:
    return f"Rs {x:,.2f}"


def _level(close: float | None, dist_pct: float | None) -> float | None:
    """The price a `dist_pct` distance refers to: close / (1 + dist/100)."""
    if close is None or dist_pct is None or dist_pct <= -100.0:
        return None
    return close / (1.0 + dist_pct / 100.0)


def _atr_reason(v: Mapping[str, float | None]) -> str | None:
    atr = v.get("atr14_pct")
    if atr is None:
        return None
    return f"ATR(14) is {atr:.1f}% of price; stop and targets are multiples of it"


def _rsi_reason(v: Mapping[str, float | None]) -> str | None:
    rsi = v.get("rsi14")
    return None if rsi is None else f"RSI(14) at {rsi:.0f}"


def _breakout_reasons(v: Mapping[str, float | None]) -> list[str | None]:
    close, dist = v.get("close"), v.get("brk_20d_high_pct")
    high20 = _level(close, dist)
    ret20 = v.get("ret_20d")
    vz, d200 = v.get("vol_z20"), v.get("dist_sma200_pct")
    return [
        (
            f"Close {_rs(close)} is {dist:+.1f}% above the prior 20-session high ({_rs(high20)})"
            if close is not None and dist is not None and high20 is not None
            else None
        ),
        (
            f"Volume {vz:+.1f} sd above its 20-session average (rule: >= {BREAKOUT_VOLUME_Z:+.1f})"
            if vz is not None
            else None
        ),
        f"{d200:+.1f}% above the 200-DMA: long-term uptrend intact" if d200 is not None else None,
        f"{ret20:+.1f}% over the prior 20 sessions" if ret20 is not None else None,
        _atr_reason(v),
    ]


def _breakout_invalidation(v: Mapping[str, float | None]) -> str:
    high20 = _level(v.get("close"), v.get("brk_20d_high_pct"))
    if high20 is None:
        return "A close back below the breakout level: the breakout failed"
    return f"A close back below the breakout level ({_rs(high20)}): the breakout failed"


def _momentum_reasons(v: Mapping[str, float | None]) -> list[str | None]:
    rs, rank = v.get("rs_12_1"), v.get("rs_12_1_rank")
    d50, d200 = v.get("dist_sma50_pct"), v.get("dist_sma200_pct")
    sma50 = _level(v.get("close"), d50)
    first: str | None
    if rs is not None and rank is not None:
        top = max(1, int(round((1.0 - rank) * 100.0)))
        first = (
            f"12-1 month relative strength vs NIFTY 50 {rs:+.1f} pts: top {top}% of the universe"
        )
    elif rank is not None:
        top = max(1, int(round((1.0 - rank) * 100.0)))
        first = f"12-1 month relative strength vs NIFTY 50 in the top {top}% of the universe"
    else:
        first = None
    return [
        first,
        (
            f"{d50:+.1f}% above the 50-DMA ({_rs(sma50)})"
            if d50 is not None and sma50 is not None
            else None
        ),
        f"{d200:+.1f}% versus the 200-DMA" if d200 is not None else None,
        _rsi_reason(v),
        _atr_reason(v),
    ]


def _momentum_invalidation(v: Mapping[str, float | None]) -> str:
    sma50 = _level(v.get("close"), v.get("dist_sma50_pct"))
    if sma50 is None:
        return "A close below the 50-DMA breaks the trend"
    return f"A close below the 50-DMA ({_rs(sma50)}) breaks the trend"


def _mean_reversion_reasons(v: Mapping[str, float | None]) -> list[str | None]:
    rsi, streak = v.get("rsi14"), v.get("up_streak")
    streak = max(-streak, 0.0) if streak is not None else v.get("down_streak")
    d200, d20 = v.get("dist_sma200_pct"), v.get("dist_sma20_pct")
    sma200 = _level(v.get("close"), d200)
    return [
        f"RSI(14) at {rsi:.0f}: oversold (below {RSI_OVERSOLD:.0f})" if rsi is not None else None,
        f"{int(round(streak))} consecutive down closes" if streak is not None else None,
        (
            f"Still {d200:+.1f}% above the 200-DMA ({_rs(sma200)}): a dip inside an uptrend"
            if d200 is not None and sma200 is not None
            else None
        ),
        f"{d20:+.1f}% from the 20-DMA" if d20 is not None else None,
        _atr_reason(v),
    ]


def _mean_reversion_invalidation(v: Mapping[str, float | None]) -> str:
    sma200 = _level(v.get("close"), v.get("dist_sma200_pct"))
    if sma200 is None:
        return "A close below the 200-DMA turns the dip into a downtrend"
    return f"A close below the 200-DMA ({_rs(sma200)}) turns the dip into a downtrend"


def _pullback_reasons(v: Mapping[str, float | None]) -> list[str | None]:
    close = v.get("close")
    d20, d50, d200 = v.get("dist_sma20_pct"), v.get("dist_sma50_pct"), v.get("dist_sma200_pct")
    s20, s50, s200 = _level(close, d20), _level(close, d50), _level(close, d200)
    run = v.get("ret_20d")
    return [
        (
            f"Uptrend: 20-DMA {_rs(s20)} > 50-DMA {_rs(s50)} > 200-DMA {_rs(s200)}"
            if s20 is not None and s50 is not None and s200 is not None
            else None
        ),
        (
            f"Close {d20:+.1f}% from the 20-DMA (within +/-{PULLBACK_BAND_PCT:.1f}%)"
            if d20 is not None
            else None
        ),
        f"{run:+.1f}% run over the prior 20 sessions" if run is not None else None,
        _rsi_reason(v),
        _atr_reason(v),
    ]


def _pullback_invalidation(v: Mapping[str, float | None]) -> str:
    sma50 = _level(v.get("close"), v.get("dist_sma50_pct"))
    if sma50 is None:
        return "A close below the 50-DMA ends the pullback setup"
    return f"A close below the 50-DMA ({_rs(sma50)}) ends the pullback setup"


_REASONS: dict[str, Callable[[Mapping[str, float | None]], list[str | None]]] = {
    "breakout_20d": _breakout_reasons,
    "momentum_12_1": _momentum_reasons,
    "mean_reversion_rsi": _mean_reversion_reasons,
    "pullback_20dma": _pullback_reasons,
}
_INVALIDATION: dict[str, Callable[[Mapping[str, float | None]], str]] = {
    "breakout_20d": _breakout_invalidation,
    "momentum_12_1": _momentum_invalidation,
    "mean_reversion_rsi": _mean_reversion_invalidation,
    "pullback_20dma": _pullback_invalidation,
}
MAX_REASONS = 5


def reasons_for(strategy: str, values: Mapping[str, float | None]) -> list[str]:
    """Three to five reasons with numbers for a candidate; a reason whose number is
    missing is left out rather than written around a default. `values` maps the
    `COLUMNS` concepts to floats (None when missing)."""
    if strategy not in _REASONS:
        raise ValueError(f"unknown strategy {strategy!r}")
    return [r for r in _REASONS[strategy](values) if r is not None][:MAX_REASONS]


def invalidation_for(strategy: str, values: Mapping[str, float | None]) -> str:
    """The one observation that ends the setup, with its price level when known."""
    if strategy not in _INVALIDATION:
        raise ValueError(f"unknown strategy {strategy!r}")
    return _INVALIDATION[strategy](values)


# --------------------------------------------------------------------------- one-hots


def one_hot_names() -> list[str]:
    """The strategy one-hot feature names, in STRATEGY_NAMES order."""
    return [f"{ONE_HOT_PREFIX}{s}" for s in STRATEGY_NAMES]


def one_hot_row(strategy: str) -> dict[str, float]:
    """{strategy_<name>: 1.0 or 0.0} for one candidate; unknown strategies raise."""
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"unknown strategy {strategy!r}")
    return {f"{ONE_HOT_PREFIX}{s}": (1.0 if s == strategy else 0.0) for s in STRATEGY_NAMES}


def one_hots(strategy: pd.Series) -> pd.DataFrame:
    """One-hot columns for a `strategy` column, aligned with its index; NaN strategy -> all 0."""
    values = strategy.astype(object)
    return pd.DataFrame(
        {f"{ONE_HOT_PREFIX}{s}": (values == s).astype(float) for s in STRATEGY_NAMES},
        index=strategy.index,
    )

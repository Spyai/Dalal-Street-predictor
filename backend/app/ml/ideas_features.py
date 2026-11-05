"""Point-in-time stock features, universe filter, triple-barrier labels and purged
walk-forward folds for the stock ideas (brief §4.2, §4.3; Phase 3 contract §1.A, §2).

Vocabulary
- A *signal session* D is the session whose close the idea is generated from: the
  premarket job runs at 07:30 on the next session and uses data dated <= D. The entry
  is the open of the session after D; the label is the path of the following
  `HorizonSpec.days` sessions (D+1 .. D+days, the entry session included).
- Row (symbol, D) of `stock_features` uses the stock's own bars dated <= D and the
  NIFTY 50 bars dated <= D (relative strength). The six market-context columns are
  copied from `features.build_frame(db, "NIFTY 50")` row D, which is built from data
  dated <= D-1 (what the market call for D knew): one session older than the stock
  features, never newer. All of it is causal; `tests/test_ideas_features.py` proves it by
  truncating and perturbing the database.
- `universe_at(d)` and the `eligible` column apply the same rule to data <= d: an
  `in_universe` NSE stock with >= `ideas_universe_min_bars` session bars, a bar within
  the last RECENT_BAR_SESSIONS sessions, close >= MIN_PRICE and a 20-session mean
  turnover (close x volume, volume in shares) >= `min_avg_turnover_cr` crore. No ASM/GSM
  source is reachable, so surveillance is unknown everywhere (`Idea.surveillance_unknown`).
- Labels (contract §2): target = entry x (1 + k_t ATR14(D)/close(D)), stop = entry x
  (1 - k_s ATR14(D)/close(D)); walking forward over the window with high/low, the target
  touched first -> 1, the stop first -> 0, both on one day -> stop (conservative), neither
  -> 0 with `expired=True` and the realised return to the last close kept for EV
  estimation. A signal is labelled only when its whole window has closed (purging): the
  last `days` sessions are unlabelled, whatever happened inside the partial window, so
  early resolutions are not over-represented. The exit price follows the live grading
  rule: the barrier price, or the open when the session gapped through it.
- `walk_forward_folds` are expanding windows over time-ordered sessions with an embargo
  of `days` sessions between the last training signal and the first test signal, so no
  training label window reaches a test signal.

Numpy/pandas only: the API may import this module (never LightGBM; see
tests/test_models.py::test_ml_not_in_api). Missing numbers stay NaN; nothing is filled
with a default.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.logging import get_logger
from app.ml import features
from app.ml.core import ENTRY_ZONE_ATR, HorizonSpec
from app.models import DailyBar, Instrument, MarketHoliday
from app.services import indicators as ind
from app.services.calendar import TradingCalendar
from app.services.instruments import get_instrument, universe_instruments

log = get_logger(__name__)

NIFTY_SYMBOL = "NIFTY 50"
MIN_PRICE = 20.0  # close floor in rupees (contract §2 "Universe at D")
RECENT_BAR_SESSIONS = 5  # a universe stock must have traded within the last five sessions
TURNOVER_SESSIONS = 20  # the turnover filter averages this many sessions
CRORE = 1e7
RETURN_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 250)
RS_WINDOWS: tuple[int, ...] = (20, 60, 120, 250)
DMA_WINDOWS: tuple[int, ...] = (20, 50, 200)
MOMENTUM_LONG, MOMENTUM_SKIP = 250, 20  # "12-1" momentum: 250 sessions back to 20 sessions back
BREAKOUT_WINDOW = 20
HIGH_WINDOW = 250
ATR_WINDOW = 14
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Insertion order IS the column order. Stock features: the stock's bars <= D (+ NIFTY 50
# bars <= D for relative strength). A name shared with `app.ml.features` (ret_*, dist_sma*,
# rsi14, atr14_pct, vol_z20, rvol_20d, up_streak) has the same definition and label there.
STOCK_FEATURE_LABELS: dict[str, str] = {
    "ret_1d": "1-day return (%)",
    "ret_5d": "5-day return (%)",
    "ret_10d": "10-day return (%)",
    "ret_20d": "20-day return (%)",
    "ret_60d": "60-day return (%)",
    "ret_120d": "120-day return (%)",
    "ret_250d": "250-day return (%)",
    "ret_12_1": "12-month minus 1-month return (%)",
    "rs_20d": "Relative strength vs Nifty 50, 20 days (pp)",
    "rs_60d": "Relative strength vs Nifty 50, 60 days (pp)",
    "rs_120d": "Relative strength vs Nifty 50, 120 days (pp)",
    "rs_250d": "Relative strength vs Nifty 50, 250 days (pp)",
    "rs_12_1": "12-1 month relative strength vs Nifty 50 (pp)",
    "dist_sma20_pct": "Distance from 20-DMA (%)",
    "dist_sma50_pct": "Distance from 50-DMA (%)",
    "dist_sma200_pct": "Distance from 200-DMA (%)",
    "rsi14": "RSI (14)",
    "atr14_pct": "ATR (14) as % of price",
    "dist_high20_pct": "Close vs prior 20-session high (%)",
    "dist_low20_pct": "Close vs prior 20-session low (%)",
    "dist_high250_pct": "Close vs 250-session high (%)",
    "days_since_high250": "Sessions since the 250-session high",
    "vol_z20": "Volume z-score (20 days)",
    "rel_vol_20d": "Volume vs 20-session average (x)",
    "rvol_20d": "Realised volatility, 20 days (daily %)",
    "up_streak": "Consecutive up (+) / down (-) sessions",
    "gap_pct": "Opening gap of the session (%)",
    "turnover_20d_cr": "Average daily turnover, 20 sessions (Rs cr)",
}
STOCK_FEATURE_NAMES: list[str] = list(STOCK_FEATURE_LABELS)
# Market context: `features.build_frame(db, "NIFTY 50")` row D (data <= D-1), renamed.
MARKET_CONTEXT_SOURCE: dict[str, str] = {
    "ret_5d": "mkt_ret_5d",
    "ret_20d": "mkt_ret_20d",
    "vix_level": "mkt_vix_level",
    "pct_above_200dma": "mkt_pct_above_200dma",
    "adv_decl_ratio": "mkt_adv_decl_ratio",
    "rvol_20d": "mkt_rvol_20d",
}
MARKET_CONTEXT_LABELS: dict[str, str] = {
    "mkt_ret_5d": "Nifty 50 5-day return, previous session (%)",
    "mkt_ret_20d": "Nifty 50 20-day return, previous session (%)",
    "mkt_vix_level": "India VIX level, previous session",
    "mkt_pct_above_200dma": "Universe stocks above their 200-DMA (%)",
    "mkt_adv_decl_ratio": "Advance/decline ratio, previous session",
    "mkt_rvol_20d": "Nifty 50 realised volatility, 20 days (daily %)",
}
MARKET_CONTEXT_NAMES: list[str] = list(MARKET_CONTEXT_LABELS)
assert list(MARKET_CONTEXT_SOURCE.values()) == MARKET_CONTEXT_NAMES
assert all(src in features.FEATURE_NAMES for src in MARKET_CONTEXT_SOURCE)
# The model's feature vocabulary (owner B adds the strategy one-hots).
FEATURE_LABELS: dict[str, str] = {**STOCK_FEATURE_LABELS, **MARKET_CONTEXT_LABELS}
FEATURE_NAMES: list[str] = list(FEATURE_LABELS)
# Not features: identity, price and the point-in-time universe verdict of each row.
META_COLUMNS: tuple[str, ...] = ("instrument_id", "close", "n_bars", "eligible")
# The label definition `triple_barrier_labels` implements, stored in every ideas model's
# artifact. Bump it whenever the labelled event changes (anchor, zone, tie rule, window):
# a champion trained on another definition is then replaced on the next training run
# whatever its metrics say, because its out-of-sample numbers describe a different event.
LABEL_VERSION = "2026-11-05.close-anchored.entry-zone.stop-wins-ties"
# `triple_barrier_labels` output, in order.
LABEL_COLUMNS: tuple[str, ...] = (
    "entry_session",
    "entry_price",
    "target_pct",
    "stop_pct",
    "target_price",
    "stop_price",
    "window_closed",
    "invalidated",
    "label",
    "expired",
    "exit_session",
    "exit_price",
    "realised_return_pct",
    "days_held",
)
CANDIDATE_COLUMNS: tuple[str, ...] = ("symbol", "session_date", "strategy", "setup")
INDEX_NAMES: tuple[str, str] = ("symbol", "session_date")


class Fold(NamedTuple):
    """One walk-forward fold over signal sessions: train on sessions <= train_end, test on
    test_start .. test_end (inclusive). Unpacks as (train_end, test_start, test_end)."""

    train_end: date
    test_start: date
    test_end: date


# ----------------------------------------------------------------------------- helpers


def _as_date(x: object) -> date:
    if isinstance(x, datetime):  # pd.Timestamp is a datetime subclass
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(str(x)).date()


def _numeric_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Float OHLCV (None -> NaN; a missing column -> NaN) on a tz-naive DatetimeIndex."""
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None))
    for col in PRICE_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col].to_numpy(), errors="coerce").astype(float)
        else:
            out[col] = np.nan
    return out


def _pct_ret(close: pd.Series, k: int) -> pd.Series:
    return (close / close.shift(k) - 1.0) * 100.0


def _skip_ret(close: pd.Series, skip: int, long: int) -> pd.Series:
    """Return from `long` sessions back to `skip` sessions back (the 12-1 momentum)."""
    return (close.shift(skip) / close.shift(long) - 1.0) * 100.0


def _days_since_max(s: pd.Series, n: int) -> pd.Series:
    """Sessions since the most recent maximum inside the trailing n-session window
    (0 = today is the window high). NaN during warm-up or when the window has a gap."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) >= n:
        filled = np.where(np.isnan(vals), -np.inf, vals)
        win = sliding_window_view(filled, n)[:, ::-1]  # most recent first
        since = np.argmax(win, axis=1).astype(float)
        since[np.isneginf(win).any(axis=1)] = np.nan
        out[n - 1 :] = since
    return pd.Series(out, index=s.index)


def _streak(close: pd.Series) -> pd.Series:
    """+n after n consecutive up closes, -n after n down closes, 0 on an unchanged close
    (the `app.ml.features` convention; "three down closes" is `up_streak <= -3`)."""
    sign = np.sign(close.diff())
    run_id = sign.ne(sign.shift()).cumsum()
    run_len = sign.groupby(run_id).cumcount() + 1
    return (sign * run_len).where(sign.notna())


def _turnover_cr(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Traded value per session in crore; NaN when the volume is not reported."""
    return close * volume / CRORE


def _reported(volume: pd.Series) -> pd.Series:
    """Volume with 0 / missing treated as not reported (never as a real zero)."""
    return volume.where(volume > 0)


def _sessions_back(cal: TradingCalendar, d: date, k: int) -> date:
    """The k-th trading session before the last session <= d (k = 0 is that session)."""
    cur = d if cal.is_trading_day(d) else cal.previous_trading_day(d)
    for _ in range(k):
        cur = cal.previous_trading_day(cur)
    return cur


def _sql_session_filter() -> Any:
    """SQL form of `TradingCalendar.is_trading_day`: weekdays that are not CM holidays."""
    holidays = select(MarketHoliday.holiday_date).where(MarketHoliday.segment == "CM")
    return and_(
        func.extract("dow", DailyBar.trade_date).notin_([0, 6]),
        DailyBar.trade_date.notin_(holidays),
    )


# ------------------------------------------------------------------------------ loading


def _symbols(db: Session, ids: Sequence[int]) -> dict[int, str]:
    rows = db.execute(select(Instrument.id, Instrument.symbol).where(Instrument.id.in_(ids))).all()
    by_id = {int(i): str(sym) for i, sym in rows}
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown instrument ids: {unknown[:10]}")
    seen: dict[str, int] = {}
    for iid, sym in by_id.items():
        if sym in seen:
            raise ValueError(f"symbol {sym} maps to instruments {seen[sym]} and {iid}")
        seen[sym] = iid
    return by_id


def _load_bars(
    db: Session, ids: Sequence[int], through: date, cal: TradingCalendar
) -> dict[int, pd.DataFrame]:
    """Session bars dated <= through per instrument (weekend/holiday bars dropped),
    float OHLCV on a DatetimeIndex, ascending. Instruments without bars are absent."""
    if not ids:
        return {}
    rows = db.execute(
        select(
            DailyBar.instrument_id,
            DailyBar.trade_date,
            DailyBar.open,
            DailyBar.high,
            DailyBar.low,
            DailyBar.close,
            DailyBar.volume,
        )
        .where(DailyBar.instrument_id.in_(list(ids)), DailyBar.trade_date <= through)
        .order_by(DailyBar.instrument_id, DailyBar.trade_date)
    ).all()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["iid", "date", *PRICE_COLUMNS])
    is_session = {d: cal.is_trading_day(d) for d in pd.unique(df["date"].to_numpy())}
    df = df[df["date"].map(is_session).to_numpy(dtype=bool)]
    out: dict[int, pd.DataFrame] = {}
    for _, g in df.groupby("iid", sort=True):
        iid = int(g["iid"].to_numpy()[0])
        bars = g.set_index(pd.DatetimeIndex(pd.to_datetime(g["date"])))[list(PRICE_COLUMNS)]
        out[iid] = _numeric_bars(bars)
    return out


def _nifty_close(db: Session, through: date, cal: TradingCalendar) -> pd.Series:
    inst = get_instrument(db, NIFTY_SYMBOL, "INDEX")
    bars = _load_bars(db, [inst.id], through, cal) if inst is not None else {}
    if inst is None or inst.id not in bars:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return bars[inst.id]["close"]


# ----------------------------------------------------------------------- stock features


def _stock_block(bars: pd.DataFrame, nifty_close: pd.Series) -> pd.DataFrame:
    """STOCK_FEATURE_NAMES as of each bar date t from the stock's rows <= t and the
    NIFTY 50 closes on the same dates (all indicators are causal; see indicators.py)."""
    close, open_, high, low = bars["close"], bars["open"], bars["high"], bars["low"]
    volume = _reported(bars["volume"])
    nifty = nifty_close.reindex(bars.index)
    out = pd.DataFrame(index=bars.index)
    for k in RETURN_WINDOWS:
        out[f"ret_{k}d"] = _pct_ret(close, k)
    out["ret_12_1"] = _skip_ret(close, MOMENTUM_SKIP, MOMENTUM_LONG)
    for k in RS_WINDOWS:
        out[f"rs_{k}d"] = out[f"ret_{k}d"] - _pct_ret(nifty, k)
    out["rs_12_1"] = out["ret_12_1"] - _skip_ret(nifty, MOMENTUM_SKIP, MOMENTUM_LONG)
    for n in DMA_WINDOWS:
        out[f"dist_sma{n}_pct"] = (close / ind.sma(close, n) - 1.0) * 100.0
    out["rsi14"] = ind.rsi(close, ATR_WINDOW)
    out["atr14_pct"] = ind.atr(high, low, close, ATR_WINDOW) / close * 100.0
    prior_high = high.shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max()
    prior_low = low.shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).min()
    out["dist_high20_pct"] = (close / prior_high - 1.0) * 100.0
    out["dist_low20_pct"] = (close / prior_low - 1.0) * 100.0
    high_250 = high.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW).max()
    out["dist_high250_pct"] = (close / high_250 - 1.0) * 100.0
    out["days_since_high250"] = _days_since_max(high, HIGH_WINDOW)
    out["vol_z20"] = ind.volume_zscore(volume, 20)
    out["rel_vol_20d"] = volume / volume.shift(1).rolling(20, min_periods=20).mean()
    out["rvol_20d"] = out["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
    out["up_streak"] = _streak(close)
    out["gap_pct"] = (open_ / close.shift(1) - 1.0) * 100.0
    turnover = _turnover_cr(close, volume)
    out["turnover_20d_cr"] = turnover.rolling(
        TURNOVER_SESSIONS, min_periods=TURNOVER_SESSIONS
    ).mean()
    return out


def _empty_features_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*META_COLUMNS, *FEATURE_NAMES])
    frame.index = pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES)
    return frame


def _features_frame(
    db: Session,
    bars_by_id: Mapping[int, pd.DataFrame],
    symbol_by_id: Mapping[int, str],
    through: date,
    cal: TradingCalendar,
    settings: Settings,
) -> pd.DataFrame:
    nifty_close = _nifty_close(db, through, cal)
    blocks: list[pd.DataFrame] = []
    for iid in sorted(bars_by_id, key=lambda i: symbol_by_id[i]):
        bars = bars_by_id[iid]
        if bars.empty:
            continue
        block = _stock_block(bars, nifty_close)
        n = len(bars)
        meta = pd.DataFrame(
            {
                "instrument_id": np.full(n, iid, dtype=int),
                "close": bars["close"].to_numpy(dtype=float),
                "n_bars": np.arange(1, n + 1, dtype=int),
            },
            index=bars.index,
        )
        meta["eligible"] = (
            (meta["n_bars"] >= settings.ideas_universe_min_bars)
            & (meta["close"] >= MIN_PRICE)
            & (block["turnover_20d_cr"] >= settings.min_avg_turnover_cr)
        )
        both = pd.concat([meta, block], axis=1)
        both.index = pd.MultiIndex.from_arrays(
            [[symbol_by_id[iid]] * n, [t.date() for t in bars.index]], names=INDEX_NAMES
        )
        blocks.append(both)
    if not blocks:
        return _empty_features_frame()
    frame = pd.concat(blocks)

    # Market context: build_frame row D (data <= D-1), aligned on the session_date level.
    mkt = features.build_frame(db, NIFTY_SYMBOL, end=through, settings=settings)
    ctx = mkt.reindex(columns=list(MARKET_CONTEXT_SOURCE)).rename(columns=MARKET_CONTEXT_SOURCE)
    aligned = ctx.reindex(frame.index.get_level_values("session_date"))
    for col in MARKET_CONTEXT_NAMES:
        frame[col] = aligned[col].to_numpy(dtype=float)
    frame = frame.reindex(columns=[*META_COLUMNS, *FEATURE_NAMES])
    log.debug(
        "stock_feature_frame_built",
        stocks=len(blocks),
        rows=len(frame),
        through=str(through),
        eligible_rows=int(frame["eligible"].sum()),
    )
    return frame


def stock_features(
    db: Session,
    instrument_ids: Iterable[int],
    through: date,
    settings: Settings | None = None,
    *,
    start: date | None = None,
) -> pd.DataFrame:
    """Point-in-time features per (symbol, session_date) for the given NSE instruments.

    One row per session bar dated <= `through` (weekend/holiday bars dropped). Index:
    MultiIndex (symbol, session_date as python date), sorted. Columns: META_COLUMNS
    (instrument_id, close at D, n_bars = session bars <= D, eligible = the universe rule at
    D; a row exists only on a bar date, so "a bar in the last 5 sessions" holds by
    construction) then FEATURE_NAMES (STOCK_FEATURE_NAMES from bars <= D, MARKET_CONTEXT_NAMES from
    `features.build_frame` row D). `start` only trims the returned rows (session_date >=
    start); the features are always computed on the full history. Missing values are NaN.
    Raises ValueError for unknown ids or two ids sharing a symbol.
    """
    s = settings or get_settings()
    ids = sorted({int(i) for i in instrument_ids})
    if not ids:
        return _empty_features_frame()
    symbol_by_id = _symbols(db, ids)
    cal = TradingCalendar.load(db)
    bars = _load_bars(db, ids, _as_date(through), cal)
    frame = _features_frame(db, bars, symbol_by_id, _as_date(through), cal, s)
    if start is not None and not frame.empty:
        keep = np.array([d >= start for d in frame.index.get_level_values("session_date")])
        frame = frame[keep]
    return frame


# ------------------------------------------------------------------------------ universe


def universe_at(db: Session, d: date, settings: Settings | None = None) -> list[Instrument]:
    """The idea universe as of `d`, from data dated <= d only (contract §2).

    `in_universe` NSE instruments with >= `ideas_universe_min_bars` session bars, a bar
    within the last RECENT_BAR_SESSIONS sessions ending at d, latest close >= MIN_PRICE,
    and the mean turnover (close x volume, crore) of their last TURNOVER_SESSIONS session
    bars >= `min_avg_turnover_cr`; a stock whose volume is unreported on any of those
    sessions has no turnover and is out. Sorted by symbol. Surveillance (ASM/GSM) is not
    checked here or anywhere: there is no reachable source.
    """
    s = settings or get_settings()
    d = _as_date(d)
    members = universe_instruments(db)
    if not members:
        return []
    cal = TradingCalendar.load(db)
    recent_cutoff = _sessions_back(cal, d, RECENT_BAR_SESSIONS - 1)
    ids = [m.id for m in members]
    sessions_only = _sql_session_filter()
    counts = db.execute(
        select(DailyBar.instrument_id, func.count(), func.max(DailyBar.trade_date))
        .where(DailyBar.instrument_id.in_(ids), DailyBar.trade_date <= d, sessions_only)
        .group_by(DailyBar.instrument_id)
    ).all()
    candidates = [
        int(iid)
        for iid, n_bars, last in counts
        if int(n_bars) >= s.ideas_universe_min_bars and last >= recent_cutoff
    ]
    if not candidates:
        return []
    rn = (
        func.row_number()
        .over(partition_by=DailyBar.instrument_id, order_by=DailyBar.trade_date.desc())
        .label("rn")
    )
    recent = (
        select(DailyBar.instrument_id, DailyBar.close, DailyBar.volume, rn)
        .where(DailyBar.instrument_id.in_(candidates), DailyBar.trade_date <= d, sessions_only)
        .subquery()
    )
    rows = db.execute(
        select(recent.c.instrument_id, recent.c.close, recent.c.volume, recent.c.rn).where(
            recent.c.rn <= TURNOVER_SESSIONS
        )
    ).all()
    df = pd.DataFrame(rows, columns=["iid", "close", "volume", "rn"])
    close = pd.to_numeric(df["close"].to_numpy(), errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"].to_numpy(), errors="coerce").astype(float)
    df["close"] = close
    df["turnover_cr"] = _turnover_cr(df["close"], _reported(pd.Series(volume, index=df.index)))
    by = df.groupby("iid")
    full_window = (by.size() == TURNOVER_SESSIONS) & (
        by["turnover_cr"].count() == TURNOVER_SESSIONS
    )
    turnover = by["turnover_cr"].mean().where(full_window)
    last_close = df[df["rn"] == 1].set_index("iid")["close"]
    ok = {
        int(iid)
        for iid in turnover.index
        if turnover[iid] >= s.min_avg_turnover_cr and last_close.get(iid, np.nan) >= MIN_PRICE
    }
    return [m for m in members if m.id in ok]


# -------------------------------------------------------------------------------- labels


def _empty_labels(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for col in LABEL_COLUMNS:
        if col in ("entry_session", "exit_session"):
            out[col] = pd.Series([None] * len(index), index=index, dtype=object)
        elif col in ("window_closed", "invalidated", "expired"):
            out[col] = False
        else:
            out[col] = np.nan
    return out


def triple_barrier_labels(
    bars: pd.DataFrame, signal_sessions: Iterable[date], horizon: HorizonSpec
) -> pd.DataFrame:
    """Triple-barrier outcome of the idea the desk would have recorded on each signal session.

    The event is the one the ledger grades (app.services.idea_grading): levels are set
    from the signal session D, target = close(D) x (1 + k_t x ATR14(D)/close(D)) and
    stop = close(D) x (1 - k_s x ...); the idea is filled at the next session's open only
    when that open lies inside the entry zone close(D) +/- core.ENTRY_ZONE_ATR x ATR14(D),
    otherwise it is `invalidated` (never traded, so never labelled: these rows are
    excluded from training and reported as a share). The window is the `days` bars from
    the entry bar on; the stop wins a same-bar tie.

    `bars`: one stock's session bars (DatetimeIndex; open/high/low/close; as returned by
    `services.market_data.daily_frame`, with non-session bars already dropped). Returns a
    frame indexed by the signal sessions in the given order with LABEL_COLUMNS:
    entry_session/entry_price (the next bar and its open; None/NaN when there is none yet),
    target_pct/stop_pct (k_t, k_s x ATR14(D)/close(D) in %), target_price/stop_price
    (anchored at close(D)), window_closed (`days` bars exist after D), invalidated (the
    open was outside the zone), label (1.0 target first, 0.0 stop first or expired, NaN
    when not labelled), expired, exit_session/exit_price (barrier price, the open if the
    session gapped through it, or the last close on expiry), realised_return_pct (exit vs
    entry, %), days_held (entry session = 1). A signal not among the bar dates or with no
    ATR yet stays unlabelled. Nothing after the window is read.
    """
    signals = [_as_date(d) for d in signal_sessions]
    index = pd.Index(signals, dtype=object, name="session_date")
    days = int(horizon.days)
    if days < 1:
        raise ValueError(f"horizon.days must be positive, got {days}")
    out = _empty_labels(index)
    b = _numeric_bars(bars).sort_index()
    n, m = len(b), len(signals)
    if n == 0 or m == 0:
        return out

    bar_dates = b.index.to_numpy().astype("datetime64[D]")
    bar_pydates = [t.date() for t in b.index]
    sig64 = np.array(signals, dtype="datetime64[D]")
    pos = np.searchsorted(bar_dates, sig64, side="left")
    posc = np.clip(pos, 0, n - 1)
    valid = (pos < n) & (bar_dates[posc] == sig64)
    o = b["open"].to_numpy(dtype=float)
    h = b["high"].to_numpy(dtype=float)
    lo = b["low"].to_numpy(dtype=float)
    c = b["close"].to_numpy(dtype=float)
    atr = ind.atr(b["high"], b["low"], b["close"], ATR_WINDOW).to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        atr_pct = np.where(valid, atr[posc] / c[posc] * 100.0, np.nan)
        target_pct = horizon.target_atr * atr_pct
        stop_pct = horizon.stop_atr * atr_pct
        entry_pos = pos + 1
        has_entry = valid & (entry_pos < n)
        entry_price = np.where(has_entry, o[np.clip(entry_pos, 0, n - 1)], np.nan)
        anchor = np.where(valid, c[posc], np.nan)  # close(D): the levels the ledger records
        target_price = anchor * (1.0 + target_pct / 100.0)
        stop_price = anchor * (1.0 - stop_pct / 100.0)
        zone_half = ENTRY_ZONE_ATR * np.where(valid, atr[posc], np.nan)
        priced = np.isfinite(target_pct) & (target_pct > 0.0) & (stop_pct > 0.0)
        in_zone = has_entry & priced & (np.abs(entry_price - anchor) <= zone_half)
        invalidated = has_entry & priced & ~in_zone  # the 09:20 rule would not have filled
        window_closed = valid & (pos + days < n)
        labelled = window_closed & in_zone

        rows = np.arange(m)
        idx = np.clip(entry_pos[:, None] + np.arange(days)[None, :], 0, n - 1)
        in_window = labelled[:, None]
        hit_target = (h[idx] >= target_price[:, None]) & in_window
        hit_stop = (lo[idx] <= stop_price[:, None]) & in_window
        hit_any = hit_target | hit_stop
        any_hit = hit_any.any(axis=1)
        first = np.argmax(hit_any, axis=1)
        stop_first = any_hit & hit_stop[rows, first]  # both on one day -> stop
        target_first = any_hit & ~stop_first
        expired = labelled & ~any_hit
        exit_off = np.where(any_hit, first, days - 1)
        exit_pos = np.clip(entry_pos + exit_off, 0, n - 1)
        open_at_exit = o[idx[rows, first]]
        last_close = c[idx[:, days - 1]]
        exit_price = np.where(
            stop_first,
            np.minimum(open_at_exit, stop_price),
            np.where(target_first, np.maximum(open_at_exit, target_price), last_close),
        )
        exit_price = np.where(labelled, exit_price, np.nan)
        label = np.where(labelled, np.where(target_first, 1.0, 0.0), np.nan)
        realised = (exit_price / entry_price - 1.0) * 100.0
        days_held = np.where(labelled, exit_off + 1.0, np.nan)

    out["entry_session"] = pd.Series(
        [bar_pydates[p] if ok else None for p, ok in zip(entry_pos, has_entry, strict=True)],
        index=index,
        dtype=object,
    )
    out["entry_price"] = entry_price
    out["target_pct"] = target_pct
    out["stop_pct"] = stop_pct
    out["target_price"] = target_price
    out["stop_price"] = stop_price
    out["window_closed"] = window_closed
    out["invalidated"] = invalidated
    out["label"] = label
    out["expired"] = expired
    out["exit_session"] = pd.Series(
        [bar_pydates[p] if ok else None for p, ok in zip(exit_pos, labelled, strict=True)],
        index=index,
        dtype=object,
    )
    out["exit_price"] = exit_price
    out["realised_return_pct"] = realised
    out["days_held"] = days_held
    return out


# --------------------------------------------------------------------- candidate frame


def _normalise_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Columns symbol, session_date (python date), strategy, setup; index levels accepted."""
    cand = candidates
    if not set(CANDIDATE_COLUMNS) <= set(cand.columns):
        cand = cand.reset_index()
    missing = [c for c in CANDIDATE_COLUMNS if c not in cand.columns]
    if missing:
        raise ValueError(f"candidates lack columns {missing}")
    cand = cand.loc[:, list(CANDIDATE_COLUMNS)].reset_index(drop=True)
    cand["symbol"] = cand["symbol"].astype(str).str.upper()
    cand["session_date"] = pd.Series([_as_date(d) for d in cand["session_date"]], dtype=object)
    cand["strategy"] = cand["strategy"].astype(str)
    cand["setup"] = cand["setup"].astype(str)
    return cand


def _candidate_columns() -> list[str]:
    return [
        *CANDIDATE_COLUMNS,
        "horizon",
        "horizon_days",
        *META_COLUMNS,
        *FEATURE_NAMES,
        *LABEL_COLUMNS,
    ]


def build_candidate_frame(
    db: Session,
    candidates: pd.DataFrame,
    horizon: HorizonSpec,
    settings: Settings | None = None,
    *,
    through: date | None = None,
) -> pd.DataFrame:
    """Features + triple-barrier labels for strategy candidates (the model's training frame).

    `candidates`: columns (or index levels) symbol, session_date, strategy, setup; the same
    (symbol, session_date) may appear once per strategy. Returns one row per candidate,
    sorted by (session_date, symbol, strategy) with a RangeIndex, columns CANDIDATE_COLUMNS
    + horizon, horizon_days + META_COLUMNS + FEATURE_NAMES + LABEL_COLUMNS. Bars dated
    > `through` (default: the last bar of these symbols) are not read; the last
    `horizon.days` sessions therefore have NaN labels (purged). Raises ValueError for an
    unknown NSE symbol or a candidate dated on a session the stock has no bar for.
    """
    s = settings or get_settings()
    cand = _normalise_candidates(candidates)
    if cand.empty:
        return pd.DataFrame(columns=_candidate_columns())
    symbols = sorted(set(cand["symbol"]))
    insts = db.scalars(
        select(Instrument).where(Instrument.exchange == "NSE", Instrument.symbol.in_(symbols))
    ).all()
    by_symbol = {inst.symbol: inst for inst in insts}
    unknown = [sym for sym in symbols if sym not in by_symbol]
    if unknown:
        raise ValueError(f"unknown NSE symbols: {unknown[:10]}")
    ids = [by_symbol[sym].id for sym in symbols]
    if through is None:
        last = db.scalar(
            select(func.max(DailyBar.trade_date)).where(DailyBar.instrument_id.in_(ids))
        )
        if last is None:
            raise ValueError("no daily bars for the candidate symbols")
        through = last
    through = _as_date(through)
    cal = TradingCalendar.load(db)
    bars = _load_bars(db, ids, through, cal)
    symbol_by_id = {by_symbol[sym].id: sym for sym in symbols}
    feats = _features_frame(db, bars, symbol_by_id, through, cal, s)

    key = pd.MultiIndex.from_arrays(
        [cand["symbol"].to_numpy(), cand["session_date"].to_numpy()], names=INDEX_NAMES
    )
    present = key.isin(feats.index) if not feats.empty else np.zeros(len(key), dtype=bool)
    if not present.all():
        bad = [tuple(k) for k in key[~present][:10]]
        raise ValueError(f"candidates without a bar on their signal session: {bad}")
    rows = feats.reindex(key).reset_index(drop=True)

    labels: list[pd.DataFrame] = []
    for sym in symbols:
        sub = cand[cand["symbol"] == sym]
        lab = triple_barrier_labels(
            bars.get(by_symbol[sym].id, pd.DataFrame()), list(sub["session_date"]), horizon
        )
        lab.index = sub.index
        labels.append(lab)
    lab_all = pd.concat(labels).reindex(cand.index)

    out = pd.concat([cand, rows, lab_all], axis=1)
    out["horizon"] = horizon.name
    out["horizon_days"] = np.full(len(out), horizon.days, dtype=int)
    out = out.sort_values(["session_date", "symbol", "strategy"], kind="stable").reset_index(
        drop=True
    )
    log.debug(
        "candidate_frame_built",
        horizon=horizon.name,
        rows=len(out),
        labelled=int(out["label"].notna().sum()),
        through=str(through),
    )
    return out.reindex(columns=_candidate_columns())


# ---------------------------------------------------------------------- walk-forward


def walk_forward_folds(
    sessions: Iterable[date],
    min_train_sessions: int = 250,
    step: int = 21,
    embargo: int | None = None,
) -> list[Fold]:
    """Expanding-window folds over time-ordered signal sessions with an embargo.

    `embargo` has no default on purpose: pass the horizon's `days` (ValueError otherwise).

    `sessions` (deduplicated and sorted) should be every session the frame covers. Fold
    k tests sessions [k, k + step) and trains on sessions <= sessions[k - embargo - 1], so
    exactly `embargo` sessions separate the last training signal from the first test
    signal; with embargo = horizon.days no training label window (D+1 .. D+days) reaches a
    test signal. The first fold has `min_train_sessions` training sessions. Passing
    sparser sessions than the trading calendar only widens the real gap.
    """
    if embargo is None:
        raise ValueError("embargo is required: pass the horizon's days (HorizonSpec.days)")
    if min_train_sessions < 1 or step < 1:
        raise ValueError("min_train_sessions and step must be positive")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    uniq = sorted({_as_date(d) for d in sessions})
    n = len(uniq)
    folds: list[Fold] = []
    for k in range(min_train_sessions + embargo, n, step):
        folds.append(Fold(uniq[k - embargo - 1], uniq[k], uniq[min(k + step, n) - 1]))
    return folds

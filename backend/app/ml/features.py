"""Point-in-time feature pipeline for the market call (brief §4.3; Phase 2 contract §3.A).

One row per trading session D (the IST day whose close is predicted). Every feature of
row D is computed from data dated strictly before D:

- target / India VIX / global-cue daily bars dated <= D-1 (the last bar before D, so a
  bar dated D, even a partial one fetched before the open, is never used);
- FII/DII flows dated on the reference session (published the evening of that day);
- NSE calendar flags for D itself (weekday, sessions left in the month: known in
  advance). There is no expiry-day flag: the weekly expiry weekday moved in 2023, 2024
  and 2025, so a single weekday would mislabel most of the history, and `dow` already
  carries whatever day-of-week effect exists;
- scheduled-event flags from `app.ml.events`.

The same code builds the training frame and the live row (`live_row` is
`build_frame(end=session_date)` and takes the last row), so there is no separate "live"
feature path that could drift from training. `tests/test_features.py` proves the
no-lookahead property by truncating and perturbing the database.

Sessions are weekdays that are not NSE holidays. Bars dated on a weekend or holiday
(Muhurat trading, the Saturday budget session) are dropped before anything is computed.
Missing numbers stay NaN (None in a snapshot); nothing is ever filled with a default.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.logging import get_logger
from app.ml.core import FeatureRow, class_of, finite
from app.ml.events import EVENT_KEYS, load_events
from app.models import DailyBar, FiiDiiFlow
from app.services import indicators as ind
from app.services.calendar import TradingCalendar
from app.services.instruments import GLOBAL_CUES, get_instrument, universe_instruments
from app.services.market_data import daily_frame

log = get_logger(__name__)

VIX_SYMBOL = "INDIA VIX"
RETURN_WINDOWS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)
DMA_WINDOWS: tuple[int, ...] = (20, 50, 200)
GLOBAL_RET_1D: tuple[str, ...] = (
    "sp500",
    "nasdaq",
    "es_fut",
    "nq_fut",
    "nikkei",
    "hang_seng",
    "crude_brent",
    "usdinr",
    "dxy",
)
GLOBAL_RET_5D: tuple[str, ...] = ("sp500", "crude_brent", "usdinr")
# A VIX or global-cue bar older than this (calendar days before D) is treated as missing
# rather than passed off as the latest print. Covers long weekends and short holidays.
MAX_CUE_AGE_DAYS = 7
# Breadth needs at least this many universe instruments with a bar on the day.
MIN_BREADTH_INSTRUMENTS = 20
# The last five sessions of a month (monthly expiry, rebalancing flows).
MONTH_LAST_WEEK_SESSIONS = 5
LABEL_COLUMNS: tuple[str, ...] = ("label_pct", "label_class", "reference_date", "reference_close")

# Insertion order here IS the feature order of every matrix and snapshot.
FEATURE_LABELS: dict[str, str] = {
    "ret_1d": "1-day return (%)",
    "ret_2d": "2-day return (%)",
    "ret_3d": "3-day return (%)",
    "ret_5d": "5-day return (%)",
    "ret_10d": "10-day return (%)",
    "ret_20d": "20-day return (%)",
    "ret_60d": "60-day return (%)",
    "dist_sma20_pct": "Distance from 20-DMA (%)",
    "dist_sma50_pct": "Distance from 50-DMA (%)",
    "dist_sma200_pct": "Distance from 200-DMA (%)",
    "rsi14": "RSI (14)",
    "macd_hist_pct": "MACD histogram (% of price)",
    "atr14_pct": "ATR (14) as % of price",
    "bb_z": "Bollinger z-score (20)",
    "rvol_10d": "Realised volatility, 10 days (daily %)",
    "rvol_20d": "Realised volatility, 20 days (daily %)",
    "vol_z20": "Volume z-score (20 days)",
    "gap_pct_prev": "Previous session's opening gap (%)",
    "range_pct_prev": "Previous session's high-low range (%)",
    "clv_prev": "Previous session's close within its range (-1..1)",
    "up_streak": "Consecutive up (+) / down (-) sessions",
    "dow": "Day of week (0 = Monday)",
    "sessions_to_month_end": "Sessions left in the month",
    "is_month_last_week": "Last five sessions of the month",
    "vix_level": "India VIX level",
    "vix_chg_1d": "India VIX 1-day change (%)",
    "vix_chg_5d": "India VIX 5-day change (%)",
    "sp500_ret_1d": "S&P 500 1-day return (%)",
    "sp500_ret_5d": "S&P 500 5-day return (%)",
    "nasdaq_ret_1d": "Nasdaq 1-day return (%)",
    "es_fut_ret_1d": "S&P 500 futures 1-day return (%)",
    "nq_fut_ret_1d": "Nasdaq futures 1-day return (%)",
    "nikkei_ret_1d": "Nikkei 225 1-day return (%)",
    "hang_seng_ret_1d": "Hang Seng 1-day return (%)",
    "crude_brent_ret_1d": "Brent crude 1-day return (%)",
    "crude_brent_ret_5d": "Brent crude 5-day return (%)",
    "usdinr_ret_1d": "USD/INR 1-day change (%)",
    "usdinr_ret_5d": "USD/INR 5-day change (%)",
    "dxy_ret_1d": "Dollar index 1-day return (%)",
    "us10y_chg_1d": "US 10Y yield 1-day change (points)",
    "fii_net_cr": "FII net cash flow, previous session (Rs cr)",
    "dii_net_cr": "DII net cash flow, previous session (Rs cr)",
    "fii_net_5d_cr": "FII net cash flow, last 5 sessions (Rs cr)",
    "dii_net_5d_cr": "DII net cash flow, last 5 sessions (Rs cr)",
    "pct_above_200dma": "Universe stocks above their 200-DMA (%)",
    "adv_decl_ratio": "Advance/decline ratio, previous session",
    "budget_day": "Union Budget day",
    "rbi_policy_day": "RBI policy day",
    "fomc_overnight": "FOMC decision overnight",
    "us_cpi_overnight": "US CPI release overnight",
    "events_loaded": "Event calendar loaded from events.csv",
}
FEATURE_NAMES: list[str] = list(FEATURE_LABELS)
assert all(k in FEATURE_NAMES for k in EVENT_KEYS)

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


# ----------------------------------------------------------------------------- loading


def _numeric_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Float OHLCV (None -> NaN) on a tz-naive DatetimeIndex."""
    out = pd.DataFrame(index=df.index)
    for col in PRICE_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return out


def _session_bars(df: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
    """Rows dated on trading sessions only: weekend and holiday bars are dropped."""
    if df.empty:
        return df
    days = pd.DatetimeIndex(df.index).date
    keep = np.array([cal.is_trading_day(d) for d in days], dtype=bool)
    return df[keep]


def _through(df: pd.DataFrame, end: date) -> pd.DataFrame:
    """Bars dated <= end. Rows after `end` are never needed (row `end` uses data < end
    for features and only its own close for the label)."""
    if df.empty:
        return df
    return df[df.index <= pd.Timestamp(end)]


def _flow_table(db: Session, end: date) -> pd.DataFrame:
    """FII/DII net cash flows (INR crore) by trade date; NaN where a side is missing."""
    rows = db.execute(
        select(FiiDiiFlow.trade_date, FiiDiiFlow.category, FiiDiiFlow.net_value_cr).where(
            FiiDiiFlow.segment == "cash", FiiDiiFlow.trade_date <= end
        )
    ).all()
    out = pd.DataFrame(columns=["fii_net_cr", "dii_net_cr"], dtype=float)
    if not rows:
        return out
    df = pd.DataFrame(rows, columns=["date", "category", "net"])
    wide = df.pivot(index="date", columns="category", values="net").reindex(columns=["FII", "DII"])
    out = pd.DataFrame(index=pd.to_datetime(wide.index))
    out["fii_net_cr"] = pd.to_numeric(wide["FII"].to_numpy(), errors="coerce").astype(float)
    out["dii_net_cr"] = pd.to_numeric(wide["DII"].to_numpy(), errors="coerce").astype(float)
    return out.sort_index()


def _breadth_table(db: Session, cal: TradingCalendar, end: date) -> pd.DataFrame | None:
    """Market breadth per session over the trading universe, or None without a universe.

    pct_above_200dma: share of instruments (with a close and a 200-DMA that day) above
    their 200-DMA. adv_decl_ratio: advances / declines versus each instrument's previous
    session bar (NaN when nothing declined). Both NaN on days with fewer than
    MIN_BREADTH_INSTRUMENTS instruments reporting. Weekend/holiday bars are dropped.
    """
    ids = [inst.id for inst in universe_instruments(db)]
    if len(ids) < MIN_BREADTH_INSTRUMENTS:
        return None
    rows = db.execute(
        select(DailyBar.trade_date, DailyBar.instrument_id, DailyBar.close).where(
            DailyBar.instrument_id.in_(ids), DailyBar.trade_date <= end
        )
    ).all()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "iid", "close"])
    wide = df.pivot(index="date", columns="iid", values="close").astype(float).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = _session_bars(wide, cal)
    if wide.empty:
        return None
    n_bars = wide.notna().sum(axis=1)
    sma200 = wide.rolling(200, min_periods=200).mean()
    n_valid = (wide.notna() & sma200.notna()).sum(axis=1)
    enough = n_valid.where(n_valid >= MIN_BREADTH_INSTRUMENTS)
    pct_above = (wide > sma200).sum(axis=1) / enough * 100.0
    prev = wide.shift(1)
    adv = (wide > prev).sum(axis=1)
    dec = (wide < prev).sum(axis=1)
    ratio = (adv / dec.where(dec > 0)).where(n_bars >= MIN_BREADTH_INSTRUMENTS)
    return pd.DataFrame({"pct_above_200dma": pct_above, "adv_decl_ratio": ratio})


# ---------------------------------------------------------------------- per-bar features


def _pct_ret(close: pd.Series, k: int) -> pd.Series:
    return (close / close.shift(k) - 1.0) * 100.0


def _streak(close: pd.Series) -> pd.Series:
    """+n after n consecutive up closes, -n after n down closes, 0 on an unchanged close."""
    sign = np.sign(close.diff())
    run_id = sign.ne(sign.shift()).cumsum()
    run_len = sign.groupby(run_id).cumcount() + 1
    return (sign * run_len).where(sign.notna())


def _target_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Technical features as of each bar date t (rows <= t only; see indicators.py)."""
    close, open_, high, low = bars["close"], bars["open"], bars["high"], bars["low"]
    volume = bars["volume"].where(bars["volume"] > 0)  # 0 / missing = not reported
    out = pd.DataFrame(index=bars.index)
    for k in RETURN_WINDOWS:
        out[f"ret_{k}d"] = _pct_ret(close, k)
    for n in DMA_WINDOWS:
        out[f"dist_sma{n}_pct"] = (close / ind.sma(close, n) - 1.0) * 100.0
    out["rsi14"] = ind.rsi(close, 14)
    out["macd_hist_pct"] = ind.macd(close)[2] / close * 100.0
    out["atr14_pct"] = ind.atr(high, low, close, 14) / close * 100.0
    sd20 = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_z"] = (close - ind.sma(close, 20)) / sd20.replace(0.0, np.nan)
    out["rvol_10d"] = out["ret_1d"].rolling(10, min_periods=10).std(ddof=0)
    out["rvol_20d"] = out["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
    out["vol_z20"] = ind.volume_zscore(volume, 20)
    out["gap_pct_prev"] = (open_ / close.shift(1) - 1.0) * 100.0
    out["range_pct_prev"] = (high - low) / close * 100.0
    out["clv_prev"] = (2.0 * close - high - low) / (high - low).replace(0.0, np.nan)
    out["up_streak"] = _streak(close)
    return out


def _vix_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    return pd.DataFrame(
        {
            "vix_level": close,
            "vix_chg_1d": _pct_ret(close, 1),
            "vix_chg_5d": _pct_ret(close, 5),
        }
    )


def _cue_features(key: str, bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    if key == "us10y":
        out["us10y_chg_1d"] = close.diff(1)
    if key in GLOBAL_RET_1D:
        out[f"{key}_ret_1d"] = _pct_ret(close, 1)
    if key in GLOBAL_RET_5D:
        out[f"{key}_ret_5d"] = _pct_ret(close, 5)
    return out


# --------------------------------------------------------------------- session alignment


def _asof_block(table: pd.DataFrame, sessions: np.ndarray, max_age_days: int) -> pd.DataFrame:
    """Row of `table` from the last date strictly before each session (positional index).

    NaN where no earlier row exists or the latest one is more than `max_age_days`
    calendar days old.
    """
    values = np.full((len(sessions), len(table.columns)), np.nan)
    if not table.empty:
        dates = table.index.to_numpy().astype("datetime64[D]")
        pos = np.searchsorted(dates, sessions, side="left") - 1
        valid = pos >= 0
        age_days = (sessions - dates[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(int)
        valid &= age_days <= max_age_days
        values[valid] = table.to_numpy(dtype=float)[pos[valid]]
    return pd.DataFrame(values, columns=list(table.columns))


def _sessions_to_month_end(cal: TradingCalendar, d: date) -> int:
    n, cur = 0, d + timedelta(days=1)
    while cur.month == d.month:
        if cal.is_trading_day(cur):
            n += 1
        cur += timedelta(days=1)
    return n


def _calendar_block(cal: TradingCalendar, sessions: list[date], settings: Settings) -> pd.DataFrame:
    """Calendar and scheduled-event flags for each session (all known in advance)."""
    events = load_events(settings)
    rows: list[dict[str, float]] = []
    for d in sessions:
        left = _sessions_to_month_end(cal, d)
        row = {
            "dow": float(d.weekday()),
            "sessions_to_month_end": float(left),
            "is_month_last_week": float(left < MONTH_LAST_WEEK_SESSIONS),
        }
        row.update(events.flags(d, cal.previous_trading_day(d)))
        rows.append(row)
    return pd.DataFrame(rows)


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*FEATURE_NAMES, *LABEL_COLUMNS])
    frame.index = pd.Index([], dtype=object, name="session_date")
    return frame


# ------------------------------------------------------------------------------- public


def build_frame(
    db: Session,
    target: str,
    *,
    start: date | None = None,
    end: date | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Feature + label frame for an INDEX target ("NIFTY 50", "NIFTY BANK").

    Index: `session_date` (python date, ascending). Columns: FEATURE_NAMES, then
    `label_pct` (close_D vs reference_close, %), `label_class` (UP/FLAT/DOWN or None),
    `reference_date` (last session before D with a bar) and `reference_close`.
    A session with no close yet has NaN label_pct / None label_class but full features,
    so the last row is the live row. `end` is the last session to include (default: the
    session after the last bar); `start` the first. Empty frame if the target has no bars.
    """
    s = settings or get_settings()
    cal = TradingCalendar.load(db)
    inst = get_instrument(db, target, "INDEX")
    if inst is None:
        return _empty_frame()
    bars = _numeric_bars(_session_bars(daily_frame(db, inst.id), cal))
    if bars.empty:
        return _empty_frame()
    last_bar: date = bars.index[-1].date()
    end = end if end is not None else cal.next_trading_day(last_bar)
    bars = _through(bars, end)
    if bars.empty:
        return _empty_frame()
    first = start if start is not None else bars.index[0].date()
    sessions = [d for d in _days(first, end) if cal.is_trading_day(d)]

    bar_dates = bars.index.to_numpy().astype("datetime64[D]")
    sess64 = np.array(sessions, dtype="datetime64[D]")
    ref_pos = np.searchsorted(bar_dates, sess64, side="left") - 1
    has_ref = ref_pos >= 0  # the first bar can only ever be a reference
    sessions = [d for d, ok in zip(sessions, has_ref, strict=True) if ok]
    sess64, ref_pos = sess64[has_ref], ref_pos[has_ref]
    if not sessions:
        return _empty_frame()

    # Everything known as of the reference bar date t (data dated <= t).
    as_of_ref = _target_features(bars)
    flows = _flow_table(db, end).reindex(bars.index)
    as_of_ref["fii_net_cr"] = flows["fii_net_cr"]
    as_of_ref["dii_net_cr"] = flows["dii_net_cr"]
    as_of_ref["fii_net_5d_cr"] = flows["fii_net_cr"].rolling(5, min_periods=5).sum()
    as_of_ref["dii_net_5d_cr"] = flows["dii_net_cr"].rolling(5, min_periods=5).sum()
    breadth = _breadth_table(db, cal, end)
    if breadth is None:
        as_of_ref["pct_above_200dma"] = np.nan
        as_of_ref["adv_decl_ratio"] = np.nan
    else:
        aligned = breadth.reindex(bars.index)
        as_of_ref["pct_above_200dma"] = aligned["pct_above_200dma"]
        as_of_ref["adv_decl_ratio"] = aligned["adv_decl_ratio"]
    target_block = as_of_ref.iloc[ref_pos].reset_index(drop=True)

    # India VIX and global cues: last bar strictly before D, each on its own calendar.
    blocks = [target_block]
    vix_inst = get_instrument(db, VIX_SYMBOL, "INDEX")
    vix_bars = (
        _through(_numeric_bars(_session_bars(daily_frame(db, vix_inst.id), cal)), end)
        if vix_inst is not None
        else _numeric_bars(pd.DataFrame(columns=list(PRICE_COLUMNS)))
    )
    blocks.append(_asof_block(_vix_features(vix_bars), sess64, MAX_CUE_AGE_DAYS))
    for ref in GLOBAL_CUES:
        if ref.key not in GLOBAL_RET_1D and ref.key != "us10y":
            continue
        cue_inst = get_instrument(db, ref.symbol, "GLOBAL")
        cue_bars = (
            _through(_numeric_bars(daily_frame(db, cue_inst.id)), end)
            if cue_inst is not None
            else _numeric_bars(pd.DataFrame(columns=list(PRICE_COLUMNS)))
        )
        blocks.append(_asof_block(_cue_features(ref.key, cue_bars), sess64, MAX_CUE_AGE_DAYS))
    blocks.append(_calendar_block(cal, sessions, s))

    frame = pd.concat(blocks, axis=1).reindex(columns=FEATURE_NAMES).astype(float)

    # Labels: close of D (if the bar exists) against the reference close.
    close = bars["close"].to_numpy(dtype=float)
    ref_close = close[ref_pos]
    close_d = bars["close"].reindex(pd.DatetimeIndex(sess64)).to_numpy(dtype=float)
    label_pct = (close_d / ref_close - 1.0) * 100.0
    thr = s.market_flat_threshold_pct
    frame["label_pct"] = label_pct
    frame["label_class"] = pd.Series(
        [class_of(float(p), thr) if np.isfinite(p) else None for p in label_pct], dtype=object
    )
    ref_dates = [d.astype(object) for d in bar_dates[ref_pos]]
    frame["reference_date"] = pd.Series(ref_dates, dtype=object)
    frame["reference_close"] = ref_close
    frame.index = pd.Index(sessions, dtype=object, name="session_date")
    log.debug(
        "feature_frame_built",
        target=target,
        rows=len(frame),
        first=str(frame.index[0]),
        last=str(frame.index[-1]),
        labelled=int(np.isfinite(label_pct).sum()),
    )
    return frame


def live_row(
    db: Session, target: str, session_date: date, *, settings: Settings | None = None
) -> tuple[FeatureRow, date, float]:
    """(feature_snapshot, reference_date, reference_close) for one session.

    The snapshot has every FEATURE_NAMES key in order, values through `core.finite()`
    (None for missing). Built with the training code path (`build_frame(end=session_date)`).
    Raises LookupError if the target has no bars before `session_date` or `session_date`
    is not a trading session.
    """
    frame = build_frame(db, target, end=session_date, settings=settings)
    if frame.empty:
        raise LookupError(f"{target}: no daily bars before {session_date}")
    if session_date not in frame.index:
        raise LookupError(
            f"{target}: {session_date} is not a trading session (last session in frame: "
            f"{frame.index[-1]})"
        )
    pos = int(frame.index.get_indexer(pd.Index([session_date], dtype=object))[0])
    snapshot: FeatureRow = {name: finite(frame.at[session_date, name]) for name in FEATURE_NAMES}
    reference_date = frame.at[session_date, "reference_date"]
    reference_close = float(frame["reference_close"].to_numpy(dtype=float)[pos])
    if not isinstance(reference_date, date) or not np.isfinite(reference_close):
        raise LookupError(f"{target}: no reference close before {session_date}")
    return snapshot, reference_date, reference_close


def frame_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Feature matrix in FEATURE_NAMES order, NaN for missing (never a filled default)."""
    return frame.reindex(columns=FEATURE_NAMES).astype(float).to_numpy()

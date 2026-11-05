"""Stock-idea features: point-in-time proofs, triple-barrier labels, purging, embargo, universe.

Leakage is the thing to prove. A stock feature of (symbol, D) that read a bar dated after D,
or a label that read a bar on or before its signal session, would inflate every idea metric,
so the proofs run against the database: rebuilding from a database truncated at D leaves
every row <= D identical; corrupting everything dated > D leaves every row <= D identical;
a label's path is shown to start at the next session's open and never look back.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml import features
from app.ml import ideas_features as idf
from app.ml.core import HorizonSpec, horizons
from app.ml.ideas_features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    LABEL_COLUMNS,
    MARKET_CONTEXT_NAMES,
    META_COLUMNS,
    STOCK_FEATURE_LABELS,
    STOCK_FEATURE_NAMES,
    Fold,
    build_candidate_frame,
    stock_features,
    triple_barrier_labels,
    universe_at,
    walk_forward_folds,
)
from app.models import Instrument
from app.providers.base import Bar, HolidayRecord
from app.services import indicators as ind
from app.services import instruments
from app.services.calendar import TradingCalendar, store_holidays
from app.services.market_data import daily_frame, upsert_daily_bars

FIRST, LAST = date(2024, 1, 1), date(2025, 3, 31)
HOLIDAYS = {
    date(2024, 3, 25): "Holi",
    date(2024, 8, 15): "Independence Day",
    date(2025, 2, 26): "Mahashivratri",
}
N_NORMAL = 25
NORMAL = [f"STK{i:03d}" for i in range(N_NORMAL)]
LISTED = date(2024, 9, 2)  # NEWLIST's first bar: too few bars for the default universe rule
HALT_LAST = date(2025, 3, 10)  # HALTED's last bar: no bar in the last five sessions of March
SWING = HorizonSpec("swing", 10, 2.0, 1.0)


def _weekdays(first: date, last: date) -> list[date]:
    return [
        first + timedelta(i)
        for i in range((last - first).days + 1)
        if (first + timedelta(i)).weekday() < 5
    ]


def _bars(
    rng: np.random.Generator,
    days: list[date],
    *,
    start: float,
    vol_lo: float,
    vol_hi: float,
    daily_vol: float = 0.012,
) -> list[Bar]:
    close = start * np.exp(np.cumsum(rng.normal(0.0, daily_vol, len(days))))
    out: list[Bar] = []
    for i, (d, c) in enumerate(zip(days, close, strict=True)):
        prev = close[i - 1] if i else c
        o = prev * (1 + rng.normal(0, 0.004))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.006)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.006)))
        vol = float(rng.uniform(vol_lo, vol_hi)) if vol_hi > 0 else 0.0
        out.append(Bar(d, float(o), float(hi), float(lo), float(c), vol))
    return out


@dataclass
class World:
    db: Session
    cal: TradingCalendar
    sessions: list[date]
    inst: dict[str, Instrument]
    bars: dict[str, list[Bar]]
    nifty: dict[date, Bar]

    def ids(self, symbols: list[str] | None = None) -> list[int]:
        return [self.inst[s].id for s in (symbols or sorted(self.inst))]

    def universe_ids(self) -> list[int]:
        return [i.id for i in self.inst.values() if i.in_universe]

    def by_date(self, symbol: str) -> dict[date, Bar]:
        return {b.trade_date: b for b in self.bars[symbol]}


@pytest.fixture()
def world(db: Session) -> World:
    """15 months of synthetic NSE data: Nifty 50 + VIX, 25 ordinary universe stocks and
    one stock per universe-rule failure (thin turnover, penny price, late listing, halted,
    unreported volume) plus one that is not a universe member at all."""
    rng = np.random.default_rng(7)
    store_holidays(db, [HolidayRecord(d, desc) for d, desc in HOLIDAYS.items()], "test")
    instruments.ensure_reference_instruments(db)
    db.flush()
    cal = TradingCalendar.load(db)
    sessions = [d for d in _weekdays(FIRST, LAST) if d not in HOLIDAYS]

    nifty = instruments.get_instrument(db, "NIFTY 50", "INDEX")
    vix = instruments.get_instrument(db, "INDIA VIX", "INDEX")
    assert nifty is not None and vix is not None
    nifty_bars = _bars(rng, sessions, start=22_000, vol_lo=1e5, vol_hi=2e5, daily_vol=0.008)
    upsert_daily_bars(db, nifty.id, nifty_bars, "yfinance")
    upsert_daily_bars(db, vix.id, _bars(rng, sessions, start=14, vol_lo=0, vol_hi=0), "yfinance")

    inst: dict[str, Instrument] = {}
    bars: dict[str, list[Bar]] = {}

    def add(
        symbol: str,
        days: list[date],
        start: float,
        vol_lo: float,
        vol_hi: float,
        *,
        member: bool = True,
    ) -> None:
        eq = instruments.get_or_create_equity(db, symbol, "NSE")
        eq.in_universe = member
        b = _bars(rng, days, start=start, vol_lo=vol_lo, vol_hi=vol_hi)
        upsert_daily_bars(db, eq.id, b, "yfinance")
        inst[symbol], bars[symbol] = eq, b

    for sym in NORMAL:
        add(sym, sessions, float(rng.uniform(800, 3000)), 3e5, 6e5)
    add("THIN", sessions, 800.0, 500, 1500)  # ~0.1 cr a day
    add("PENNY", sessions, 8.0, 3e5, 6e5)  # below the Rs 20 floor
    add("NEWLIST", [d for d in sessions if d >= LISTED], 900.0, 3e5, 6e5)
    add("HALTED", [d for d in sessions if d <= HALT_LAST], 900.0, 3e5, 6e5)
    add("NOVOL", sessions, 900.0, 0, 0)  # volume never reported
    add("WATCH", sessions, 900.0, 3e5, 6e5, member=False)
    db.flush()
    return World(db, cal, sessions, inst, bars, {b.trade_date: b for b in nifty_bars})


def _rows_through(frame: pd.DataFrame, d: date, *, inclusive: bool = True) -> pd.DataFrame:
    level = frame.index.get_level_values("session_date")
    mask = np.array([(x <= d) if inclusive else (x < d) for x in level], dtype=bool)
    return frame[mask]


def _pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def _flat_bars(n: int, start: date = date(2024, 6, 3)) -> pd.DataFrame:
    """n business days at 100 with a 2-point range: ATR14 is exactly 2.0 from bar 14 on."""
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e5}, index=idx
    )


# ------------------------------------------------------------------------- vocabulary


def test_feature_vocabulary() -> None:
    assert [*STOCK_FEATURE_NAMES, *MARKET_CONTEXT_NAMES] == FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert set(FEATURE_LABELS) == set(FEATURE_NAMES) and set(STOCK_FEATURE_LABELS) == set(
        STOCK_FEATURE_NAMES
    )
    assert all(n.replace("_", "").isalnum() and n == n.lower() for n in FEATURE_NAMES)
    assert all(label and len(label) <= 60 for label in FEATURE_LABELS.values())
    assert not set(FEATURE_NAMES) & set(META_COLUMNS) and not set(FEATURE_NAMES) & set(
        LABEL_COLUMNS
    )
    assert all(n.startswith("mkt_") for n in MARKET_CONTEXT_NAMES)
    assert set(idf.MARKET_CONTEXT_SOURCE) == {
        "ret_5d",
        "ret_20d",
        "vix_level",
        "pct_above_200dma",
        "adv_decl_ratio",
        "rvol_20d",
    }
    assert set(idf.MARKET_CONTEXT_SOURCE) <= set(features.FEATURE_NAMES)
    for name in ("ret_1d", "ret_250d", "rs_250d", "dist_sma200_pct", "rsi14", "atr14_pct"):
        assert name in STOCK_FEATURE_NAMES
    for name in ("dist_high20_pct", "dist_low20_pct", "vol_z20", "days_since_high250"):
        assert name in STOCK_FEATURE_NAMES
    for name in ("up_streak", "gap_pct", "turnover_20d_cr", "rel_vol_20d", "rvol_20d"):
        assert name in STOCK_FEATURE_NAMES
    # A name shared with the market-call features means the same thing, in the same words.
    shared = set(STOCK_FEATURE_NAMES) & set(features.FEATURE_NAMES)
    assert shared >= {"ret_1d", "dist_sma200_pct", "rsi14", "atr14_pct", "vol_z20", "up_streak"}
    assert all(STOCK_FEATURE_LABELS[n] == features.FEATURE_LABELS[n] for n in shared)


def test_streak_and_days_since_max_helpers() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 11.5, 11.0, 11.0, 13.0], dtype=float)
    streak = idf._streak(close)
    assert math.isnan(streak.iloc[0])
    assert list(streak.iloc[1:]) == [1.0, 2.0, -1.0, -2.0, 0.0, 1.0]
    high = pd.Series([5.0, 9.0, 7.0, 9.0, 8.0, 6.0, np.nan, 7.0], dtype=float)
    since = idf._days_since_max(high, 3)
    assert since.iloc[:2].isna().all()
    # Window [5, 9, 7] -> the high was one session ago; ties resolve to the most recent.
    assert list(since.iloc[2:6]) == [1.0, 0.0, 1.0, 2.0]
    assert since.iloc[6:].isna().all()  # a NaN inside the window: unknown, not guessed


def test_module_is_numpy_pandas_only() -> None:
    """The API imports this module, so it must never pull in LightGBM or scikit-learn."""
    tree = ast.parse(Path(idf.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"lightgbm", "sklearn", "mlflow"}
    assert imported >= {"numpy", "pandas", "sqlalchemy"}


# --------------------------------------------------------------------- stock features


def test_stock_features_hand_checks(world: World) -> None:
    db = world.db
    frame = stock_features(db, world.universe_ids(), LAST)
    assert list(frame.index.names) == ["symbol", "session_date"]
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert list(frame.columns) == [*META_COLUMNS, *FEATURE_NAMES]
    assert frame["close"].notna().all() and frame["n_bars"].notna().all()
    assert frame["eligible"].dtype == bool

    sym = "STK003"
    bars = world.bars[sym]
    sessions = [b.trade_date for b in bars]
    d = sessions[-1]
    row = frame.loc[(sym, d)]
    close = [b.close for b in bars]
    high = [b.high for b in bars]
    low = [b.low for b in bars]
    vol = [b.volume or math.nan for b in bars]
    n = len(bars)
    assert row["instrument_id"] == world.inst[sym].id
    assert row["close"] == close[-1] and row["n_bars"] == n
    assert row["ret_1d"] == pytest.approx(_pct(close[-1], close[-2]))
    assert row["ret_5d"] == pytest.approx(_pct(close[-1], close[-6]))
    assert row["ret_250d"] == pytest.approx(_pct(close[-1], close[-251]))
    assert row["ret_12_1"] == pytest.approx(_pct(close[-21], close[-251]))
    assert row["dist_sma20_pct"] == pytest.approx(_pct(close[-1], float(np.mean(close[-20:]))))
    nifty_close = [world.nifty[s].close for s in sessions]
    assert row["rs_20d"] == pytest.approx(
        _pct(close[-1], close[-21]) - _pct(nifty_close[-1], nifty_close[-21])
    )
    assert row["rs_12_1"] == pytest.approx(
        _pct(close[-21], close[-251]) - _pct(nifty_close[-21], nifty_close[-251])
    )
    assert row["dist_high20_pct"] == pytest.approx(_pct(close[-1], max(high[-21:-1])))
    assert row["dist_low20_pct"] == pytest.approx(_pct(close[-1], min(low[-21:-1])))
    assert row["dist_high250_pct"] == pytest.approx(_pct(close[-1], max(high[-250:])))
    window = high[-250:]
    assert row["days_since_high250"] == 249 - max(
        i for i, h in enumerate(window) if h == max(window)
    )
    assert row["gap_pct"] == pytest.approx(_pct(bars[-1].open, close[-2]))
    turnover = [c * v / 1e7 for c, v in zip(close[-20:], vol[-20:], strict=True)]
    assert row["turnover_20d_cr"] == pytest.approx(float(np.mean(turnover)))
    assert row["rel_vol_20d"] == pytest.approx(vol[-1] / float(np.mean(vol[-21:-1])))
    sign = np.sign(close[-1] - close[-2])
    streak = 0
    for i in range(n - 1, 0, -1):
        if np.sign(close[i] - close[i - 1]) == sign:
            streak += 1
        else:
            break
    assert row["up_streak"] == sign * streak
    series = pd.Series(close, dtype=float)
    assert row["rsi14"] == pytest.approx(float(ind.rsi(series, 14).iloc[-1]))
    atr = ind.atr(pd.Series(high, dtype=float), pd.Series(low, dtype=float), series, 14)
    assert row["atr14_pct"] == pytest.approx(float(atr.iloc[-1]) / close[-1] * 100.0)
    rets = series.pct_change() * 100.0
    assert row["rvol_20d"] == pytest.approx(float(rets.iloc[-20:].std(ddof=0)))
    assert row["vol_z20"] == pytest.approx(
        (vol[-1] - np.mean(vol[-21:-1])) / np.std(vol[-21:-1], ddof=0)
    )
    # Default universe rule: 250 bars, Rs 20, 10 crore -> an ordinary stock is eligible at
    # the end and not before it has 250 bars.
    assert bool(row["eligible"]) is True
    assert bool(frame.loc[(sym, sessions[248]), "eligible"]) is False
    assert bool(frame.loc[(sym, sessions[249]), "eligible"]) is True
    assert not frame.loc["THIN", "eligible"].any()
    assert not frame.loc["PENNY", "eligible"].any()
    assert not frame.loc["NOVOL", "eligible"].any()
    assert frame.loc["NOVOL", "turnover_20d_cr"].isna().all()
    assert frame.loc["NOVOL", "vol_z20"].isna().all()
    assert frame.loc["NOVOL", "ret_1d"].iloc[1:].notna().all()
    # Warm-up stays NaN; nothing is filled.
    first = frame.loc[(sym, sessions[0])]
    assert math.isnan(first["ret_1d"]) and math.isnan(first["up_streak"])
    assert math.isnan(frame.loc[(sym, sessions[248]), "ret_250d"])
    assert not math.isnan(frame.loc[(sym, sessions[250]), "ret_250d"])


def test_market_context_is_build_frame_row_d(world: World) -> None:
    db = world.db
    frame = stock_features(db, world.ids(["STK001", "STK002"]), LAST)
    mkt = features.build_frame(db, "NIFTY 50")
    for d in (world.sessions[-1], world.sessions[-30], world.sessions[40]):
        row = frame.loc[("STK001", d)]
        for src, col in idf.MARKET_CONTEXT_SOURCE.items():
            expected = mkt.at[d, src]
            if math.isnan(expected):
                assert math.isnan(row[col]), col
            else:
                assert row[col] == expected, col
        # The same session carries the same context for every stock.
        np.testing.assert_array_equal(
            row[MARKET_CONTEXT_NAMES].to_numpy(float),
            frame.loc[("STK002", d), MARKET_CONTEXT_NAMES].to_numpy(float),
        )
    # Row D of build_frame is built from data <= D-1: the Nifty 5-day return of D equals the
    # stock frame's own view of the Nifty one session later.
    d, prev = world.sessions[-1], world.sessions[-2]
    nifty = [world.nifty[s].close for s in world.sessions]
    assert frame.loc[("STK001", d), "mkt_ret_5d"] == pytest.approx(_pct(nifty[-2], nifty[-7]))
    assert frame.loc[("STK001", prev), "mkt_ret_5d"] == pytest.approx(_pct(nifty[-3], nifty[-8]))
    # The first session has no build_frame row (no reference bar): context missing, not faked.
    assert frame.loc[("STK001", world.sessions[0]), MARKET_CONTEXT_NAMES].isna().all()
    assert frame.loc[("STK001", d), "mkt_pct_above_200dma"] == pytest.approx(
        mkt.at[d, "pct_above_200dma"]
    )
    assert 0 <= frame.loc[("STK001", d), "mkt_pct_above_200dma"] <= 100


def test_stock_features_empty_start_and_errors(world: World) -> None:
    db = world.db
    empty = stock_features(db, [], LAST)
    assert empty.empty and list(empty.columns) == [*META_COLUMNS, *FEATURE_NAMES]
    assert list(empty.index.names) == ["symbol", "session_date"]
    with pytest.raises(ValueError, match="unknown instrument"):
        stock_features(db, [999_999], LAST)
    full = stock_features(db, world.ids(["STK004"]), LAST)
    part = stock_features(db, world.ids(["STK004"]), LAST, start=date(2025, 1, 1))
    assert part.index.get_level_values("session_date").min() == date(2025, 1, 1)
    pd.testing.assert_frame_equal(part, full.loc[part.index])
    # Settings may be passed positionally (contract: stock_features(db, ids, through)).
    pd.testing.assert_frame_equal(
        stock_features(db, world.ids(["STK004"]), LAST, get_settings()), full
    )
    # `through` before any bar: nothing, no error.
    assert stock_features(db, world.ids(["STK004"]), date(2023, 1, 1)).empty
    # A bar on a Sunday (Muhurat-style) is dropped before anything is computed.
    eq = world.inst["STK004"]
    upsert_daily_bars(db, eq.id, [Bar(date(2024, 11, 3), 9e5, 9e6, 9e4, 9e5, 9e9)], "yfinance")
    db.flush()
    pd.testing.assert_frame_equal(stock_features(db, [eq.id], LAST), full)


# ------------------------------------------------------------------ lookahead leakage


def _cuts(world: World) -> list[date]:
    sessions = world.sessions
    monday = next(d for d in sessions if d.weekday() == 0 and d > date(2025, 1, 6))
    after_holiday = world.cal.next_trading_day(date(2025, 2, 26))
    return sorted({sessions[-15], monday, after_holiday})


def test_truncation_invariance(world: World) -> None:
    """Rows <= D built from a database that ends at D equal rows <= D of the full frame."""
    db = world.db
    ids = world.universe_ids()
    full = stock_features(db, ids, LAST)
    for cut in sorted(_cuts(world), reverse=True):  # later cuts first: deletes accumulate
        expected = _rows_through(full, cut)
        # `through` alone must not matter: data after it is never read.
        pd.testing.assert_frame_equal(stock_features(db, ids, cut), expected)
        # Now the data after the cut does not exist at all (stocks, Nifty, VIX alike).
        db.execute(text("DELETE FROM daily_bar WHERE trade_date > :d"), {"d": cut})
        db.flush()
        trunc = stock_features(db, ids, cut)
        assert trunc.index.get_level_values("session_date").max() == cut
        pd.testing.assert_frame_equal(trunc, expected)
        # Data dated exactly D is used by the row of D (it is the signal's own close).
        assert np.isfinite(trunc.loc[("STK000", cut), STOCK_FEATURE_NAMES].to_numpy(float)).all()


def test_future_bars_do_not_change_past_rows(world: World) -> None:
    """Corrupting every bar dated > D leaves every row <= D untouched."""
    db = world.db
    ids = world.universe_ids()
    base = stock_features(db, ids, LAST)
    cut = _cuts(world)[1]
    nxt = world.cal.next_trading_day(cut)
    db.execute(
        text(
            "UPDATE daily_bar SET open = open * 3, high = high * 3, low = low * 3, "
            "close = close * 3, volume = COALESCE(volume, 0) * 7 + 1 WHERE trade_date > :d"
        ),
        {"d": cut},
    )
    db.flush()
    shocked = stock_features(db, ids, LAST)
    assert list(shocked.index) == list(base.index)
    pd.testing.assert_frame_equal(_rows_through(shocked, cut), _rows_through(base, cut))
    # Sanity: the corruption is real where it should be.
    assert shocked.loc[("STK000", nxt), "ret_1d"] != base.loc[("STK000", nxt), "ret_1d"]
    assert shocked.loc[("STK000", nxt), "rs_20d"] != base.loc[("STK000", nxt), "rs_20d"]
    assert (
        shocked.loc[("STK000", nxt), "turnover_20d_cr"]
        != base.loc[("STK000", nxt), "turnover_20d_cr"]
    )

    # The Nifty bar dated D itself: relative strength of D uses it (data <= D), the market
    # context of D does not (build_frame row D is built from data <= D-1).
    nifty = instruments.get_instrument(db, "NIFTY 50", "INDEX")
    assert nifty is not None
    db.execute(
        text(
            "UPDATE daily_bar SET close = close * 1.5, high = high * 1.5 "
            "WHERE instrument_id = :i AND trade_date = :d"
        ),
        {"i": nifty.id, "d": cut},
    )
    db.flush()
    nifty_shock = stock_features(db, ids, cut)
    at_cut = _rows_through(base, cut)
    assert nifty_shock.loc[("STK000", cut), "rs_20d"] != at_cut.loc[("STK000", cut), "rs_20d"]
    own = [c for c in STOCK_FEATURE_NAMES if not c.startswith("rs_")]
    pd.testing.assert_frame_equal(
        nifty_shock[[*META_COLUMNS, *own, *MARKET_CONTEXT_NAMES]],
        at_cut[[*META_COLUMNS, *own, *MARKET_CONTEXT_NAMES]],
    )
    before = _rows_through(nifty_shock, cut, inclusive=False)
    pd.testing.assert_frame_equal(before, _rows_through(base, cut, inclusive=False))


# ------------------------------------------------------------------------------ labels


def test_triple_barrier_rules_on_hand_built_paths() -> None:
    h = HorizonSpec("t", 5, 2.0, 1.0)  # ATR 2 -> target +4 %, stop -2 %
    d = 15  # signal position; entry = open of bar 16; window = bars 16..20

    def run(**mods: dict[int, float]) -> pd.Series:
        bars = _flat_bars(30)
        for col, changes in mods.items():
            for pos, value in changes.items():
                bars.iloc[pos, bars.columns.get_loc(col)] = value
        out = triple_barrier_labels(bars, [bars.index[d].date()], h)
        assert list(out.columns) == list(LABEL_COLUMNS)
        assert list(out.index) == [bars.index[d].date()]
        row = out.iloc[0]
        assert row["entry_session"] == bars.index[d + 1].date()
        assert row["entry_price"] == bars.iloc[d + 1]["open"]
        return row

    base = _flat_bars(30)
    dates = [t.date() for t in base.index]

    # Target touched on the third session of the window.
    row = run(high={18: 104.5})
    assert row["target_pct"] == pytest.approx(4.0) and row["stop_pct"] == pytest.approx(2.0)
    assert row["target_price"] == pytest.approx(104.0) and row["stop_price"] == pytest.approx(98.0)
    assert row["window_closed"] and row["label"] == 1.0 and not row["expired"]
    assert row["exit_session"] == dates[18] and row["exit_price"] == pytest.approx(104.0)
    assert row["realised_return_pct"] == pytest.approx(4.0) and row["days_held"] == 3
    # Stop touched on the second session; a later target touch is irrelevant.
    row = run(low={17: 97.0}, high={19: 110.0})
    assert row["label"] == 0.0 and not row["expired"] and row["exit_session"] == dates[17]
    assert row["exit_price"] == pytest.approx(98.0) and row["realised_return_pct"] == pytest.approx(
        -2.0
    )
    assert row["days_held"] == 2
    # Both barriers on the same day: the stop (conservative).
    row = run(low={18: 97.0}, high={18: 105.0})
    assert row["label"] == 0.0 and row["exit_price"] == pytest.approx(98.0)
    assert row["exit_session"] == dates[18]
    # Barrier touched on the entry session itself counts (the window includes it).
    row = run(high={16: 104.0})
    assert row["label"] == 1.0 and row["exit_session"] == dates[16] and row["days_held"] == 1
    # Gapped through the target: exit at the (better) open. Gapped below the stop: the open.
    row = run(open={18: 106.0}, high={18: 107.0})
    assert row["label"] == 1.0 and row["exit_price"] == pytest.approx(106.0)
    assert row["realised_return_pct"] == pytest.approx(6.0)
    row = run(open={17: 95.0}, low={17: 94.0}, high={17: 101.0})
    assert row["label"] == 0.0 and row["exit_price"] == pytest.approx(95.0)
    assert row["realised_return_pct"] == pytest.approx(-5.0)
    # Neither barrier: expired, label 0, realised return to the last close of the window.
    row = run(close={20: 101.0})
    assert row["label"] == 0.0 and row["expired"] and row["window_closed"]
    assert row["exit_session"] == dates[20] and row["exit_price"] == pytest.approx(101.0)
    assert row["realised_return_pct"] == pytest.approx(1.0) and row["days_held"] == 5
    # A touch just after the window is not seen.
    row = run(high={21: 120.0})
    assert row["label"] == 0.0 and row["expired"] and row["exit_session"] == dates[20]
    # Touch means >= / <=, exactly.
    row = run(high={19: 103.999})
    assert row["expired"]
    row = run(high={19: 104.0})
    assert row["label"] == 1.0 and row["exit_session"] == dates[19]

    # The signal's own bar is not part of the path: a huge range on D changes the barriers
    # (ATR of D) but never resolves the trade on D.
    bars = _flat_bars(30)
    bars.iloc[d, bars.columns.get_loc("high")] = 150.0
    bars.iloc[d, bars.columns.get_loc("low")] = 60.0
    out = triple_barrier_labels(bars, [dates[d]], h).iloc[0]
    assert out["entry_session"] == dates[d + 1] and out["exit_session"] > dates[d]
    assert out["target_pct"] > 4.0 and out["label"] == 0.0 and out["expired"]


def test_triple_barrier_unlabelled_cases() -> None:
    h = HorizonSpec("t", 5, 2.0, 1.0)
    bars = _flat_bars(30)
    dates = [t.date() for t in bars.index]
    # Window not closed (fewer than 5 bars after D): entry known, no label (purged).
    out = triple_barrier_labels(bars, [dates[25], dates[24], dates[29], dates[5]], h)
    row = out.iloc[0]
    assert not row["window_closed"] and math.isnan(row["label"]) and not row["expired"]
    assert row["entry_session"] == dates[26] and row["entry_price"] == 100.0
    assert row["exit_session"] is None and math.isnan(row["exit_price"])
    assert row["target_pct"] == pytest.approx(4.0)
    # Exactly 5 bars after D: labelled.
    assert out.iloc[1]["window_closed"] and out.iloc[1]["label"] == 0.0
    # Last bar: no entry yet (the live candidate).
    row = out.iloc[2]
    assert row["entry_session"] is None and math.isnan(row["entry_price"])
    assert not row["window_closed"] and math.isnan(row["label"])
    assert row["target_pct"] == pytest.approx(4.0)  # the barrier distance is known
    # No ATR yet (warm-up): nothing is labelled even though the window closed.
    row = out.iloc[3]
    assert math.isnan(row["target_pct"]) and math.isnan(row["label"]) and row["window_closed"]
    assert row["entry_session"] == dates[6]
    # A date that is not a bar (weekend), and timestamps as input.
    sat = date(2024, 6, 8)
    out = triple_barrier_labels(bars, [sat, pd.Timestamp(dates[15])], h)
    assert out.index[0] == sat and out.iloc[0]["entry_session"] is None
    assert out.iloc[0][["entry_price", "label", "target_pct"]].isna().all()
    assert not out.iloc[0]["window_closed"]
    assert out.index[1] == dates[15] and out.iloc[1]["window_closed"]
    # Duplicated signals and an empty signal list.
    out = triple_barrier_labels(bars, [dates[15], dates[15]], h)
    assert len(out) == 2 and (out["label"] == 0.0).all()
    assert triple_barrier_labels(bars, [], h).empty
    assert list(triple_barrier_labels(bars, [], h).columns) == list(LABEL_COLUMNS)
    # Empty bars: every signal unlabelled.
    out = triple_barrier_labels(pd.DataFrame(columns=["open", "high", "low", "close"]), [sat], h)
    assert math.isnan(out.iloc[0]["label"]) and out.iloc[0]["entry_session"] is None
    with pytest.raises(ValueError):
        triple_barrier_labels(bars, [dates[15]], HorizonSpec("bad", 0, 2.0, 1.0))
    # Dtypes: label float with NaN, flags bool.
    assert out["label"].dtype == float
    assert out["window_closed"].dtype == bool and out["expired"].dtype == bool


def test_label_uses_only_bars_after_the_signal(world: World) -> None:
    """The path starts at the next open and ends with the window; bars <= D only set the
    barrier distances (ATR % of close); bars beyond the window are never read."""
    db = world.db
    sym = "STK005"
    inst = world.inst[sym]
    sessions = [b.trade_date for b in world.bars[sym]]
    d = sessions[-40]
    window_end = sessions[sessions.index(d) + SWING.days]
    base = triple_barrier_labels(daily_frame(db, inst.id), sessions, SWING)
    labelled = base[base["label"].notna()]
    invalidated = base[base["invalidated"].astype(bool)]
    # ATR warm-up and the purged tail are unlabelled; so is every signal whose next open
    # fell outside the entry zone (the 09:20 rule would not have filled it). A no-fill is
    # known as soon as the next open is, window closed or not.
    closed_invalidated = invalidated[invalidated["window_closed"]]
    assert len(labelled) + len(closed_invalidated) == len(sessions) - 13 - SWING.days
    assert invalidated["label"].isna().all()
    assert set(labelled["label"]) <= {0.0, 1.0} and labelled["label"].mean() not in (0.0, 1.0)
    assert (labelled["exit_session"] > labelled.index).all()
    assert all(
        row["entry_session"] == sessions[sessions.index(sig) + 1]
        and row["entry_price"] == world.by_date(sym)[row["entry_session"]].open
        for sig, row in labelled.iterrows()
    )

    # (1) Everything dated after D's window is garbage: labels <= D are identical.
    db.execute(
        text(
            "UPDATE daily_bar SET open = open * 3, high = high * 5, low = low / 5, "
            "close = close * 3 WHERE instrument_id = :i AND trade_date > :d"
        ),
        {"i": inst.id, "d": window_end},
    )
    db.flush()
    after = triple_barrier_labels(daily_frame(db, inst.id), sessions, SWING)
    through_d = [s for s in sessions if s <= d]
    pd.testing.assert_frame_equal(after.loc[through_d], base.loc[through_d])
    # Sanity: the signal whose window is entirely garbage does change.
    assert after.at[window_end, "exit_price"] != base.at[window_end, "exit_price"]

    # (2) Doubling every price (D's bars and the path alike): ATR(D)/close(D), the fill
    # decision, the % distances, the label and the realised return are unchanged, and every
    # price column (entry, the barriers anchored at close(D), the exit) doubles with them.
    db.execute(
        text(
            "UPDATE daily_bar SET open = open * 2, high = high * 2, low = low * 2, "
            "close = close * 2 WHERE instrument_id = :i"
        ),
        {"i": inst.id},
    )
    db.flush()
    scaled = triple_barrier_labels(daily_frame(db, inst.id), [d], SWING).iloc[0]
    prices = {"entry_price", "target_price", "stop_price", "exit_price"}
    for col in LABEL_COLUMNS:
        a, b = scaled[col], base.at[d, col]
        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), col
        elif col in prices:
            assert a == pytest.approx(2.0 * b), col
        else:
            assert a == pytest.approx(b) if isinstance(a, float) else a == b, col


def test_purging_leaves_the_last_horizon_days_unlabelled(world: World) -> None:
    db = world.db
    symbols = ["STK006", "STK007", "STK008"]
    cands = pd.DataFrame(
        [(s, d, "breakout_20d", "breakout") for s in symbols for d in world.sessions],
        columns=["symbol", "session_date", "strategy", "setup"],
    )
    frame = build_candidate_frame(db, cands, SWING)
    assert len(frame) == len(cands)
    tail = set(world.sessions[-SWING.days :])
    for sym in symbols:
        sub = frame[frame["symbol"] == sym].set_index("session_date")
        unl = set(sub.index[sub["label"].isna()])
        inv = set(sub.index[sub["invalidated"].astype(bool)])
        assert unl == tail | set(world.sessions[:13]) | inv  # purged tail + ATR warm-up + no fill
        assert set(sub.index[~sub["window_closed"]]) == tail
        # The live candidate (last session) has no entry yet; the one before has an entry.
        assert sub.at[world.sessions[-1], "entry_session"] is None
        assert sub.at[world.sessions[-2], "entry_session"] == world.sessions[-1]
    # `through` moves the purge boundary with it (candidates after it have no bar: refused).
    through = world.sessions[-30]
    with pytest.raises(ValueError, match="without a bar"):
        build_candidate_frame(db, cands, SWING, through=through)
    earlier = build_candidate_frame(
        db, cands[cands["session_date"] <= through], SWING, through=through
    )
    sub = earlier[earlier["symbol"] == "STK006"].set_index("session_date")
    assert sub.index.max() == through
    idx = world.sessions.index(through)
    assert set(sub.index[~sub["window_closed"]]) == set(
        world.sessions[idx - SWING.days + 1 : idx + 1]
    )
    # Labels that were final stay final: rows labelled at `through` equal the full run.
    full = frame[frame["symbol"] == "STK006"].set_index("session_date")
    done = sub.index[sub["label"].notna()]
    pd.testing.assert_frame_equal(
        sub.loc[done, list(LABEL_COLUMNS)], full.loc[done, list(LABEL_COLUMNS)]
    )


def test_build_candidate_frame_shape_and_consistency(world: World) -> None:
    db = world.db
    s = world.sessions
    cands = pd.DataFrame(
        {
            "symbol": ["STK010", "STK010", "STK011", "stk012", "STK010"],
            "session_date": [s[-50], s[-50], s[-60], pd.Timestamp(s[-70]), s[-3]],
            "strategy": [
                "breakout_20d",
                "momentum_12_1",
                "pullback_20dma",
                "mean_reversion_rsi",
                "x",
            ],
            "setup": ["breakout", "momentum", "pullback", "mean_reversion", "x"],
        }
    )
    frame = build_candidate_frame(db, cands, SWING)
    assert len(frame) == 5 and isinstance(frame.index, pd.RangeIndex)
    assert list(frame.columns) == [
        "symbol",
        "session_date",
        "strategy",
        "setup",
        "horizon",
        "horizon_days",
        *META_COLUMNS,
        *FEATURE_NAMES,
        *LABEL_COLUMNS,
    ]
    assert list(frame["session_date"]) == sorted(frame["session_date"])
    assert list(frame["symbol"][:3]) == ["STK012", "STK011", "STK010"]
    assert (frame["horizon"] == "swing").all() and (frame["horizon_days"] == 10).all()
    assert list(frame.loc[frame["session_date"] == s[-50], "strategy"]) == [
        "breakout_20d",
        "momentum_12_1",
    ]
    # Features equal the feature frame; labels equal the labeller run directly.
    feats = stock_features(db, world.ids(["STK010", "STK011", "STK012"]), s[-1])
    for _, row in frame.iterrows():
        exp = feats.loc[(row["symbol"], row["session_date"])]
        np.testing.assert_array_equal(
            row[list(FEATURE_NAMES)].to_numpy(float), exp[list(FEATURE_NAMES)].to_numpy(float)
        )
        assert row["close"] == exp["close"] and row["eligible"] == exp["eligible"]
        lab = triple_barrier_labels(
            daily_frame(db, world.inst[row["symbol"]].id), [row["session_date"]], SWING
        ).iloc[0]
        for col in LABEL_COLUMNS:
            a, b = row[col], lab[col]
            assert (isinstance(a, float) and math.isnan(a) and math.isnan(b)) or a == b, col
    # The row at s[-3] is purged (window open), the others labelled.
    assert frame["label"].notna().sum() == 4 and math.isnan(frame.iloc[-1]["label"])
    # Index levels instead of columns; positional settings; empty input; unknown symbol;
    # signal without a bar.
    as_index = cands.set_index(["symbol", "session_date"])
    pd.testing.assert_frame_equal(build_candidate_frame(db, as_index, SWING), frame)
    pd.testing.assert_frame_equal(build_candidate_frame(db, cands, SWING, get_settings()), frame)
    empty = build_candidate_frame(db, cands.iloc[:0], SWING)
    assert empty.empty and list(empty.columns) == list(frame.columns)
    with pytest.raises(ValueError, match="unknown NSE symbols"):
        build_candidate_frame(db, cands.assign(symbol="NOPE"), SWING)
    with pytest.raises(ValueError, match="without a bar"):
        build_candidate_frame(db, cands.assign(session_date=date(2024, 3, 25)), SWING)  # Holi
    with pytest.raises(ValueError, match="lack columns"):
        build_candidate_frame(db, cands.drop(columns=["setup"]), SWING)
    # The positional horizon labels differently (wider barriers, longer window).
    pos = build_candidate_frame(db, cands, horizons()["positional"])
    assert (pos["horizon"] == "positional").all() and (pos["horizon_days"] == 60).all()
    assert pos["target_pct"].iloc[0] == pytest.approx(2 * frame["target_pct"].iloc[0])
    assert math.isnan(pos.loc[pos["session_date"] == s[-50], "label"].iloc[0])  # window open


# ----------------------------------------------------------------------- walk-forward


def test_walk_forward_folds_embargo(world: World) -> None:
    sessions = world.sessions
    n = len(sessions)
    folds = walk_forward_folds(sessions, min_train_sessions=100, step=21, embargo=SWING.days)
    assert folds and all(isinstance(f, Fold) and len(f) == 3 for f in folds)
    train_end, test_start, test_end = folds[0]
    assert train_end == sessions[99] and test_start == sessions[110] and test_end == sessions[130]
    assert folds[-1].test_end == sessions[-1]
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert cur.test_start == sessions[sessions.index(prev.test_end) + 1]
        assert cur.train_end > prev.train_end
    for f in folds:
        assert f.train_end < f.test_start <= f.test_end
        assert sessions.index(f.test_start) - sessions.index(f.train_end) - 1 == SWING.days
        assert sessions.index(f.test_end) - sessions.index(f.test_start) + 1 <= 21
    assert sum(sessions.index(f.test_end) - sessions.index(f.test_start) + 1 for f in folds) == (
        n - 110
    )

    # No training label window reaches a test signal: with every session a signal, the
    # latest exit of any training label is before the first test signal, for every fold.
    labels = triple_barrier_labels(daily_frame(world.db, world.inst["STK013"].id), sessions, SWING)
    exits = labels["exit_session"].dropna()
    for f in folds:
        train = exits[[d <= f.train_end for d in exits.index]]
        assert train.max() < f.test_start
        assert sessions.index(f.train_end) + SWING.days < sessions.index(f.test_start)
    # Without the embargo the proof fails: that is what it guards against.
    leaky = walk_forward_folds(sessions, min_train_sessions=100, step=21, embargo=0)
    f = leaky[0]
    assert exits[[d <= f.train_end for d in exits.index]].max() >= f.test_start

    # Input is deduplicated and sorted; too few sessions give no folds; bad args raise.
    shuffled = list(reversed(sessions)) + sessions[:5]
    assert walk_forward_folds(shuffled, min_train_sessions=100, step=21, embargo=10) == folds
    assert walk_forward_folds(sessions, 100, 21, 10) == folds  # positional form, per contract
    assert walk_forward_folds(sessions[:110], min_train_sessions=100, step=21, embargo=10) == []
    assert len(walk_forward_folds(sessions[:111], min_train_sessions=100, step=21, embargo=10)) == 1
    # Defaults: 250 training sessions, 21-session steps.
    default = walk_forward_folds(sessions, embargo=10)
    assert default[0] == (sessions[249], sessions[260], sessions[280])
    assert walk_forward_folds(sessions[:260], embargo=10) == []
    with pytest.raises(ValueError):
        walk_forward_folds(sessions, min_train_sessions=0, embargo=10)
    with pytest.raises(ValueError):
        walk_forward_folds(sessions, step=0, embargo=10)
    with pytest.raises(ValueError):
        walk_forward_folds(sessions, embargo=-1)
    with pytest.raises(ValueError, match="embargo"):
        walk_forward_folds(sessions)  # the embargo is never defaulted away


# ---------------------------------------------------------------------------- universe


def test_universe_at(world: World) -> None:
    db = world.db
    s = get_settings()
    assert s.ideas_universe_min_bars == 250 and s.min_avg_turnover_cr == 10.0
    members = universe_at(db, LAST)
    symbols = [i.symbol for i in members]
    assert symbols == sorted(symbols) == NORMAL
    assert all(isinstance(i, Instrument) and i.in_universe for i in members)
    # Each exclusion for its own reason.
    last = world.by_date("PENNY")[world.sessions[-1]]
    assert last.close < 20
    assert world.bars["NEWLIST"][0].trade_date == LISTED and len(world.bars["NEWLIST"]) < 250
    assert world.bars["HALTED"][-1].trade_date == HALT_LAST
    assert world.bars["THIN"][-1].close * 1500 / 1e7 < 10
    # Relax the bar count: NEWLIST (~147 bars) comes in; the others stay out for their reasons.
    relaxed = Settings(ideas_universe_min_bars=100)
    assert 100 <= len(world.bars["NEWLIST"]) < 250
    assert [i.symbol for i in universe_at(db, LAST, relaxed)] == sorted([*NORMAL, "NEWLIST"])
    # Earlier: point in time. Nobody has 250 bars after 200 sessions; with 100 required the
    # ordinary stocks and the (then trading) HALTED stock qualify, NEWLIST (40 bars) does not.
    d200 = world.sessions[199]
    assert universe_at(db, d200) == []
    assert [i.symbol for i in universe_at(db, d200, relaxed)] == sorted([*NORMAL, "HALTED"])
    # "A bar in the last five sessions": HALTED is in through the 5th session after its last
    # bar's session and out from the 6th.
    idx = world.sessions.index(HALT_LAST)
    assert "HALTED" in {i.symbol for i in universe_at(db, world.sessions[idx + 4], relaxed)}
    assert "HALTED" not in {i.symbol for i in universe_at(db, world.sessions[idx + 5], relaxed)}
    # A non-session date uses the last session before it; the universe is identical.
    saturday = date(2025, 3, 29)
    assert [i.symbol for i in universe_at(db, saturday)] == [
        i.symbol for i in universe_at(db, date(2025, 3, 28))
    ]
    # Turnover threshold bites: raising it to the largest stock's level empties the universe.
    frame = stock_features(db, world.ids(NORMAL), LAST)
    at_last = frame.xs(LAST, level="session_date")
    strict = Settings(min_avg_turnover_cr=float(at_last["turnover_20d_cr"].max()) + 1.0)
    assert universe_at(db, LAST, strict) == []
    loose = Settings(min_avg_turnover_cr=float(at_last["turnover_20d_cr"].min()))
    assert [i.symbol for i in universe_at(db, LAST, loose)] == NORMAL
    # universe_at(d) is the `eligible` column of the feature frame at d, for members.
    all_frame = stock_features(db, world.universe_ids(), LAST)
    eligible = set(all_frame.xs(LAST, level="session_date").query("eligible").index)
    assert eligible == set(symbols)
    d = world.sessions[-20]
    eligible_d = set(all_frame.xs(d, level="session_date").query("eligible").index)
    assert eligible_d == {i.symbol for i in universe_at(db, d)}
    # Weekend (Muhurat-style) bars count for nothing: not bars, not turnover.
    thin = world.inst["THIN"]
    sunday_bars = [Bar(date(2025, 3, 30), 900.0, 950.0, 880.0, 900.0, 5e7)]
    upsert_daily_bars(db, thin.id, sunday_bars, "yfinance")
    db.flush()
    assert "THIN" not in {i.symbol for i in universe_at(db, LAST)}
    # Volume reported as 0 on one of the 20 sessions: turnover unknown, stock out (no default).
    db.execute(
        text("UPDATE daily_bar SET volume = 0 WHERE instrument_id = :i AND trade_date = :d"),
        {"i": world.inst["STK000"].id, "d": world.sessions[-10]},
    )
    db.flush()
    assert "STK000" not in {i.symbol for i in universe_at(db, LAST)}
    assert "STK000" in {i.symbol for i in universe_at(db, world.sessions[-11])}
    # Nobody in the universe table: nothing.
    db.execute(text("UPDATE instrument SET in_universe = false"))
    db.flush()
    assert universe_at(db, LAST) == []

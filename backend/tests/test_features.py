"""Feature pipeline: no lookahead, label semantics, session filtering, live snapshot.

The leakage tests are the heart of this file. A feature of session D that peeks at any
bar or flow dated on or after D would silently inflate every metric in the product, so
they are proven against the database itself: (1) rebuilding the frame after deleting
everything dated >= D leaves row D identical; (2) corrupting everything dated >= D
leaves every row <= D identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml import events, features
from app.ml.core import class_of, finite, pct_change
from app.ml.features import FEATURE_LABELS, FEATURE_NAMES, build_frame, frame_to_matrix, live_row
from app.models import Instrument
from app.providers.base import Bar, FlowRecord, HolidayRecord
from app.services import flows, instruments
from app.services.calendar import TradingCalendar, store_holidays
from app.services.market_data import upsert_daily_bars

TARGET = "NIFTY 50"
FIRST, LAST = date(2024, 1, 1), date(2025, 4, 30)
# Known holidays (stored in the calendar): no bars on these days.
HOLIDAYS = {date(2024, 3, 25): "Holi", date(2024, 8, 15): "Independence Day"}
# A weekday with no bar and no holiday row (calendar gap): still a session row, no label.
GAP = date(2025, 2, 26)
# Flows missing for this session: the next session's fii/dii features must be NaN.
NO_FLOW_DAY = date(2024, 10, 9)
US_CLOSED = {date(2024, 7, 4), date(2024, 11, 28)}
NIKKEI_CLOSED = {date(2024, 5, 3), date(2024, 5, 6)}
N_UNIVERSE = 25


def _weekdays(first: date, last: date) -> list[date]:
    return [
        first + timedelta(i)
        for i in range((last - first).days + 1)
        if (first + timedelta(i)).weekday() < 5
    ]


def _walk(rng: np.random.Generator, n: int, start: float = 100.0, vol: float = 0.012) -> np.ndarray:
    return start * np.exp(np.cumsum(rng.normal(0.0, vol, n)))


def _bars(rng: np.random.Generator, days: list[date], volume: bool = True) -> list[Bar]:
    close = _walk(rng, len(days))
    out: list[Bar] = []
    for i, (d, c) in enumerate(zip(days, close, strict=True)):
        prev = close[i - 1] if i else c
        o = prev * (1 + rng.normal(0, 0.004))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
        vol = float(rng.integers(100_000, 400_000)) if volume else 0.0
        out.append(Bar(d, float(o), float(hi), float(lo), float(c), vol))
    return out


@dataclass
class World:
    db: Session
    sessions: list[date]  # trading sessions with a target bar
    target_close: dict[date, float]
    cue_close: dict[str, dict[date, float]]
    flow_net: dict[date, tuple[float, float]]
    cal: TradingCalendar

    def session_after(self, d: date) -> date:
        return self.cal.next_trading_day(d)

    @property
    def live_session(self) -> date:
        return self.cal.next_trading_day(self.sessions[-1])


@pytest.fixture()
def world(db: Session) -> World:
    """A synthetic 16-month market: target, VIX, global cues on their own calendars,
    FII/DII flows, a 25-stock universe, two holidays and one calendar gap."""
    rng = np.random.default_rng(2024)
    store_holidays(db, [HolidayRecord(d, desc) for d, desc in HOLIDAYS.items()], "test")
    instruments.ensure_reference_instruments(db)
    db.flush()
    cal = TradingCalendar.load(db)
    weekdays = _weekdays(FIRST, LAST)
    sessions = [d for d in weekdays if d not in HOLIDAYS and d != GAP]

    target = instruments.get_instrument(db, TARGET, "INDEX")
    assert target is not None
    target_bars = _bars(rng, sessions)
    upsert_daily_bars(db, target.id, target_bars, "yfinance")
    vix = instruments.get_instrument(db, "INDIA VIX", "INDEX")
    assert vix is not None
    upsert_daily_bars(db, vix.id, _bars(rng, sessions), "yfinance")

    cue_close: dict[str, dict[date, float]] = {}
    for ref in instruments.GLOBAL_CUES:
        days = weekdays  # global markets trade on Indian holidays
        if ref.group in {"US", "US futures", "Rates", "FX"}:
            days = [d for d in days if d not in US_CLOSED]
        if ref.key == "nikkei":
            days = [d for d in days if d not in NIKKEI_CLOSED]
        inst = instruments.get_instrument(db, ref.symbol, "GLOBAL")
        assert inst is not None
        bars = _bars(rng, days)
        upsert_daily_bars(db, inst.id, bars, "yfinance")
        cue_close[ref.key] = {b.trade_date: b.close for b in bars}

    flow_net: dict[date, tuple[float, float]] = {}
    records: list[FlowRecord] = []
    for d in sessions:
        if d == NO_FLOW_DAY:
            continue
        fii, dii = float(rng.normal(0, 2000)), float(rng.normal(500, 1500))
        flow_net[d] = (fii, dii)
        records.append(FlowRecord(d, "FII", abs(fii) + 100, 100.0, fii))
        records.append(FlowRecord(d, "DII", abs(dii) + 100, 100.0, dii))
    flows.store(db, records, "nse")

    for i in range(N_UNIVERSE):
        eq = instruments.get_or_create_equity(db, f"STK{i:03d}", "NSE")
        eq.in_universe = True
        upsert_daily_bars(db, eq.id, _bars(rng, sessions), "nse_bhavcopy")
    db.flush()
    return World(
        db, sessions, {b.trade_date: b.close for b in target_bars}, cue_close, flow_net, cal
    )


def _rows_through(frame: pd.DataFrame, d: date, *, inclusive: bool = True) -> pd.DataFrame:
    mask = np.array([(x <= d) if inclusive else (x < d) for x in frame.index], dtype=bool)
    return frame[mask]


def _assert_same_features(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    assert list(a.index) == list(b.index), what
    np.testing.assert_array_equal(frame_to_matrix(a), frame_to_matrix(b), err_msg=what)


def _assert_same_labels(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    np.testing.assert_array_equal(
        a["label_pct"].to_numpy(float), b["label_pct"].to_numpy(float), err_msg=what
    )
    assert list(a["label_class"]) == list(b["label_class"]), what
    assert list(a["reference_date"]) == list(b["reference_date"]), what
    np.testing.assert_array_equal(
        a["reference_close"].to_numpy(float), b["reference_close"].to_numpy(float), err_msg=what
    )


def _cuts(world: World) -> list[date]:
    """Sessions to cut at: a mid-week day, a Monday, the day after a holiday, the gap."""
    sessions = world.sessions
    monday = next(d for d in sessions if d.weekday() == 0 and d > date(2024, 11, 1))
    midweek = next(d for d in sessions if d.weekday() == 2 and d > date(2024, 12, 1))
    return sorted({midweek, monday, world.session_after(date(2024, 8, 15)), GAP})


# ------------------------------------------------------------------ lookahead leakage


def test_truncation_invariance(world: World) -> None:
    """Row D built from a database that ends at D-1 equals row D of the full frame."""
    db = world.db
    full = build_frame(db, TARGET)
    assert full.index[-1] == world.live_session
    for cut in sorted(_cuts(world), reverse=True):  # later cuts first: deletes accumulate
        # The `end` argument alone must not matter: data after `end` is never used.
        by_end = build_frame(db, TARGET, end=cut)
        _assert_same_features(by_end, _rows_through(full, cut), f"end={cut}")
        _assert_same_labels(by_end, _rows_through(full, cut), f"end={cut}")

        # Now the real test: the data dated >= cut does not exist at all.
        db.execute(text("DELETE FROM daily_bar WHERE trade_date >= :d"), {"d": cut})
        db.execute(text("DELETE FROM fii_dii_flow WHERE trade_date >= :d"), {"d": cut})
        db.flush()
        trunc = build_frame(db, TARGET, end=cut)
        assert trunc.index[-1] == cut
        _assert_same_features(trunc, _rows_through(full, cut), f"truncated at {cut}")
        _assert_same_labels(
            _rows_through(trunc, cut, inclusive=False),
            _rows_through(full, cut, inclusive=False),
            f"labels before {cut}",
        )
        # Session `cut` has no close yet: label missing, features present.
        assert math.isnan(trunc.at[cut, "label_pct"])
        assert trunc.at[cut, "label_class"] is None
        assert np.isfinite(frame_to_matrix(trunc.loc[[cut]])).sum() >= 40

        # Without an explicit end, the default end is the session after the last bar.
        assert build_frame(db, TARGET).index[-1] == cut


def test_future_data_does_not_change_past_rows(world: World) -> None:
    """Corrupting every bar and flow dated >= D leaves every row <= D untouched."""
    db = world.db
    base = build_frame(db, TARGET)
    cut = _cuts(world)[1]
    db.execute(
        text(
            "UPDATE daily_bar SET open = open * 3, high = high * 3, low = low * 3, "
            "close = close * 3, volume = COALESCE(volume, 0) * 7 + 1 WHERE trade_date >= :d"
        ),
        {"d": cut},
    )
    db.execute(
        text(
            "UPDATE fii_dii_flow SET net_value_cr = COALESCE(net_value_cr, 0) + 12345 "
            "WHERE trade_date >= :d"
        ),
        {"d": cut},
    )
    # A bar dated exactly D for a global cue (today's partial print) and a flow dated D.
    sp = instruments.get_instrument(db, "^GSPC", "GLOBAL")
    assert sp is not None
    upsert_daily_bars(db, sp.id, [Bar(cut, 1.0, 2.0, 0.5, 1.0, 1.0)], "yfinance")
    flows.store(db, [FlowRecord(cut, "FII", 1.0, 1.0, 99999.0)], "nse")
    db.flush()

    shocked = build_frame(db, TARGET)
    assert list(shocked.index) == list(base.index)
    _assert_same_features(_rows_through(shocked, cut), _rows_through(base, cut), "rows <= D")
    _assert_same_labels(
        _rows_through(shocked, cut, inclusive=False),
        _rows_through(base, cut, inclusive=False),
        "labels < D",
    )
    # Sanity: the corruption is real and visible where it should be.
    assert shocked.at[cut, "label_pct"] != base.at[cut, "label_pct"]
    nxt = world.session_after(cut)
    assert shocked.at[nxt, "ret_1d"] != base.at[nxt, "ret_1d"]
    assert shocked.at[nxt, "fii_net_cr"] != base.at[nxt, "fii_net_cr"]


def test_global_cues_use_last_bar_before_session_on_their_own_calendar(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    sp = world.cue_close["sp500"]

    def ret(d: date, prev: date) -> float:
        return pct_change(sp[d], sp[prev])

    # Monday: Friday's S&P move, not anything dated on the weekend or Monday.
    monday = date(2024, 9, 9)
    assert frame.at[monday, "sp500_ret_1d"] == pytest.approx(
        ret(date(2024, 9, 6), date(2024, 9, 5))
    )
    # Indian session after a US holiday (4 Jul 2024): the last US bar is 3 Jul.
    assert frame.at[date(2024, 7, 5), "sp500_ret_1d"] == pytest.approx(
        ret(date(2024, 7, 3), date(2024, 7, 2))
    )
    # Indian holiday with the US open: Friday 16 Aug sees Thursday's US close.
    assert frame.at[date(2024, 8, 16), "sp500_ret_1d"] == pytest.approx(
        ret(date(2024, 8, 15), date(2024, 8, 14))
    )
    # Nikkei closed 3 and 6 May: 7 May uses the 2 May bar (5 days old, within the cap).
    nk = world.cue_close["nikkei"]
    assert frame.at[date(2024, 5, 7), "nikkei_ret_1d"] == pytest.approx(
        pct_change(nk[date(2024, 5, 2)], nk[date(2024, 5, 1)])
    )
    # us10y is a level change, not a return.
    tnx = world.cue_close["us10y"]
    assert frame.at[monday, "us10y_chg_1d"] == pytest.approx(
        tnx[date(2024, 9, 6)] - tnx[date(2024, 9, 5)]
    )


def test_stale_cue_is_missing_not_reused(world: World) -> None:
    db = world.db
    dxy = instruments.get_instrument(db, "DX-Y.NYB", "GLOBAL")
    assert dxy is not None
    db.execute(
        text("DELETE FROM daily_bar WHERE instrument_id = :i AND trade_date > :d"),
        {"i": dxy.id, "d": LAST - timedelta(days=20)},
    )
    db.flush()
    frame = build_frame(db, TARGET)
    assert math.isnan(frame.at[world.live_session, "dxy_ret_1d"])
    assert not math.isnan(frame.at[world.live_session, "sp500_ret_1d"])
    snap, _, _ = live_row(db, TARGET, world.live_session)
    assert snap["dxy_ret_1d"] is None


# ------------------------------------------------------------------- labels and sessions


def test_label_uses_close_of_d_and_previous_bar_close(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    close = world.target_close
    thr = get_settings().market_flat_threshold_pct

    # Ordinary day.
    d = date(2024, 6, 12)
    prev = date(2024, 6, 11)
    assert frame.at[d, "reference_date"] == prev
    assert frame.at[d, "reference_close"] == close[prev]
    assert frame.at[d, "label_pct"] == pytest.approx(pct_change(close[d], close[prev]))
    # Day after a holiday: the reference is the last session with a bar.
    after = date(2024, 8, 16)
    assert date(2024, 8, 15) not in frame.index
    assert frame.at[after, "reference_date"] == date(2024, 8, 14)
    assert frame.at[after, "label_pct"] == pytest.approx(
        pct_change(close[after], close[date(2024, 8, 14)])
    )
    # Monday references Friday.
    assert frame.at[date(2024, 9, 9), "reference_date"] == date(2024, 9, 6)
    # Every labelled row: class follows the configured threshold; the live row has none.
    labelled = frame[frame["label_pct"].notna()]
    assert len(labelled) == len(world.sessions) - 1
    assert all(
        c == class_of(float(p), thr)
        for p, c in zip(labelled["label_pct"], labelled["label_class"], strict=True)
    )
    live = frame.index[-1]
    assert live == world.live_session
    assert math.isnan(frame.at[live, "label_pct"]) and frame.at[live, "label_class"] is None
    assert frame.at[live, "reference_date"] == world.sessions[-1]
    # ret_1d of D is exactly yesterday's label (the same_as_yesterday baseline relies on it).
    assert frame.at[after, "ret_1d"] == pytest.approx(frame.at[date(2024, 8, 14), "label_pct"])


def test_calendar_gap_is_a_session_without_a_label(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    assert GAP in frame.index
    assert math.isnan(frame.at[GAP, "label_pct"]) and frame.at[GAP, "label_class"] is None
    assert frame.at[GAP, "reference_date"] == GAP - timedelta(days=1)
    nxt = world.session_after(GAP)
    assert frame.at[nxt, "reference_date"] == GAP - timedelta(days=1)
    assert frame.at[nxt, "label_pct"] == pytest.approx(
        pct_change(world.target_close[nxt], world.target_close[GAP - timedelta(days=1)])
    )
    assert frame.index.is_monotonic_increasing and frame.index.is_unique


def test_weekend_and_holiday_bars_are_dropped(world: World) -> None:
    db = world.db
    base = build_frame(db, TARGET)
    wild = [
        Bar(date(2024, 11, 3), 9e5, 9e6, 9e4, 9e5, 1.0),  # Sunday (Muhurat-style)
        Bar(date(2025, 2, 1), 9e5, 9e6, 9e4, 9e5, 1.0),  # Saturday budget session
        Bar(date(2024, 8, 15), 9e5, 9e6, 9e4, 9e5, 1.0),  # known holiday
    ]
    for symbol, exchange in ((TARGET, "INDEX"), ("INDIA VIX", "INDEX"), ("STK001", "NSE")):
        inst = instruments.get_instrument(db, symbol, exchange)
        assert inst is not None
        upsert_daily_bars(db, inst.id, wild, "yfinance")
    db.flush()
    after = build_frame(db, TARGET)
    assert all(d.weekday() < 5 and d not in HOLIDAYS for d in after.index)
    _assert_same_features(after, base, "weekend/holiday bars must not touch any feature")
    _assert_same_labels(after, base, "weekend/holiday bars must not touch any label")


def test_start_and_end_select_sessions(world: World) -> None:
    full = build_frame(world.db, TARGET)
    part = build_frame(world.db, TARGET, start=date(2024, 10, 1), end=date(2024, 10, 31))
    assert part.index[0] == date(2024, 10, 1) and part.index[-1] == date(2024, 10, 31)
    _assert_same_features(part, full.loc[list(part.index)], "slice equals the full frame")
    assert build_frame(world.db, "NIFTY BANK").empty  # instrument exists, no bars
    assert build_frame(world.db, "NO SUCH INDEX").empty
    assert list(build_frame(world.db, "NO SUCH INDEX").columns) == [
        *FEATURE_NAMES,
        *features.LABEL_COLUMNS,
    ]


# ----------------------------------------------------------------- flows, breadth, volume


def test_flows_are_exactly_the_reference_sessions_numbers(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    d = date(2024, 6, 12)
    fii, dii = world.flow_net[date(2024, 6, 11)]
    assert frame.at[d, "fii_net_cr"] == pytest.approx(fii)
    assert frame.at[d, "dii_net_cr"] == pytest.approx(dii)
    last5 = [
        date(2024, 6, 5),
        date(2024, 6, 6),
        date(2024, 6, 7),
        date(2024, 6, 10),
        date(2024, 6, 11),
    ]
    assert frame.at[d, "fii_net_5d_cr"] == pytest.approx(sum(world.flow_net[x][0] for x in last5))
    # Missing flows for the reference session: NaN, never the day before's number.
    nxt = world.session_after(NO_FLOW_DAY)
    assert math.isnan(frame.at[nxt, "fii_net_cr"]) and math.isnan(frame.at[nxt, "dii_net_cr"])
    assert math.isnan(frame.at[nxt, "fii_net_5d_cr"])
    # The 5-session sum needs all five sessions.
    five_later = frame.index[list(frame.index).index(nxt) + 4]
    six_later = frame.index[list(frame.index).index(nxt) + 5]
    assert math.isnan(frame.at[five_later, "fii_net_5d_cr"])
    assert not math.isnan(frame.at[six_later, "fii_net_5d_cr"])


def test_breadth_needs_twenty_instruments(world: World) -> None:
    db = world.db
    frame = build_frame(db, TARGET)
    tail = frame.iloc[-20:]
    assert tail["pct_above_200dma"].between(0, 100).all()
    assert (tail["adv_decl_ratio"] > 0).all()
    # Warm-up: the first rows have no 200-DMA for anyone.
    assert math.isnan(frame.iloc[5]["pct_above_200dma"])
    # Hand-check one day: breadth of D comes from the universe's bars on the reference date.
    d = frame.index[-1]
    ref = frame.at[d, "reference_date"]
    rows = db.execute(
        text(
            "SELECT b.close, p.close FROM daily_bar b JOIN instrument i ON i.id = b.instrument_id "
            "JOIN daily_bar p ON p.instrument_id = b.instrument_id AND p.trade_date = :prev "
            "WHERE i.in_universe AND b.trade_date = :ref"
        ),
        {"ref": ref, "prev": world.sessions[-2]},
    ).all()
    adv = sum(1 for c, p in rows if c > p)
    dec = sum(1 for c, p in rows if c < p)
    assert frame.at[d, "adv_decl_ratio"] == pytest.approx(adv / dec)

    db.execute(
        update(Instrument)
        .where(Instrument.symbol.in_([f"STK{i:03d}" for i in range(6)]))
        .values(in_universe=False)
    )
    db.flush()
    thin = build_frame(db, TARGET)
    assert thin["pct_above_200dma"].isna().all() and thin["adv_decl_ratio"].isna().all()
    other = thin.drop(columns=["pct_above_200dma", "adv_decl_ratio"])
    _assert_same_features(
        other.reindex(columns=FEATURE_NAMES),
        frame.assign(pct_above_200dma=np.nan, adv_decl_ratio=np.nan),
        "only breadth changes",
    )


def test_unreported_volume_gives_nan_not_zero(world: World) -> None:
    db = world.db
    frame = build_frame(db, TARGET)
    assert frame["vol_z20"].iloc[-10:].notna().all()
    bank = instruments.get_instrument(db, "NIFTY BANK", "INDEX")
    assert bank is not None
    upsert_daily_bars(
        db, bank.id, _bars(np.random.default_rng(5), world.sessions, volume=False), "yfinance"
    )
    db.flush()
    bank_frame = build_frame(db, "NIFTY BANK")
    assert bank_frame["vol_z20"].isna().all()
    assert bank_frame["ret_1d"].iloc[-10:].notna().all()


# --------------------------------------------------------------- calendar and event flags


def test_calendar_features(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    for d in frame.index:
        assert frame.at[d, "dow"] == d.weekday()
    assert frame.at[date(2024, 8, 30), "sessions_to_month_end"] == 0  # last Friday of August
    assert frame.at[date(2024, 8, 29), "sessions_to_month_end"] == 1
    assert frame.at[date(2024, 8, 9), "sessions_to_month_end"] == 14  # 15 Aug is a holiday
    assert frame.at[date(2024, 8, 29), "is_month_last_week"] == 1.0
    assert frame.at[date(2024, 8, 23), "is_month_last_week"] == 0.0  # 5 sessions left
    expiry_wd = get_settings().nifty_expiry_weekday
    assert all(frame.at[d, "is_expiry_day"] == float(d.weekday() == expiry_wd) for d in frame.index)
    # Budget days are known without events.csv; everything else is unknown (0, not loaded).
    assert frame.at[date(2024, 2, 1), "budget_day"] == 1.0
    assert frame.at[date(2024, 7, 23), "budget_day"] == 1.0
    assert frame.at[date(2025, 2, 3), "budget_day"] == 1.0  # Saturday budget -> Monday
    assert frame["budget_day"].sum() == 3.0
    assert (frame["events_loaded"] == 0.0).all()
    for col in ("rbi_policy_day", "fomc_overnight", "us_cpi_overnight"):
        assert (frame[col] == 0.0).all()


def test_event_flags_without_events_csv(tmp_path: Path) -> None:
    s = Settings(imports_dir=str(tmp_path))
    flags = events.event_flags(date(2024, 2, 1), s)
    assert list(flags) == list(events.EVENT_KEYS)
    assert flags == {
        "budget_day": 1.0,
        "rbi_policy_day": 0.0,
        "fomc_overnight": 0.0,
        "us_cpi_overnight": 0.0,
        "events_loaded": 0.0,
    }
    assert events.event_flags(date(2024, 2, 2), s)["budget_day"] == 0.0
    assert events.event_flags(date(2024, 7, 23), s)["budget_day"] == 1.0
    # Weekend budgets (2025, 2026) flag the first session that can react.
    assert events.event_flags(date(2025, 2, 3), s)["budget_day"] == 1.0
    assert events.event_flags(date(2025, 2, 4), s)["budget_day"] == 0.0
    cal = events.load_events(s)
    assert not cal.loaded and cal.of("rbi_policy") == frozenset() and cal.of("fomc") == frozenset()


def test_event_flags_from_events_csv(tmp_path: Path) -> None:
    (tmp_path / "events.csv").write_text(
        "date,kind\n"
        "2024-06-12,fomc\n"
        "2024-06-12,us_cpi\n"
        "12-Jul-2024,us_cpi\n"
        "2024-06-07,rbi_policy\n"
        "2024-07-23,budget\n"
        "2024-06-20,quad_witching\n"
        "not-a-date,fomc\n",
        encoding="utf-8",
    )
    s = Settings(imports_dir=str(tmp_path))
    cal = events.load_events(s)
    assert cal.loaded and cal.of("fomc") == frozenset({date(2024, 6, 12)})
    assert date(2024, 2, 1) in cal.of("budget") and date(2024, 7, 23) in cal.of("budget")

    f = events.event_flags(date(2024, 6, 13), s)  # Thursday after a Wednesday decision
    assert f["fomc_overnight"] == 1.0 and f["us_cpi_overnight"] == 1.0 and f["events_loaded"] == 1.0
    assert events.event_flags(date(2024, 6, 12), s)["fomc_overnight"] == 0.0  # after the close
    assert events.event_flags(date(2024, 6, 14), s)["fomc_overnight"] == 0.0
    # Friday CPI lands on Monday.
    assert events.event_flags(date(2024, 7, 15), s)["us_cpi_overnight"] == 1.0
    assert events.event_flags(date(2024, 7, 16), s)["us_cpi_overnight"] == 0.0
    # Indian-hours event: same day only.
    assert events.event_flags(date(2024, 6, 7), s)["rbi_policy_day"] == 1.0
    assert events.event_flags(date(2024, 6, 10), s)["rbi_policy_day"] == 0.0
    # With the real previous session (Thursday 13 June a holiday), Friday still sees it.
    f = events.event_flags(date(2024, 6, 14), s, previous_session=date(2024, 6, 12))
    assert f["fomc_overnight"] == 1.0
    with pytest.raises(ValueError):
        events.event_flags(date(2024, 6, 14), s, previous_session=date(2024, 6, 14))


# ----------------------------------------------------------------- live row and matrix


def test_live_row_is_finite_and_ordered(world: World) -> None:
    db = world.db
    snapshot, ref_date, ref_close = live_row(db, TARGET, world.live_session)
    assert list(snapshot) == FEATURE_NAMES
    for name, value in snapshot.items():
        assert value is None or (isinstance(value, float) and math.isfinite(value)), name
    assert ref_date == world.sessions[-1]
    assert ref_close == world.target_close[world.sessions[-1]]
    frame = build_frame(db, TARGET)
    assert snapshot == {n: finite(frame.at[world.live_session, n]) for n in FEATURE_NAMES}
    for name in (
        "ret_1d",
        "rsi14",
        "dist_sma200_pct",
        "vix_level",
        "sp500_ret_1d",
        "fii_net_cr",
        "pct_above_200dma",
    ):
        assert snapshot[name] is not None, name
    # The snapshot is JSON-ready: every value is a rounded float or None.
    assert all(v is None or v == round(v, 10) for v in snapshot.values())

    with pytest.raises(LookupError):
        live_row(db, "NIFTY BANK", world.live_session)  # no bars
    with pytest.raises(LookupError):
        live_row(db, TARGET, date(2025, 5, 3))  # a Saturday
    # A past session works through the same path and reports its own reference.
    snap2, ref2, _ = live_row(db, TARGET, date(2024, 8, 16))
    assert ref2 == date(2024, 8, 14) and list(snap2) == FEATURE_NAMES


def test_frame_to_matrix_order_and_nan(world: World) -> None:
    frame = build_frame(world.db, TARGET)
    m = frame_to_matrix(frame)
    assert m.shape == (len(frame), len(FEATURE_NAMES)) and m.dtype == float
    for j, name in enumerate(FEATURE_NAMES):
        np.testing.assert_array_equal(m[:, j], frame[name].to_numpy(float))
    assert np.isnan(m[0, FEATURE_NAMES.index("ret_60d")])  # warm-up stays NaN
    # Shuffled columns and a missing column: order restored, missing -> NaN.
    shuffled = frame[list(reversed(FEATURE_NAMES))].drop(columns=["bb_z"])
    m2 = frame_to_matrix(shuffled)
    assert np.isnan(m2[:, FEATURE_NAMES.index("bb_z")]).all()
    np.testing.assert_array_equal(m2[:, 0], frame["ret_1d"].to_numpy(float))
    assert frame_to_matrix(features._empty_frame()).shape == (0, len(FEATURE_NAMES))


def test_feature_names_are_a_fixed_labelled_vocabulary() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert set(FEATURE_NAMES) == set(FEATURE_LABELS)
    assert not set(FEATURE_NAMES) & set(features.LABEL_COLUMNS)
    assert all(n.replace("_", "").isalnum() and n == n.lower() for n in FEATURE_NAMES)
    assert all(label and len(label) <= 60 for label in FEATURE_LABELS.values())
    assert FEATURE_NAMES[-5:] == list(events.EVENT_KEYS)

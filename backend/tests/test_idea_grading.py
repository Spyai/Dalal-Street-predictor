"""Idea grading: the entry rule, the exit walk (target / stop / expiry / same-day tie / gaps),
costs and R-multiples, idempotent upserts, voiding."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.core import CostModel
from app.models import IdeaOutcome
from app.services import idea_grading, ledger
from app.services.idea_grading import (
    SessionBar,
    fill_decision,
    grade_idea,
    open_check_fill,
    session_bars,
    void_idea,
    walk_exit,
)
from app.services.ledger import record_idea
from tests.test_idea_ledger import SESSION, make_idea, make_instrument

OPEN_NOW = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)  # 09:20 IST on the entry session
NOW = datetime(2026, 11, 2, 10, 16, tzinfo=UTC)  # 15:46 IST
COST = CostModel.from_settings().round_trip_pct()
# The test idea (tests.test_idea_ledger.make_idea): zone 990..1010, stop 980, target 1040.
STOP_PCT = 2.0  # risk from an entry at 1000 to the stop at 980


def bars(*rows: tuple[int, float, float, float, float]) -> pd.DataFrame:
    """A daily frame like `market_data.daily_frame`: (day offset, open, high, low, close)."""
    idx = pd.to_datetime([SESSION + timedelta(days=d) for d, *_ in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0e6] * len(rows),
            "delivery_pct": [math.nan] * len(rows),
            "source": ["test"] * len(rows),
        },
        index=idx,
    )


def quiet(days: int, start_day: int = 0) -> list[tuple[int, float, float, float, float]]:
    """`days` sessions drifting gently upward inside the zone: no target, no stop."""
    return [(start_day + d, 1000.0 + d, 1006.0 + d, 996.0 + d, 1002.0 + d) for d in range(days)]


def _rows(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(IdeaOutcome)) or 0)


@pytest.fixture()
def idea(db: Session):  # type: ignore[no-untyped-def]
    inst = make_instrument(db)
    row = record_idea(db, make_idea(inst.id))
    db.commit()
    return row


# ------------------------------------------------------------------ the entry rule


@pytest.mark.parametrize("open_", [990.0, 1000.0, 1010.0])
def test_open_inside_the_zone_fills_at_the_open(db: Session, idea, open_: float) -> None:  # type: ignore[no-untyped-def]
    o = open_check_fill(db, idea, open_, now=OPEN_NOW, source="kite")
    assert o.status == "open" and o.entry_price == open_ and o.entry_date == SESSION
    assert o.exit_price is None and o.net_return_pct is None and o.days_held is None
    assert o.graded_at == OPEN_NOW and o.source == "kite" and o.notes is None
    assert [i.id for i in ledger.open_ideas(db)] == [idea.id]


@pytest.mark.parametrize(
    ("open_", "side"), [(1010.01, "above"), (989.99, "below"), (1200.0, "above")]
)
def test_open_outside_the_zone_in_either_direction_invalidates(
    db: Session,
    idea,
    open_: float,
    side: str,  # type: ignore[no-untyped-def]
) -> None:
    o = open_check_fill(db, idea, open_, now=OPEN_NOW, source="kite")
    assert o.status == "invalidated" and o.entry_price is None and o.entry_date is None
    assert o.notes is not None and f"gapped {side} the entry zone [990.00, 1010.00]" in o.notes
    assert ledger.open_ideas(db) == [], "an invalidated idea is not waited on"
    with pytest.raises(ValueError, match="invalidated: nothing to grade"):
        grade_idea(db, idea, bars((0, open_, open_ + 50, open_ - 50, open_)), now=NOW, source="t")


def test_fill_is_idempotent_and_never_resets_a_graded_idea(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    first = open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    again = open_check_fill(db, idea, 1000.0, now=OPEN_NOW + timedelta(minutes=1), source="yf")
    assert again.id == first.id and again.graded_at == OPEN_NOW and again.source == "kite"
    assert _rows(db) == 1
    corrected = open_check_fill(db, idea, 1001.0, now=OPEN_NOW + timedelta(minutes=2), source="nse")
    assert (
        corrected.id == first.id and corrected.entry_price == 1001.0 and corrected.source == "nse"
    )

    graded = grade_idea(db, idea, bars((0, 1001.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    assert graded.status == "target"
    untouched = open_check_fill(db, idea, 1001.0, now=NOW + timedelta(days=1), source="kite")
    assert untouched.status == "target" and untouched.exit_price == 1040.0
    assert _rows(db) == 1


def test_fill_refuses_a_non_finite_open(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="session open"):
        open_check_fill(db, idea, math.nan, now=OPEN_NOW, source="kite")
    with pytest.raises(ValueError, match="session open"):
        fill_decision(idea, 0.0)
    assert _rows(db) == 0


# ------------------------------------------------------------------ the exit walk


def test_target_hit_exits_at_the_target_with_costs_and_r_multiple(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1020.0, 995.0, 1015.0), (1, 1016.0, 1045.0, 1010.0, 1040.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "target" and o.exit_date == SESSION + timedelta(days=1)
    assert o.exit_price == 1040.0 and o.days_held == 2
    assert o.gross_return_pct == pytest.approx(4.0)
    assert o.costs_pct == pytest.approx(COST)
    assert o.net_return_pct == pytest.approx(4.0 - COST)
    assert o.r_multiple == pytest.approx((4.0 - COST) / STOP_PCT)
    assert o.max_favorable_pct == pytest.approx(4.5)  # the exit day's high, 1045
    assert o.max_adverse_pct == pytest.approx(-0.5)  # the entry day's low, 995
    assert o.entry_price == 1000.0 and o.entry_date == SESSION
    assert o.source == "yfinance" and o.graded_at == NOW and o.notes == "target hit"
    assert ledger.open_ideas(db) == []


def test_gap_through_the_target_exits_at_the_open(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1010.0, 995.0, 1008.0), (1, 1052.0, 1060.0, 1048.0, 1055.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "target" and o.exit_price == 1052.0
    assert o.gross_return_pct == pytest.approx(5.2)
    assert o.notes is not None and "gapped through the target" in o.notes


def test_stop_hit_exits_at_the_stop(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1010.0, 995.0, 1002.0), (1, 998.0, 1005.0, 975.0, 990.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "stop" and o.exit_price == 980.0 and o.days_held == 2
    assert o.gross_return_pct == pytest.approx(-2.0)
    assert o.net_return_pct == pytest.approx(-2.0 - COST)
    assert o.r_multiple == pytest.approx((-2.0 - COST) / STOP_PCT)  # a touch worse than -1R
    assert o.max_adverse_pct == pytest.approx(-2.5) and o.max_favorable_pct == pytest.approx(1.0)
    assert o.notes == "stop hit"


def test_gap_below_the_stop_exits_at_the_open(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1010.0, 995.0, 1002.0), (1, 965.0, 972.0, 960.0, 970.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "stop" and o.exit_price == 965.0
    assert o.gross_return_pct == pytest.approx(-3.5)
    assert o.notes is not None and "gapped below the stop" in o.notes


def test_target_and_stop_on_the_same_session_count_as_the_stop(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1005.0, 996.0, 1001.0), (1, 1000.0, 1045.0, 975.0, 1030.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "stop" and o.exit_price == 980.0
    assert o.notes is not None and "conservative" in o.notes
    assert o.max_favorable_pct == pytest.approx(4.5)  # the target was touched, and it says so


def test_stop_on_the_entry_session_itself(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    o = grade_idea(db, idea, bars((0, 1000.0, 1004.0, 979.0, 985.0)), now=NOW, source="yf")
    assert o.status == "stop" and o.exit_price == 980.0 and o.days_held == 1
    assert o.exit_date == SESSION


def test_expiry_exits_at_the_close_of_the_last_session(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    rows = quiet(12)  # two more sessions than the swing horizon: they must not count
    o = grade_idea(db, idea, bars(*rows), now=NOW, source="yfinance")
    assert idea.horizon_days == 10
    assert o.status == "expired" and o.days_held == 10
    assert o.exit_date == SESSION + timedelta(days=9) and o.exit_price == rows[9][4]
    assert o.gross_return_pct == pytest.approx((rows[9][4] / 1000.0 - 1) * 100)
    assert o.net_return_pct == pytest.approx(o.gross_return_pct - COST)
    assert o.max_favorable_pct == pytest.approx((rows[9][2] / 1000.0 - 1) * 100)
    assert o.max_adverse_pct == pytest.approx((996.0 / 1000.0 - 1) * 100)
    assert o.notes is not None and "neither target nor stop in 10 sessions" in o.notes


def test_not_enough_sessions_keeps_the_idea_open_with_its_excursions(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    o = grade_idea(db, idea, bars(*quiet(4)), now=NOW, source="yfinance")
    assert o.status == "open" and o.exit_date is None and o.exit_price is None
    assert o.net_return_pct is None and o.days_held is None and o.r_multiple is None
    assert o.max_favorable_pct == pytest.approx(0.9) and o.max_adverse_pct == pytest.approx(-0.4)
    assert o.notes == "4 of 10 sessions elapsed"
    assert [i.id for i in ledger.open_ideas(db)] == [idea.id]

    # More bars later: the same row moves on to its exit.
    later = grade_idea(
        db, idea, bars(*quiet(4), (4, 1004.0, 1041.0, 1000.0, 1038.0)), now=NOW, source="yf"
    )
    assert later.id == o.id and later.status == "target" and later.days_held == 5
    assert _rows(db) == 1


def test_grading_a_pending_idea_fills_it_from_the_entry_session(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    frame = bars((0, 1004.0, 1012.0, 1001.0, 1010.0), (1, 1011.0, 1042.0, 1008.0, 1040.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "target" and o.entry_price == 1004.0 and o.entry_date == SESSION
    assert o.gross_return_pct == pytest.approx((1040.0 / 1004.0 - 1) * 100)
    # The risk is measured from the actual entry: (1004 - 980) / 1004.
    assert o.r_multiple == pytest.approx(o.net_return_pct / ((1004.0 - 980.0) / 1004.0 * 100))


def test_grading_a_pending_idea_that_gapped_invalidates_it(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    frame = bars((0, 1015.0, 1050.0, 1012.0, 1045.0))  # would have hit the target, but no entry
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "invalidated" and o.entry_price is None and o.exit_price is None
    assert o.notes is not None and "gapped above" in o.notes


def test_grading_a_pending_idea_without_its_entry_bar_is_refused(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="no bar for its entry session 2026-11-02"):
        grade_idea(db, idea, bars((1, 1000.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    with pytest.raises(ValueError, match="no bar for its entry session"):
        grade_idea(db, idea, bars(), now=NOW, source="yf")
    assert _rows(db) == 0


def test_bars_before_the_entry_session_are_ignored(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((-3, 900.0, 1100.0, 850.0, 950.0), (0, 1000.0, 1041.0, 999.0, 1035.0))
    o = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    assert o.status == "target" and o.days_held == 1
    assert o.max_adverse_pct == pytest.approx(-0.1)


def test_a_bar_with_a_missing_price_is_an_error_not_a_guess(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1010.0, 995.0, 1002.0), (1, 1001.0, math.nan, 998.0, 1003.0))
    with pytest.raises(ValueError, match="bar for 2026-11-03 has a missing"):
        grade_idea(db, idea, frame, now=NOW, source="yfinance")
    with pytest.raises(ValueError, match="missing columns: low"):
        session_bars(frame.drop(columns=["low"]))
    o = idea_grading.outcome_of(db, idea)
    assert o is not None and o.status == "open" and o.max_favorable_pct is None


def test_grading_is_idempotent(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1020.0, 995.0, 1015.0), (1, 1016.0, 1045.0, 1010.0, 1040.0))
    a = grade_idea(db, idea, frame, now=NOW, source="yfinance")
    b = grade_idea(db, idea, frame, now=NOW + timedelta(days=1), source="nse_bhavcopy")
    assert a.id == b.id and _rows(db) == 1
    for col in ("status", "exit_date", "exit_price", "net_return_pct", "r_multiple", "days_held"):
        assert getattr(a, col) == getattr(b, col), col
    assert b.graded_at == NOW + timedelta(days=1) and b.source == "nse_bhavcopy"


def test_costs_come_from_the_cost_model_settings(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    frame = bars((0, 1000.0, 1045.0, 998.0, 1040.0))
    cheap = Settings(
        brokerage_pct=0.0,
        stt_delivery_pct=0.0,
        exchange_txn_pct=0.0,
        gst_pct=0.0,
        stamp_duty_pct=0.0,
        sebi_fee_pct=0.0,
        slippage_pct=0.0,
    )
    o = grade_idea(db, idea, frame, now=NOW, source="yf", settings=cheap)
    assert o.costs_pct == 0.0 and o.net_return_pct == pytest.approx(4.0)
    assert o.r_multiple == pytest.approx(2.0)
    dear = Settings(slippage_pct=0.5)  # 0.5 % per side
    o2 = grade_idea(db, idea, frame, now=NOW, source="yf", settings=dear)
    assert o2.costs_pct == pytest.approx(CostModel.from_settings(dear).round_trip_pct())
    assert o2.costs_pct == pytest.approx(COST + 0.8)


# ------------------------------------------------------------------ pure walk


def test_walk_exit_pure_rules() -> None:
    d = SESSION
    sb = [
        SessionBar(d, 1000.0, 1005.0, 998.0, 1002.0),
        SessionBar(d + timedelta(days=1), 1003.0, 1012.0, 1000.0, 1010.0),
    ]
    w = walk_exit(1000.0, 980.0, 1040.0, sb, 10)
    assert w.status == "open" and w.sessions == 2 and w.exit_price is None
    assert w.max_favorable_pct == pytest.approx(1.2) and w.max_adverse_pct == pytest.approx(-0.2)
    assert walk_exit(1000.0, 980.0, 1040.0, [], 10).status == "open"
    with pytest.raises(ValueError, match="horizon_days"):
        walk_exit(1000.0, 980.0, 1040.0, sb, 0)
    with pytest.raises(ValueError, match="need stop"):
        walk_exit(1000.0, 1000.0, 1040.0, sb, 10)
    with pytest.raises(ValueError, match="entry_price"):
        walk_exit(math.nan, 980.0, 1040.0, sb, 10)
    expired = walk_exit(1000.0, 980.0, 1040.0, sb, 2)
    assert expired.status == "expired" and expired.exit_price == 1010.0 and expired.sessions == 2


def test_session_bars_accepts_a_date_column_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "trade_date": [SESSION + timedelta(days=1), SESSION],
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
        }
    )
    sb = session_bars(df)
    assert [b.trade_date for b in sb] == [SESSION, SESSION + timedelta(days=1)]
    assert sb[0].open == 2.0
    assert session_bars(df, SESSION + timedelta(days=1))[0].open == 1.0
    assert session_bars(pd.DataFrame()) == []


# ------------------------------------------------------------------ voiding


def test_void_idea_records_why_and_stops_the_waiting(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    o = void_idea(db, idea, now=NOW, reason="session never traded (ad-hoc holiday)")
    assert o.status == "void" and o.error_category == "no_session" and o.source == "manual"
    assert o.notes == "voided (no_session): session never traded (ad-hoc holiday)"
    assert o.entry_price is None and o.exit_price is None
    assert ledger.open_ideas(db) == [] and _rows(db) == 1
    with pytest.raises(ValueError, match="void: nothing to grade"):
        grade_idea(db, idea, bars((0, 1000.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    untouched = open_check_fill(db, idea, 1000.0, now=NOW, source="kite")
    assert untouched.status == "void"

    with pytest.raises(ValueError, match="category"):
        void_idea(db, idea, now=NOW, reason="x", category="typo")
    with pytest.raises(ValueError, match="reason"):
        void_idea(db, idea, now=NOW, reason="   ")


def test_void_keeps_the_entry_and_refuses_a_graded_idea(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1002.0, now=OPEN_NOW, source="kite")
    o = void_idea(db, idea, now=NOW, reason="vendor bars gone", category="data_error")
    assert o.status == "void" and o.entry_price == 1002.0 and o.entry_date == SESSION
    assert o.error_category == "data_error"

    inst = make_instrument(db, "TCS")
    other = record_idea(db, make_idea(inst.id, "TCS"))
    grade_idea(db, other, bars((0, 1000.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    with pytest.raises(ValueError, match="already graded \\(target\\)"):
        void_idea(db, other, now=NOW, reason="too late")


def test_outcomes_for_many_ideas(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    inst = make_instrument(db, "TCS")
    other = record_idea(db, make_idea(inst.id, "TCS"))
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    found = idea_grading.outcomes_for(db, [idea, other])
    assert set(found) == {idea.id} and found[idea.id].status == "open"
    assert idea_grading.outcomes_for(db, []) == {}
    assert idea_grading.outcome_of(db, other.id) is None
    assert idea_grading.status_of(None) == "pending"


def test_outcome_dates_are_plain_dates(db: Session, idea) -> None:  # type: ignore[no-untyped-def]
    open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    o = grade_idea(db, idea, bars((0, 1000.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    assert isinstance(o.exit_date, date) and isinstance(o.entry_date, date)

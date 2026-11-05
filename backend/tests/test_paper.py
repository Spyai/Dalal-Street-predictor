"""The paper portfolio: trade P&L with costs, equity snapshots, drawdown, profit factor,
Sharpe / Sortino on known series, the summary with its sample sizes."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import CostModel
from app.models import PaperEquity, PaperTrade
from app.services import paper
from app.services.idea_grading import grade_idea, open_check_fill
from app.services.ledger import record_idea
from app.services.paper import (
    PortfolioSummary,
    close_trade,
    daily_returns,
    manual_trade,
    max_drawdown_pct,
    open_trade_for_idea,
    portfolio_summary,
    profit_factor,
    rebased,
    sharpe_ratio,
    snapshot_equity,
    sortino_ratio,
    sync_trade_with_outcome,
)
from tests.test_idea_grading import bars
from tests.test_idea_ledger import SESSION, make_idea, make_instrument

OPEN_NOW = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)
NOW = datetime(2026, 11, 2, 10, 16, tzinfo=UTC)
COST_PCT = CostModel.from_settings().round_trip_pct()
CAPITAL = get_settings().paper_capital  # 1,000,000 by default
D1 = SESSION + timedelta(days=1)
D2 = SESSION + timedelta(days=2)


def _trades(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(PaperTrade)) or 0)


# ------------------------------------------------------------------ pure math


def test_max_drawdown_on_a_known_series() -> None:
    assert max_drawdown_pct([100.0, 120.0, 90.0, 110.0, 80.0]) == pytest.approx(100.0 / 3.0)
    assert max_drawdown_pct([100.0, 110.0, 121.0]) == 0.0
    assert max_drawdown_pct([100.0]) == 0.0
    assert max_drawdown_pct([]) is None
    assert max_drawdown_pct([100.0, math.nan, 50.0]) == pytest.approx(50.0)  # NaN rows skipped
    assert max_drawdown_pct([100.0, 0.0]) == 100.0  # wiped out: a 100 % drawdown, not an error
    with pytest.raises(ValueError):
        max_drawdown_pct([0.0, 50.0])  # a non-positive peak has no drawdown to measure
    with pytest.raises(ValueError):
        daily_returns([100.0, 0.0, 50.0])


def test_daily_returns() -> None:
    np.testing.assert_allclose(daily_returns([100.0, 110.0, 99.0]), [0.1, -0.1])
    assert daily_returns([100.0]).size == 0 and daily_returns([]).size == 0


def test_sharpe_and_sortino_on_a_known_series() -> None:
    r = [0.01, -0.01, 0.02, 0.0]
    expected_sharpe = np.mean(r) / np.std(r, ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected_sharpe)
    assert sharpe_ratio(r) == pytest.approx(0.005 / math.sqrt(0.0005 / 3) * math.sqrt(252))
    # Downside deviation: only the -1 % day counts: sqrt(0.0001 / 4) = 0.005 -> ratio 1 x sqrt(252).
    assert sortino_ratio(r) == pytest.approx(math.sqrt(252))
    assert sharpe_ratio(np.asarray(r)) == pytest.approx(expected_sharpe)  # arrays accepted
    assert sharpe_ratio(r, periods_per_year=1) == pytest.approx(np.mean(r) / np.std(r, ddof=1))

    assert sharpe_ratio([0.01]) is None and sortino_ratio([0.01]) is None  # one return: no std
    assert sharpe_ratio([0.01, 0.01, 0.01]) is None  # constant: no deviation to divide by
    assert sortino_ratio([0.01, 0.02, 0.0]) is None  # no negative return: no downside
    assert sharpe_ratio([]) is None and sortino_ratio([]) is None


def test_profit_factor_and_rebasing() -> None:
    assert profit_factor([100.0, -50.0, 30.0, -20.0]) == pytest.approx(130.0 / 70.0)
    assert profit_factor([100.0, 30.0]) is None  # nothing lost: not a ratio
    assert profit_factor([]) is None
    assert profit_factor([-10.0]) == 0.0
    assert paper.expectancy([100.0, -50.0]) == 25.0 and paper.expectancy([]) is None
    assert rebased([None, 20000.0, 21000.0, None], 1e6) == [None, 1e6, 1.05e6, None]
    assert rebased([None, None], 1e6) == [None, None]


# ------------------------------------------------------------------ trades from ideas


@pytest.fixture()
def filled(db: Session):  # type: ignore[no-untyped-def]
    inst = make_instrument(db)
    idea = record_idea(db, make_idea(inst.id))  # qty 50, zone 990..1010, stop 980, target 1040
    outcome = open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    db.commit()
    return idea, outcome


def test_open_trade_for_idea_mirrors_the_fill_and_is_idempotent(db: Session, filled) -> None:  # type: ignore[no-untyped-def]
    idea, outcome = filled
    t = open_trade_for_idea(db, idea, outcome, now=NOW)
    assert t.idea_id == idea.id and t.symbol == "RELIANCE" and t.exchange == "NSE"
    assert t.side == "buy" and t.qty == 50 and t.status == "open"
    assert t.entry_date == SESSION and t.entry_price == 1000.0
    assert t.stop == 980.0 and t.target == 1040.0
    assert t.exit_price is None and t.net_pnl is None and t.closed_at is None
    assert t.note is not None and f"idea seq {idea.seq}" in t.note
    again = open_trade_for_idea(db, idea, outcome, now=NOW + timedelta(days=1))
    assert again is t and _trades(db) == 1


def test_open_trade_refuses_an_unfilled_or_unsized_idea(db: Session) -> None:
    inst = make_instrument(db)
    idea = record_idea(db, make_idea(inst.id))
    gapped = open_check_fill(db, idea, 1020.0, now=OPEN_NOW, source="kite")
    with pytest.raises(ValueError, match="invalidated, not a filled position"):
        open_trade_for_idea(db, idea, gapped, now=NOW)
    unsized = record_idea(db, make_idea(make_instrument(db, "TCS").id, "TCS", qty=0))
    fill = open_check_fill(db, unsized, 1000.0, now=OPEN_NOW, source="kite")
    with pytest.raises(ValueError, match="no position size"):
        open_trade_for_idea(db, unsized, fill, now=NOW)
    with pytest.raises(ValueError, match="belongs to idea"):
        open_trade_for_idea(db, idea, fill, now=NOW)
    assert _trades(db) == 0


def test_close_trade_math_with_costs(db: Session, filled) -> None:  # type: ignore[no-untyped-def]
    idea, outcome = filled
    t = open_trade_for_idea(db, idea, outcome, now=NOW)
    closed = close_trade(db, t, D1, 1040.0, now=NOW + timedelta(days=1), note="target")
    assert closed is t and t.status == "closed"
    assert t.exit_date == D1 and t.exit_price == 1040.0
    assert t.gross_pnl == pytest.approx(50 * 40.0)
    assert t.costs == pytest.approx(50 * 1000.0 * COST_PCT / 100.0)
    assert t.net_pnl == pytest.approx(2000.0 - 50 * 1000.0 * COST_PCT / 100.0)
    assert t.r_multiple == pytest.approx(t.net_pnl / (50 * 20.0))
    assert t.closed_at == NOW + timedelta(days=1) and t.note is not None and "target" in t.note

    # A loser: -1R minus costs. Re-closing recomputes in place.
    close_trade(db, t, D1, 980.0, now=NOW + timedelta(days=1))
    assert t.gross_pnl == pytest.approx(-1000.0)
    assert t.net_pnl == pytest.approx(-1000.0 - 50 * 1000.0 * COST_PCT / 100.0)
    assert t.r_multiple == pytest.approx(t.net_pnl / 1000.0) and t.r_multiple < -1.0
    with pytest.raises(ValueError, match="before the entry"):
        close_trade(db, t, SESSION - timedelta(days=1), 1000.0, now=NOW)
    with pytest.raises(ValueError, match="exit_price"):
        close_trade(db, t, D1, math.nan, now=NOW)
    cheap = Settings(
        brokerage_pct=0.0,
        stt_delivery_pct=0.0,
        exchange_txn_pct=0.0,
        gst_pct=0.0,
        stamp_duty_pct=0.0,
        sebi_fee_pct=0.0,
        slippage_pct=0.0,
    )
    close_trade(db, t, D1, 1040.0, now=NOW, settings=cheap)
    assert t.costs == 0.0 and t.net_pnl == 2000.0 and t.r_multiple == 2.0


def test_sync_trade_with_outcome_opens_then_closes(db: Session, filled) -> None:  # type: ignore[no-untyped-def]
    idea, outcome = filled
    t = sync_trade_with_outcome(db, idea, outcome, now=NOW)
    assert t is not None and t.status == "open" and _trades(db) == 1
    assert sync_trade_with_outcome(db, idea, outcome, now=NOW) is t  # in step: nothing changes

    graded = grade_idea(
        db,
        idea,
        bars((0, 1000.0, 1010.0, 995.0, 1005.0), (1, 1006.0, 1045.0, 1002.0, 1040.0)),
        now=NOW + timedelta(days=1),
        source="yf",
    )
    assert graded.status == "target"
    t2 = sync_trade_with_outcome(db, idea, graded, now=NOW + timedelta(days=1))
    assert t2 is t and t.status == "closed" and t.exit_date == D1 and t.exit_price == 1040.0
    assert t.net_pnl == pytest.approx(2000.0 - 50 * 1000.0 * COST_PCT / 100.0)
    assert t.note is not None and "closed by grading: target" in t.note
    assert sync_trade_with_outcome(db, idea, graded, now=NOW) is t and _trades(db) == 1


def test_sync_handles_an_idea_graded_in_one_go_and_ignores_the_rest(db: Session) -> None:
    inst = make_instrument(db)
    idea = record_idea(db, make_idea(inst.id))
    # No 09:20 fill ran: close grading fills and settles in one call.
    graded = grade_idea(db, idea, bars((0, 1000.0, 1045.0, 998.0, 1040.0)), now=NOW, source="yf")
    t = sync_trade_with_outcome(db, idea, graded, now=NOW)
    assert t is not None and t.status == "closed" and t.entry_price == 1000.0
    assert t.exit_price == 1040.0 and t.exit_date == SESSION

    other = record_idea(db, make_idea(make_instrument(db, "TCS").id, "TCS"))
    gapped = open_check_fill(db, other, 1020.0, now=OPEN_NOW, source="kite")
    assert sync_trade_with_outcome(db, other, gapped, now=NOW) is None
    assert _trades(db) == 1


def test_manual_trade_validation(db: Session) -> None:
    t = manual_trade(
        db,
        symbol="infy",
        qty=10,
        entry_date=SESSION,
        entry_price=1500.0,
        stop=1450.0,
        target=1600.0,
        note="hand entry",
    )
    assert t.idea_id is None and t.symbol == "INFY" and t.exchange == "NSE" and t.status == "open"
    assert t.qty == 10 and t.stop == 1450.0 and t.target == 1600.0 and t.note == "hand entry"
    with pytest.raises(ValueError, match="qty"):
        manual_trade(db, symbol="INFY", qty=0, entry_date=SESSION, entry_price=1500.0)
    with pytest.raises(ValueError, match="entry_price"):
        manual_trade(db, symbol="INFY", qty=1, entry_date=SESSION, entry_price=math.nan)
    with pytest.raises(ValueError, match="stop .* below"):
        manual_trade(db, symbol="INFY", qty=1, entry_date=SESSION, entry_price=1500.0, stop=1500.0)
    with pytest.raises(ValueError, match="target .* above"):
        manual_trade(
            db, symbol="INFY", qty=1, entry_date=SESSION, entry_price=1500.0, target=1400.0
        )
    with pytest.raises(ValueError, match="symbol"):
        manual_trade(db, symbol="  ", qty=1, entry_date=SESSION, entry_price=1500.0)
    assert _trades(db) == 1
    closed = close_trade(db, t, D1, 1600.0, now=NOW)
    assert closed.r_multiple == pytest.approx(closed.net_pnl / (10 * 50.0))
    no_stop = manual_trade(db, symbol="TCS", qty=1, entry_date=SESSION, entry_price=100.0)
    close_trade(db, no_stop, D1, 110.0, now=NOW)
    assert no_stop.r_multiple is None  # no stop, no R


# ------------------------------------------------------------------ equity snapshots


def test_snapshot_equity_marks_open_positions_to_close(db: Session, filled) -> None:  # type: ignore[no-untyped-def]
    idea, outcome = filled
    t = open_trade_for_idea(db, idea, outcome, now=NOW)
    e = snapshot_equity(db, SESSION, {"RELIANCE": 1020.0}, 25000.0, source="test")
    assert e.trade_date == SESSION and e.open_positions == 1
    assert e.cash == pytest.approx(CAPITAL - 50_000.0)
    assert e.invested == pytest.approx(50 * 1020.0)
    assert e.equity == pytest.approx(CAPITAL + 1_000.0)
    assert e.nifty_close == 25000.0 and e.source == "test"

    # Same date again (a re-run with a corrected close): one row, rewritten.
    e2 = snapshot_equity(db, SESSION, {"RELIANCE": 1010.0}, None, source="test2")
    assert e2.equity == pytest.approx(CAPITAL + 500.0) and e2.nifty_close is None
    assert db.scalar(select(func.count()).select_from(PaperEquity)) == 1

    # After the close on D1 the cash holds the realised net P&L and nothing is invested;
    # a snapshot of the earlier date still sees the position open as it was then.
    close_trade(db, t, D1, 1040.0, now=NOW)
    e3 = snapshot_equity(db, D1, {}, 25100.0)
    assert e3.open_positions == 0 and e3.invested == 0.0
    assert e3.cash == pytest.approx(CAPITAL + t.net_pnl) and e3.equity == e3.cash
    assert e3.source == "close_grading"
    e4 = snapshot_equity(db, SESSION, {"RELIANCE": 1010.0}, None)
    assert e4.open_positions == 1 and e4.equity == pytest.approx(CAPITAL + 500.0)
    # A trade entered later is not part of an earlier day's book.
    manual_trade(db, symbol="TCS", qty=10, entry_date=D2, entry_price=100.0)
    assert snapshot_equity(db, D1, {}, None).open_positions == 0
    assert snapshot_equity(db, D2, {"TCS": 101.0}, None).invested == pytest.approx(1010.0)


def test_snapshot_refuses_to_guess_a_missing_close(db: Session, filled) -> None:  # type: ignore[no-untyped-def]
    idea, outcome = filled
    open_trade_for_idea(db, idea, outcome, now=NOW)
    with pytest.raises(
        ValueError, match="no close for 2026-11-02 for open position\\(s\\): RELIANCE"
    ):
        snapshot_equity(db, SESSION, {}, 25000.0)
    with pytest.raises(ValueError, match="RELIANCE"):
        snapshot_equity(db, SESSION, {"RELIANCE": math.nan}, 25000.0)
    assert db.scalar(select(func.count()).select_from(PaperEquity)) == 0
    assert paper.equity_curve(db) == []


# ------------------------------------------------------------------ the summary


def _equity_row(
    d: date, equity: float, nifty: float | None, cash: float | None = None
) -> PaperEquity:
    return PaperEquity(
        trade_date=d,
        equity=equity,
        cash=equity if cash is None else cash,
        invested=0.0 if cash is None else equity - cash,
        open_positions=0,
        nifty_close=nifty,
        source="test",
    )


def test_empty_summary_is_honest_about_having_nothing(db: Session) -> None:
    s = portfolio_summary(db)
    assert isinstance(s, PortfolioSummary)
    assert s.capital == CAPITAL and s.n_trades == 0 and s.n_open == 0
    assert s.n_wins == 0 and s.n_losses == 0 and s.n_days == 0 and s.n_returns == 0
    for name in (
        "hit_rate",
        "avg_win",
        "avg_loss",
        "avg_win_pct",
        "avg_loss_pct",
        "expectancy",
        "expectancy_pct",
        "profit_factor",
        "total_net_pnl",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "total_return_pct",
        "nifty_return_pct",
    ):
        assert getattr(s, name) is None, name
    assert s.equity_curve == [] and s.open_trades == [] and s.recent_closed == []
    assert s.sample_warning is True
    assert "price index" in s.note and "total-return index is not available" in s.note
    assert "paper" in s.note.lower()
    assert s.as_dict()["n_trades"] == 0


def test_summary_aggregates_trades_and_the_curve(db: Session) -> None:
    # Three closed trades (two winners, one loser) and one open, all at entry 1000.
    cheap = Settings(
        brokerage_pct=0.0,
        stt_delivery_pct=0.0,
        exchange_txn_pct=0.0,
        gst_pct=0.0,
        stamp_duty_pct=0.0,
        sebi_fee_pct=0.0,
        slippage_pct=0.0,
        min_sample_size=3,
    )
    a = manual_trade(db, symbol="A", qty=10, entry_date=SESSION, entry_price=1000.0, stop=980.0)
    b = manual_trade(db, symbol="B", qty=10, entry_date=SESSION, entry_price=1000.0, stop=980.0)
    c = manual_trade(db, symbol="C", qty=10, entry_date=D1, entry_price=1000.0, stop=980.0)
    manual_trade(db, symbol="D", qty=10, entry_date=D2, entry_price=1000.0, stop=980.0)
    close_trade(db, a, D1, 1040.0, now=NOW, settings=cheap)  # +400
    close_trade(db, b, D1, 980.0, now=NOW, settings=cheap)  # -200
    close_trade(db, c, D2, 1020.0, now=NOW, settings=cheap)  # +200
    for d, eq, nifty in (
        (SESSION, 1_000_000.0, 25_000.0),
        (D1, 1_000_200.0, 25_250.0),
        (D2, 1_000_000.0, None),
        (D2 + timedelta(days=1), 1_000_400.0, 25_500.0),
    ):
        db.add(_equity_row(d, eq, nifty))
    db.commit()

    s = portfolio_summary(db, settings=cheap)
    assert s.n_trades == 3 and s.n_open == 1 and s.n_wins == 2 and s.n_losses == 1
    assert s.hit_rate == pytest.approx(2 / 3)
    assert s.avg_win == 300.0 and s.avg_loss == -200.0
    assert s.avg_win_pct == pytest.approx(3.0) and s.avg_loss_pct == pytest.approx(-2.0)
    assert s.expectancy == pytest.approx(400.0 / 3) and s.expectancy_pct == pytest.approx(4.0 / 3)
    assert s.profit_factor == pytest.approx(600.0 / 200.0)
    assert s.total_net_pnl == 400.0
    assert s.sample_warning is False  # min_sample_size 3 reached
    assert portfolio_summary(db).sample_warning is True  # the default 100 is not

    # The curve statistics start from the paper capital on the session before the first
    # snapshot, so four snapshots give four daily returns (the first one 0 here).
    equity = [1_000_000.0, 1_000_200.0, 1_000_000.0, 1_000_400.0]
    r = daily_returns([cheap.paper_capital, *equity])
    assert s.n_days == 4 and s.n_returns == 4
    assert s.max_drawdown_pct == pytest.approx(200.0 / 1_000_200.0 * 100)
    assert s.min_cash == pytest.approx(1_000_000.0) and s.levered_days == 0  # cash = equity here
    assert s.max_exposure_pct == 0.0  # the test rows carry no invested amount
    assert s.sharpe == pytest.approx(sharpe_ratio(r))
    assert s.sortino == pytest.approx(sortino_ratio(r))
    assert s.total_return_pct == pytest.approx(0.04)
    assert s.nifty_return_pct == pytest.approx(2.0)  # 25000 -> 25500 over the known closes
    assert [row["nifty_rebased"] for row in s.equity_curve] == [
        pytest.approx(1_000_000.0),
        pytest.approx(1_010_000.0),
        None,
        pytest.approx(1_020_000.0),
    ]
    assert s.equity_curve[2]["nifty_close"] is None  # missing stays missing
    assert [row["trade_date"] for row in s.equity_curve][0] == SESSION.isoformat()
    assert [t["symbol"] for t in s.open_trades] == ["D"]
    assert [t["symbol"] for t in s.recent_closed] == ["C", "B", "A"]  # newest exit first
    assert s.recent_closed[0]["net_return_pct"] == pytest.approx(2.0)
    assert s.recent_closed[1]["r_multiple"] == pytest.approx(-1.0)
    assert paper.list_trades(db, status="open")[0].symbol == "D"
    assert len(paper.list_trades(db, status="closed")) == 3


def test_summary_with_too_few_snapshots_has_no_ratios(db: Session) -> None:
    db.add(_equity_row(SESSION, 1_001_000.0, 25_000.0))
    db.commit()
    s = portfolio_summary(db)
    assert s.n_days == 1 and s.n_returns == 1  # capital -> first snapshot: one return
    assert s.sharpe is None and s.sortino is None  # one return: no deviation
    assert s.max_drawdown_pct == 0.0 and s.total_return_pct == pytest.approx(0.1)
    assert s.nifty_return_pct is None  # one close: no move to report
    # A fall on the very first session is a drawdown from the capital, not from nothing.
    db.add(_equity_row(D1, 990_000.0, 25_100.0, cash=-5_000.0))
    db.commit()
    s2 = portfolio_summary(db)
    assert s2.n_days == 2 and s2.n_returns == 2
    assert s2.max_drawdown_pct == pytest.approx(11_000.0 / 1_001_000.0 * 100)
    assert s2.min_cash == -5_000.0 and s2.levered_days == 1
    assert s2.max_exposure_pct == pytest.approx(995_000.0 / 990_000.0 * 100)
    assert s2.nifty_return_pct == pytest.approx(0.4)

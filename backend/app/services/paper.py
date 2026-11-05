"""The paper portfolio (brief §7.10, §6.3): simulated trades with costs, never real orders.

Every filled idea becomes one `PaperTrade` (`open_trade_for_idea`), closed when the idea
is graded (`close_trade`, or `sync_trade_with_outcome` which does both from an
`IdeaOutcome`); manual paper trades come in through `manual_trade` with `idea_id` NULL.
Costs are the round trip of `core.CostModel` (brokerage, STT, exchange charge, GST, stamp
duty, SEBI fee, slippage) as a % of the entry value, charged when the trade closes:
`net_pnl = gross_pnl - costs`, `r_multiple = net_pnl / (qty x (entry - stop))`.

`snapshot_equity` writes one `paper_equity` row per session: cash = paper capital + the
net P&L of every trade closed by that date - the entry value of the positions still open,
invested = those positions marked to the session's closes, equity = cash + invested. A
position whose close is missing stops the snapshot (ValueError) rather than being marked
at a guess. `nifty_close` is the Nifty 50 *price* index (the total-return index is not
available for free; the UI says so) and may be None when it was not available.

`portfolio_summary` derives everything from those rows and the trades: hit rate, average
win / loss, expectancy, profit factor, max drawdown, Sharpe and Sortino (daily returns of
the equity curve, annualised with sqrt(252), no risk-free rate subtracted), and the
equity curve beside the Nifty 50 price index rebased to the paper capital. Every metric
comes with the sample size behind it and is None when there is nothing to compute it
from; the sample-size warning is on below `Settings.min_sample_size` closed trades. It is
a paper record: simulated fills at recorded prices, no market impact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import CostModel, finite, pct_change
from app.models import Idea, IdeaOutcome, PaperEquity, PaperTrade

TRADING_DAYS_PER_YEAR = 252
MIN_RETURNS_FOR_RATIOS = 2  # Sharpe/Sortino need a standard deviation: two daily returns
TERMINAL_STATUSES: tuple[str, ...] = ("target", "stop", "expired")
PAPER_NOTE = (
    "Paper portfolio: simulated fills at recorded prices with the configured cost model, "
    "no market impact. Benchmark is the Nifty 50 price index rebased to the paper capital "
    "(the total-return index is not available). Sharpe and Sortino use daily equity "
    "returns annualised with sqrt(252) and no risk-free rate."
)


# ----------------------------------------------------------------------------- pure math


Numbers = Sequence[float | None] | np.ndarray


def _finite_list(values: Numbers) -> list[float]:
    out: list[float] = []
    for v in list(values):
        f = finite(v)
        if f is not None:
            out.append(f)
    return out


def daily_returns(equity: Numbers) -> np.ndarray:
    """Simple period returns E_t / E_{t-1} - 1 of an equity series (empty below 2 points)."""
    e = np.asarray(_finite_list(equity), dtype=float)
    if e.size < 2:
        return np.asarray([], dtype=float)
    if np.any(e[:-1] <= 0.0):
        raise ValueError("equity must stay positive to compute returns")
    return e[1:] / e[:-1] - 1.0


def max_drawdown_pct(equity: Numbers) -> float | None:
    """The deepest peak-to-trough fall of the series, in % of the peak (>= 0); None if empty."""
    e = np.asarray(_finite_list(equity), dtype=float)
    if e.size == 0:
        return None
    peaks = np.maximum.accumulate(e)
    if np.any(peaks <= 0.0):
        raise ValueError("equity must stay positive to compute a drawdown")
    return float(np.max((peaks - e) / peaks) * 100.0)


def sharpe_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """mean / sample std of the period returns, annualised; None below two returns or when
    the returns do not vary (a ratio over zero is not a number)."""
    r = np.asarray(_finite_list(returns), dtype=float)
    if r.size < MIN_RETURNS_FOR_RATIOS:
        return None
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return None
    return float(np.mean(r) / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """mean / downside deviation (root mean square of the negative returns, target 0),
    annualised; None below two returns or without a single negative return."""
    r = np.asarray(_finite_list(returns), dtype=float)
    if r.size < MIN_RETURNS_FOR_RATIOS:
        return None
    downside = float(np.sqrt(np.mean(np.square(np.minimum(r, 0.0)))))
    if downside <= 0.0:
        return None
    return float(np.mean(r) / downside * math.sqrt(periods_per_year))


def profit_factor(pnls: Numbers) -> float | None:
    """Sum of the winning trades' P&L over the losing trades' absolute P&L; None without
    a losing trade (nothing to divide by) or without trades."""
    p = _finite_list(pnls)
    if not p:
        return None
    losses = -sum(x for x in p if x < 0.0)
    if losses <= 0.0:
        return None
    return sum(x for x in p if x > 0.0) / losses


def expectancy(pnls: Numbers) -> float | None:
    """Mean P&L per trade; None without trades."""
    p = _finite_list(pnls)
    return float(np.mean(p)) if p else None


def rebased(values: Sequence[float | None], base: float) -> list[float | None]:
    """`values` scaled so the first non-None one equals `base`; None stays None."""
    first = next((v for v in values if v is not None), None)
    if first is None or first <= 0.0:
        return [None for _ in values]
    return [None if v is None else base * v / first for v in values]


# ----------------------------------------------------------------------------- trades


def _positive(value: object, name: str) -> float:
    v = finite(value)
    if v is None or v <= 0.0:
        raise ValueError(f"{name} is missing or not positive: {value!r}")
    return v


def _check_levels(entry: float, stop: float | None, target: float | None) -> None:
    if stop is not None and stop >= entry:
        raise ValueError(f"stop {stop} must be below the entry {entry}")
    if target is not None and target <= entry:
        raise ValueError(f"target {target} must be above the entry {entry}")


def trade_for_idea(db: Session, idea: Idea | int) -> PaperTrade | None:
    idea_id = idea if isinstance(idea, int) else idea.id
    return db.scalars(
        select(PaperTrade).where(PaperTrade.idea_id == idea_id).order_by(PaperTrade.id.asc())
    ).first()


def open_trade_for_idea(
    db: Session, idea: Idea, outcome: IdeaOutcome, *, now: datetime | None = None
) -> PaperTrade:
    """One paper position for a filled idea: the recorded qty at the outcome's entry price.

    Idempotent: an idea already has at most one trade, which is returned as it is. The
    outcome must be a fill (`status == "open"` with an entry); anything else is a caller
    bug. `now` is unused for the row (created_at is the database's) and kept for symmetry.
    """
    existing = trade_for_idea(db, idea)
    if existing is not None:
        return existing
    if outcome.idea_id != idea.id:
        raise ValueError(f"outcome belongs to idea {outcome.idea_id}, not {idea.id}")
    if outcome.status != "open" or outcome.entry_price is None or outcome.entry_date is None:
        raise ValueError(f"idea seq {idea.seq} is {outcome.status}, not a filled position")
    if idea.position_qty <= 0:
        raise ValueError(f"idea seq {idea.seq} has no position size (qty {idea.position_qty})")
    entry = _positive(outcome.entry_price, "entry_price")
    trade = PaperTrade(
        idea_id=idea.id,
        symbol=idea.symbol,
        exchange=idea.exchange,
        side="buy",
        qty=int(idea.position_qty),
        entry_date=outcome.entry_date,
        entry_price=entry,
        stop=finite(idea.stop),
        target=finite(idea.target_1),
        status="open",
        note=f"auto: idea seq {idea.seq} ({idea.horizon}, {idea.setup})",
    )
    db.add(trade)
    db.flush()
    return trade


def manual_trade(
    db: Session,
    *,
    symbol: str,
    qty: int,
    entry_date: date,
    entry_price: float,
    exchange: str = "NSE",
    stop: float | None = None,
    target: float | None = None,
    note: str | None = None,
) -> PaperTrade:
    """A paper position entered by hand (CLI): no idea behind it, same cost model."""
    if not symbol or not symbol.strip():
        raise ValueError("a symbol is required")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ValueError(f"qty must be a positive int, got {qty!r}")
    entry = _positive(entry_price, "entry_price")
    stop_ = finite(stop) if stop is not None else None
    target_ = finite(target) if target is not None else None
    if stop is not None and (stop_ is None or stop_ <= 0.0):
        raise ValueError(f"stop is not a positive number: {stop!r}")
    if target is not None and target_ is None:
        raise ValueError(f"target is not a number: {target!r}")
    _check_levels(entry, stop_, target_)
    trade = PaperTrade(
        idea_id=None,
        symbol=symbol.strip().upper(),
        exchange=exchange,
        side="buy",
        qty=qty,
        entry_date=entry_date,
        entry_price=entry,
        stop=stop_,
        target=target_,
        status="open",
        note=note,
    )
    db.add(trade)
    db.flush()
    return trade


def trade_costs(trade: PaperTrade, costs: CostModel) -> float:
    """Round-trip costs in rupees: the cost model's % of the entry value."""
    return trade.qty * trade.entry_price * costs.round_trip_pct() / 100.0


def close_trade(
    db: Session,
    trade: PaperTrade,
    exit_date: date,
    exit_price: float,
    *,
    now: datetime,
    settings: Settings | None = None,
    note: str | None = None,
) -> PaperTrade:
    """Settle a position: gross P&L, round-trip costs, net P&L and R-multiple.

    Closing again with another exit (a corrected print) recomputes in place; the exit
    cannot predate the entry.
    """
    exit_ = _positive(exit_price, "exit_price")
    if exit_date < trade.entry_date:
        raise ValueError(
            f"exit {exit_date.isoformat()} is before the entry {trade.entry_date.isoformat()}"
        )
    costs = CostModel.from_settings(settings or get_settings())
    gross = trade.qty * (exit_ - trade.entry_price)
    cost = trade_costs(trade, costs)
    net = gross - cost
    trade.exit_date = exit_date
    trade.exit_price = exit_
    trade.status = "closed"
    trade.gross_pnl = finite(gross)
    trade.costs = finite(cost)
    trade.net_pnl = finite(net)
    risk = trade.qty * (trade.entry_price - trade.stop) if trade.stop is not None else None
    trade.r_multiple = finite(net / risk) if risk is not None and risk > 0.0 else None
    trade.closed_at = now
    if note is not None:
        trade.note = f"{trade.note}; {note}" if trade.note else note
    db.flush()
    return trade


def sync_trade_with_outcome(
    db: Session,
    idea: Idea,
    outcome: IdeaOutcome,
    *,
    now: datetime,
    settings: Settings | None = None,
) -> PaperTrade | None:
    """Keep the paper book in step with an idea's outcome after grading.

    A filled idea without a trade gets one; a graded idea (target / stop / expired) with
    an open trade has it closed at the outcome's exit; anything else (pending,
    invalidated, void, or already in step) changes nothing. Returns the trade, or None
    when the idea has none.
    """
    if outcome.status == "open":
        return open_trade_for_idea(db, idea, outcome, now=now)
    trade = trade_for_idea(db, idea)
    if trade is None:
        if outcome.status in TERMINAL_STATUSES and outcome.entry_price is not None:
            # Graded in one go (the 09:20 fill never ran): open and close in sequence.
            filled = IdeaOutcome(
                idea_id=idea.id,
                status="open",
                entry_date=outcome.entry_date,
                entry_price=outcome.entry_price,
                graded_at=outcome.graded_at,
                source=outcome.source,
            )
            trade = open_trade_for_idea(db, idea, filled, now=now)
        else:
            return None
    if (
        outcome.status in TERMINAL_STATUSES
        and trade.status == "open"
        and outcome.exit_date is not None
        and outcome.exit_price is not None
    ):
        close_trade(
            db,
            trade,
            outcome.exit_date,
            outcome.exit_price,
            now=now,
            settings=settings,
            note=f"closed by grading: {outcome.status}",
        )
    return trade


def list_trades(db: Session, *, status: str | None = None, limit: int = 200) -> list[PaperTrade]:
    """Trades newest first (by entry date, then id); `status` "open" / "closed" filters."""
    stmt = select(PaperTrade).order_by(PaperTrade.entry_date.desc(), PaperTrade.id.desc())
    if status is not None:
        stmt = stmt.where(PaperTrade.status == status)
    return list(db.scalars(stmt.limit(limit)).all())


def trade_dict(t: PaperTrade) -> dict[str, Any]:
    entry_value = t.qty * t.entry_price
    return {
        "id": t.id,
        "idea_id": t.idea_id,
        "symbol": t.symbol,
        "exchange": t.exchange,
        "side": t.side,
        "qty": t.qty,
        "entry_date": t.entry_date.isoformat(),
        "entry_price": t.entry_price,
        "entry_value": finite(entry_value),
        "stop": t.stop,
        "target": t.target,
        "exit_date": t.exit_date.isoformat() if t.exit_date else None,
        "exit_price": t.exit_price,
        "status": t.status,
        "gross_pnl": t.gross_pnl,
        "costs": t.costs,
        "net_pnl": t.net_pnl,
        "net_return_pct": finite(t.net_pnl / entry_value * 100.0)
        if t.net_pnl is not None and entry_value > 0
        else None,
        "r_multiple": t.r_multiple,
        "note": t.note,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }


# ----------------------------------------------------------------------------- equity


def positions_as_of(db: Session, trade_date: date) -> tuple[list[PaperTrade], list[PaperTrade]]:
    """(open, closed) trades as the book stood at the close of `trade_date`: entered on or
    before it; closed ones exited on or before it, open ones not yet (or later)."""
    rows = db.scalars(
        select(PaperTrade)
        .where(PaperTrade.entry_date <= trade_date)
        .order_by(PaperTrade.entry_date.asc(), PaperTrade.id.asc())
    ).all()
    open_: list[PaperTrade] = []
    closed: list[PaperTrade] = []
    for t in rows:
        if t.exit_date is not None and t.exit_date <= trade_date:
            closed.append(t)
        else:
            open_.append(t)
    return open_, closed


def snapshot_equity(
    db: Session,
    trade_date: date,
    closes: Mapping[str, float],
    nifty_close: float | None,
    *,
    source: str = "close_grading",
    settings: Settings | None = None,
) -> PaperEquity:
    """Write (or rewrite) the paper-equity row for `trade_date` (module doc for the sums).

    `closes` maps symbol -> that session's close for every open position; a missing or
    non-finite close raises ValueError naming the symbols, and nothing is written. A
    closed trade whose net P&L was never computed is an error too.
    """
    s = settings or get_settings()
    open_, closed = positions_as_of(db, trade_date)
    missing = sorted({t.symbol for t in open_ if finite(closes.get(t.symbol)) is None})
    if missing:
        raise ValueError(
            f"no close for {trade_date.isoformat()} for open position(s): {', '.join(missing)}"
        )
    unsettled = [t.id for t in closed if finite(t.net_pnl) is None]
    if unsettled:
        raise ValueError(f"closed paper trade(s) without a net P&L: {unsettled}")
    realised = sum(float(t.net_pnl) for t in closed if t.net_pnl is not None)
    cost_basis = sum(t.qty * t.entry_price for t in open_)
    invested = sum(t.qty * float(closes[t.symbol]) for t in open_)
    cash = s.paper_capital + realised - cost_basis
    nifty = finite(nifty_close)
    values: dict[str, Any] = {
        "equity": finite(cash + invested),
        "cash": finite(cash),
        "invested": finite(invested),
        "open_positions": len(open_),
        "nifty_close": nifty,
        "source": source,
    }
    stmt = insert(PaperEquity).values(trade_date=trade_date, **values)
    stmt = stmt.on_conflict_do_update(index_elements=[PaperEquity.trade_date], set_=values)
    db.execute(stmt)
    db.flush()
    return db.execute(
        select(PaperEquity)
        .where(PaperEquity.trade_date == trade_date)
        .execution_options(populate_existing=True)
    ).scalar_one()


def equity_curve(db: Session) -> list[PaperEquity]:
    return list(db.scalars(select(PaperEquity).order_by(PaperEquity.trade_date.asc())).all())


# ----------------------------------------------------------------------------- summary


@dataclass
class PortfolioSummary:
    """The paper book's live record. Every metric carries the sample behind it:
    `n_trades` closed trades for the trade statistics, `n_days` equity snapshots and
    `n_returns` daily returns for the curve statistics."""

    capital: float
    n_trades: int
    n_open: int
    n_wins: int
    n_losses: int
    hit_rate: float | None  # share of closed trades with a positive net P&L
    avg_win: float | None  # mean net P&L of the winners (Rs)
    avg_loss: float | None  # mean net P&L of the losers (Rs, negative)
    avg_win_pct: float | None  # mean net return of the winners, % of entry value
    avg_loss_pct: float | None
    expectancy: float | None  # mean net P&L per closed trade (Rs)
    expectancy_pct: float | None
    profit_factor: float | None
    total_net_pnl: float | None
    n_days: int
    n_returns: int
    max_drawdown_pct: float | None
    sharpe: float | None
    sortino: float | None
    total_return_pct: float | None  # last equity vs the paper capital
    nifty_return_pct: float | None  # Nifty 50 price index over the same snapshots
    equity_curve: list[dict[str, Any]]
    open_trades: list[dict[str, Any]]
    recent_closed: list[dict[str, Any]]
    sample_warning: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: list[float]) -> float | None:
    return finite(float(np.mean(values))) if values else None


def portfolio_summary(
    db: Session, *, settings: Settings | None = None, recent_limit: int = 20
) -> PortfolioSummary:
    """Aggregate the paper book (module doc). Nothing is estimated: a metric without data
    behind it is None, and `sample_warning` is on below `min_sample_size` closed trades."""
    s = settings or get_settings()
    trades = list_trades(db, limit=100_000)
    closed = [t for t in trades if t.status == "closed" and t.net_pnl is not None]
    open_ = [t for t in trades if t.status == "open"]
    pnls = [float(t.net_pnl) for t in closed if t.net_pnl is not None]
    pcts = [
        float(t.net_pnl) / (t.qty * t.entry_price) * 100.0
        for t in closed
        if t.net_pnl is not None and t.qty * t.entry_price > 0
    ]
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]

    curve = equity_curve(db)
    equity = [e.equity for e in curve]
    returns = daily_returns(equity) if equity else np.asarray([], dtype=float)
    nifty = [e.nifty_close for e in curve]
    nifty_rebased = rebased(nifty, s.paper_capital)
    nifty_known = [v for v in nifty if v is not None]
    curve_rows = [
        {
            "trade_date": e.trade_date.isoformat(),
            "equity": e.equity,
            "cash": e.cash,
            "invested": e.invested,
            "open_positions": e.open_positions,
            "nifty_close": e.nifty_close,
            "nifty_rebased": finite(nb),
        }
        for e, nb in zip(curve, nifty_rebased, strict=True)
    ]
    n = len(closed)
    return PortfolioSummary(
        capital=s.paper_capital,
        n_trades=n,
        n_open=len(open_),
        n_wins=len(wins),
        n_losses=len(losses),
        hit_rate=finite(len(wins) / n) if n else None,
        avg_win=_mean(wins),
        avg_loss=_mean(losses),
        avg_win_pct=_mean([p for p in pcts if p > 0.0]),
        avg_loss_pct=_mean([p for p in pcts if p < 0.0]),
        expectancy=finite(expectancy(pnls)),
        expectancy_pct=_mean(pcts),
        profit_factor=finite(profit_factor(pnls)),
        total_net_pnl=finite(sum(pnls)) if pnls else None,
        n_days=len(curve),
        n_returns=int(returns.size),
        max_drawdown_pct=finite(max_drawdown_pct(equity)),
        sharpe=finite(sharpe_ratio(returns)),
        sortino=finite(sortino_ratio(returns)),
        total_return_pct=finite(pct_change(equity[-1], s.paper_capital)) if equity else None,
        nifty_return_pct=finite(pct_change(nifty_known[-1], nifty_known[0]))
        if len(nifty_known) >= 2
        else None,
        equity_curve=curve_rows,
        open_trades=[trade_dict(t) for t in open_],
        recent_closed=[
            trade_dict(t)
            for t in sorted(
                closed, key=lambda t: (t.exit_date or t.entry_date, t.id), reverse=True
            )[:recent_limit]
        ],
        sample_warning=n < s.min_sample_size,
        note=PAPER_NOTE,
    )

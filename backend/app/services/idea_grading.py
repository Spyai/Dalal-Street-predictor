"""Grading stock ideas against what the market did (brief §3 09:20 / 15:45, §6.3).

Outcomes live in `idea_outcome`, one mutable row per idea; the idea itself (the ledger
row) is never touched. Statuses (`core.IDEA_STATUSES`): pending (no row yet, or not
entered), open (filled), target / stop / expired (graded), invalidated (the session
opened outside the entry zone), void (can never be graded; the operator says why).

Entry rule (contract §2): the idea's `session_date` is the one session to enter. At the
open check the idea is filled at that session's open if the open lies inside
[entry_low, entry_high]; an open outside the zone in either direction invalidates it.
There is no multi-day waiting. `grade_idea` applies the same rule itself when it meets a
pending idea with the entry session's bar, so a missed 09:20 run does not lose the idea.

Exit rule, walked session by session over the `horizon_days` sessions starting with the
entry session (`walk_exit`): the stop is hit when the low <= stop, exit at the stop price
or at the open when the session gapped below it; the target is hit when the high >=
target_1, exit at the target or at the open when the session gapped through it; stop and
target on the same session count as the stop (conservative); nothing in the window ->
expired at the close of the last session. Fewer sessions than the window and no hit ->
the idea stays open and only its excursions (MFE / MAE) are updated.

Returns: gross % from entry to exit; costs = `CostModel.round_trip_pct()` (brokerage,
STT, exchange charge, GST, stamp duty, SEBI fee, slippage) in % of the entry value; net
= gross - costs; `r_multiple` = net % over the planned risk % (entry to stop);
`days_held` = sessions held, entry session included (1 = exited the session it was
entered). A bar with a missing price is an error, never a guess: the idea stays as it was
and the job reports it.

Every write is an idempotent upsert on `uq_idea_outcome_idea`: grading twice with the
same bars gives the same row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import CostModel, finite, pct_change
from app.models import Idea, IdeaOutcome

TERMINAL_STATUSES: tuple[str, ...] = ("target", "stop", "expired")
UNGRADEABLE_STATUSES: tuple[str, ...] = ("invalidated", "void")
VOID_CATEGORIES: tuple[str, ...] = ("no_session", "data_error")
VOID_SOURCE = "manual"
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

# Every mutable column of idea_outcome; each upsert writes all of them so no stale value
# from an earlier grade survives a re-grade.
_OUTCOME_COLUMNS: tuple[str, ...] = (
    "status",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "gross_return_pct",
    "costs_pct",
    "net_return_pct",
    "r_multiple",
    "days_held",
    "max_favorable_pct",
    "max_adverse_pct",
    "graded_at",
    "source",
    "notes",
    "error_category",
)


@dataclass(frozen=True)
class Fill:
    """The entry decision for one idea at its session's open."""

    status: str  # "open" | "invalidated"
    entry_price: float | None
    note: str | None


@dataclass(frozen=True)
class SessionBar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Walk:
    """What the exit walk found: a terminal status with its exit, or still open."""

    status: str  # "target" | "stop" | "expired" | "open"
    exit_date: date | None
    exit_price: float | None
    sessions: int  # sessions walked; days_held when the idea exited
    max_favorable_pct: float | None
    max_adverse_pct: float | None
    note: str | None


# ----------------------------------------------------------------------------- pure rules


def fill_decision(idea: Idea, session_open: float) -> Fill:
    """The entry rule: fill at the open inside the zone, otherwise invalidated."""
    open_ = finite(session_open)
    if open_ is None or open_ <= 0.0:
        raise ValueError(f"session open is not a positive number: {session_open!r}")
    if idea.entry_low <= open_ <= idea.entry_high:
        return Fill("open", open_, None)
    side = "above" if open_ > idea.entry_high else "below"
    return Fill(
        "invalidated",
        None,
        f"open {open_:.2f} gapped {side} the entry zone "
        f"[{idea.entry_low:.2f}, {idea.entry_high:.2f}] on {idea.session_date.isoformat()}",
    )


def _as_date(x: object) -> date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(str(x)).date()


def session_bars(bars: pd.DataFrame, start: date | None = None) -> list[SessionBar]:
    """The OHLC rows of `bars` as sessions sorted by date, from `start` on.

    Accepts `market_data.daily_frame` output (a datetime index) or a frame with a `date`
    / `trade_date` column. Raises ValueError for a missing price column or a bar with a
    non-finite open, high, low or close (naming its date): a hole in the data is not a
    price.
    """
    if bars is None or len(bars) == 0:
        return []
    df = bars
    if "date" in df.columns:
        dates = df["date"]
    elif "trade_date" in df.columns:
        dates = df["trade_date"]
    else:
        dates = pd.Series(df.index, index=df.index)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {', '.join(missing)}")
    out: list[SessionBar] = []
    for d_raw, o, h, lo, c in zip(
        dates.tolist(),
        df["open"].tolist(),
        df["high"].tolist(),
        df["low"].tolist(),
        df["close"].tolist(),
        strict=True,
    ):
        d = _as_date(d_raw)
        if start is not None and d < start:
            continue
        prices = [finite(o), finite(h), finite(lo), finite(c)]
        if any(p is None or p <= 0.0 for p in prices):
            raise ValueError(f"bar for {d.isoformat()} has a missing or non-positive price")
        o_, h_, l_, c_ = (p for p in prices if p is not None)
        out.append(SessionBar(d, o_, h_, l_, c_))
    out.sort(key=lambda b: b.trade_date)
    return out


def walk_exit(
    entry_price: float,
    stop: float,
    target: float,
    bars: Sequence[SessionBar],
    horizon_days: int,
) -> Walk:
    """Walk the sessions from the entry session on and apply the exit rule (module doc).

    `bars` are the sessions starting with the entry session, in order. The walk covers at
    most `horizon_days` of them; MFE / MAE are the best high and worst low seen relative
    to the entry, over the sessions walked (the exit session included).
    """
    entry = finite(entry_price)
    if entry is None or entry <= 0.0:
        raise ValueError(f"entry_price is not a positive number: {entry_price!r}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days!r}")
    if not stop < entry < target:
        raise ValueError(f"need stop {stop} < entry {entry} < target {target}")
    window = list(bars)[:horizon_days]
    if not window:
        return Walk("open", None, None, 0, None, None, f"0 of {horizon_days} sessions elapsed")
    best_high = window[0].high
    worst_low = window[0].low
    for k, bar in enumerate(window, start=1):
        best_high = max(best_high, bar.high)
        worst_low = min(worst_low, bar.low)
        mfe = pct_change(best_high, entry)
        mae = pct_change(worst_low, entry)
        stop_hit = bar.low <= stop
        target_hit = bar.high >= target
        if stop_hit:
            gapped = bar.open <= stop
            exit_price = bar.open if gapped else stop
            why = "gapped below the stop: exit at the open" if gapped else "stop hit"
            if target_hit:
                why += "; target touched the same session (stop counted, conservative rule)"
            return Walk("stop", bar.trade_date, exit_price, k, mfe, mae, why)
        if target_hit:
            gapped = bar.open >= target
            exit_price = bar.open if gapped else target
            why = "gapped through the target: exit at the open" if gapped else "target hit"
            return Walk("target", bar.trade_date, exit_price, k, mfe, mae, why)
    n = len(window)
    if n >= horizon_days:
        last = window[-1]
        return Walk(
            "expired",
            last.trade_date,
            last.close,
            n,
            mfe,
            mae,
            f"neither target nor stop in {horizon_days} sessions: exit at the close",
        )
    return Walk("open", None, None, n, mfe, mae, f"{n} of {horizon_days} sessions elapsed")


def returns_for(
    entry_price: float, exit_price: float, stop: float, costs: CostModel
) -> dict[str, float]:
    """gross %, costs %, net % and R-multiple of a completed long trade."""
    gross = pct_change(exit_price, entry_price)
    cost_pct = costs.round_trip_pct()
    net = gross - cost_pct
    risk_pct = (entry_price - stop) / entry_price * 100.0  # planned risk from the entry, > 0
    return {
        "gross_return_pct": gross,
        "costs_pct": cost_pct,
        "net_return_pct": net,
        "r_multiple": net / risk_pct,
    }


# ----------------------------------------------------------------------------- storage


def outcome_of(db: Session, idea: Idea | int) -> IdeaOutcome | None:
    idea_id = idea if isinstance(idea, int) else idea.id
    return db.scalars(select(IdeaOutcome).where(IdeaOutcome.idea_id == idea_id)).first()


def outcomes_for(db: Session, ideas: Sequence[Idea]) -> dict[int, IdeaOutcome]:
    """Outcome rows for many ideas at once, keyed by idea id (ideas without one are absent)."""
    ids = [i.id for i in ideas]
    if not ids:
        return {}
    rows = db.scalars(select(IdeaOutcome).where(IdeaOutcome.idea_id.in_(ids))).all()
    return {o.idea_id: o for o in rows}


def _values(**given: Any) -> dict[str, Any]:
    """A full outcome row: every column, unspecified ones None."""
    values: dict[str, Any] = dict.fromkeys(_OUTCOME_COLUMNS)
    for k, v in given.items():
        if k not in values:
            raise KeyError(k)
        values[k] = finite(v) if isinstance(v, float) else v
    return values


def _upsert(db: Session, idea_id: int, values: dict[str, Any]) -> IdeaOutcome:
    stmt = insert(IdeaOutcome).values(idea_id=idea_id, **values)
    stmt = stmt.on_conflict_do_update(constraint="uq_idea_outcome_idea", set_=values)
    db.execute(stmt)
    db.flush()
    return db.execute(
        select(IdeaOutcome)
        .where(IdeaOutcome.idea_id == idea_id)
        .execution_options(populate_existing=True)
    ).scalar_one()


def status_of(existing: IdeaOutcome | None) -> str:
    return "pending" if existing is None else existing.status


# ----------------------------------------------------------------------------- the jobs' calls


def open_check_fill(
    db: Session, idea: Idea, session_open: float, *, now: datetime, source: str
) -> IdeaOutcome:
    """09:20: fill a pending idea at its session's open, or invalidate it (entry rule).

    Idempotent: an idea already graded (target / stop / expired) or voided is returned as
    it is; one already filled or invalidated with the same open is returned unchanged; a
    different open (a corrected print) re-applies the rule.
    """
    existing = outcome_of(db, idea.id)
    status = status_of(existing)
    if status in TERMINAL_STATUSES or status == "void":
        assert existing is not None
        return existing
    fill = fill_decision(idea, session_open)
    if (
        existing is not None
        and existing.status == fill.status
        and existing.entry_price == fill.entry_price
    ):
        return existing
    values = _values(
        status=fill.status,
        entry_date=idea.session_date if fill.status == "open" else None,
        entry_price=fill.entry_price,
        graded_at=now,
        source=source,
        notes=fill.note,
    )
    return _upsert(db, idea.id, values)


def grade_idea(
    db: Session,
    idea: Idea,
    bars_after_entry: pd.DataFrame,
    *,
    now: datetime,
    source: str,
    settings: Settings | None = None,
) -> IdeaOutcome:
    """15:45: walk the sessions since entry and settle the idea (exit rule, module doc).

    `bars_after_entry` holds the daily bars of the idea's instrument from its
    `session_date` on (`market_data.daily_frame(db, instrument_id, start=session_date)`;
    earlier rows are ignored). A pending idea is first filled from the entry session's
    open with the entry rule; without that session's bar nothing can be said and
    ValueError is raised. An invalidated or voided idea is not graded (ValueError).
    """
    existing = outcome_of(db, idea.id)
    status = status_of(existing)
    if status in UNGRADEABLE_STATUSES:
        raise ValueError(f"idea seq {idea.seq} is {status}: nothing to grade")
    s = settings or get_settings()
    bars = session_bars(bars_after_entry, idea.session_date)

    if existing is None or existing.entry_price is None or existing.entry_date is None:
        if not bars or bars[0].trade_date != idea.session_date:
            raise ValueError(
                f"idea seq {idea.seq} is pending and there is no bar for its entry session "
                f"{idea.session_date.isoformat()}: cannot fill it"
            )
        fill = fill_decision(idea, bars[0].open)
        if fill.status == "invalidated":
            return _upsert(
                db,
                idea.id,
                _values(status="invalidated", graded_at=now, source=source, notes=fill.note),
            )
        assert fill.entry_price is not None
        entry_date, entry_price = idea.session_date, fill.entry_price
    else:
        entry_date, entry_price = existing.entry_date, existing.entry_price
        bars = [b for b in bars if b.trade_date >= entry_date]

    walk = walk_exit(entry_price, idea.stop, idea.target_1, bars, idea.horizon_days)
    common = {
        "entry_date": entry_date,
        "entry_price": entry_price,
        "max_favorable_pct": walk.max_favorable_pct,
        "max_adverse_pct": walk.max_adverse_pct,
        "graded_at": now,
        "source": source,
        "notes": walk.note,
    }
    if walk.status == "open":
        return _upsert(db, idea.id, _values(status="open", **common))
    assert walk.exit_price is not None
    rets = returns_for(entry_price, walk.exit_price, idea.stop, CostModel.from_settings(s))
    values = _values(
        status=walk.status,
        exit_date=walk.exit_date,
        exit_price=walk.exit_price,
        days_held=walk.sessions,
        **rets,
        **common,
    )
    return _upsert(db, idea.id, values)


def void_idea(
    db: Session,
    idea: Idea,
    *,
    now: datetime,
    reason: str,
    category: str = "no_session",
    source: str = VOID_SOURCE,
) -> IdeaOutcome:
    """Record that an idea can never be graded (`no_session`: its session never traded;
    `data_error`: the bars are unusable). The entry, if there was one, is kept for the
    history; nothing else is invented. Refuses to void a graded idea, and needs a reason.
    """
    if category not in VOID_CATEGORIES:
        raise ValueError(f"category must be one of {', '.join(VOID_CATEGORIES)}: {category!r}")
    if not reason or not reason.strip():
        raise ValueError("a reason is required to void an idea")
    existing = outcome_of(db, idea.id)
    if existing is not None and existing.status in TERMINAL_STATUSES:
        raise ValueError(
            f"idea seq {idea.seq} is already graded ({existing.status}); nothing to void"
        )
    values = _values(
        status="void",
        entry_date=existing.entry_date if existing is not None else None,
        entry_price=existing.entry_price if existing is not None else None,
        graded_at=now,
        source=source,
        notes=f"voided ({category}): {reason.strip()}",
        error_category=category,
    )
    return _upsert(db, idea.id, values)

"""Stock ideas (brief §4.2, §7.3): the ledger's ideas with their outcomes, and one idea's
full reasoning.

Everything is read from the idea ledger (`app.services.ledger`), `idea_outcome`, the paper
book and the ideas scorecard; the API never imports LightGBM. Honesty rules: every
`p_target` is shown with the live track record of its setup and horizon (`track`), a
missing number is None (the UI shows "—"), and surveillance (ASM / GSM / trade-to-trade)
is never checked because no source is reachable from the job host, so every idea carries
`surveillance_unknown=True` and `SURVEILLANCE_NOTE` says why. Lists leave the two big
JSON columns (`feature_snapshot`, `drivers`) unread; the detail view reads them and turns
them into the full factor list (`factors_out`): the top `drivers_top_n` drivers, then
every feature of the snapshot with its contribution, None where the ledger recorded none.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session, defer

from app.api.deps import current_user
from app.config import Settings, get_settings
from app.db import get_db
from app.ml import ideas_models
from app.ml.core import IDEA_STATUSES, finite, horizons
from app.ml.strategies import SETUP_LABELS, SETUPS
from app.models import AppUser, Idea, IdeaOutcome, ModelVersion
from app.schemas import (
    DriverOut,
    FactorOut,
    FactorsOut,
    IdeaDetailOut,
    IdeaOut,
    IdeaOutcomeOut,
    IdeasListOut,
    IdeaTrackOut,
    PaperTradeOut,
)
from app.services import paper
from app.services.ideas_scorecard import IdeaTrack, idea_track
from app.services.ledger import not_superseded_idea

router = APIRouter(prefix="/api/ideas", tags=["ideas"])

SURVEILLANCE_NOTE = (
    "Surveillance not checked: the NSE ASM / GSM / trade-to-trade lists are unreachable from "
    "the job host, so no idea has been screened for surveillance measures. Check the exchange "
    "before acting on one."
)
# Status filters: every outcome status plus two groups the Ideas page offers.
STATUS_GROUPS: dict[str, tuple[str, ...]] = {
    "active": ("pending", "open"),
    "graded": ("target", "stop", "expired"),
}
STATUS_FILTERS: tuple[str, ...] = (*IDEA_STATUSES, *STATUS_GROUPS)
SETUP_WORDS: dict[str, str] = {
    "breakout": "Breakout",
    "momentum": "Momentum",
    "mean_reversion": "Mean reversion",
    "pullback": "Pullback",
}
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


# ----------------------------------------------------------------------------- resolvers


def resolve_horizon(horizon: str | None, settings: Settings | None = None) -> str | None:
    """The requested horizon (lower-cased), None for "all"; 404 when it is not a horizon."""
    if horizon is None or not horizon.strip():
        return None
    wanted = horizon.strip().lower()
    known = list(horizons(settings))
    if wanted not in known:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown horizon {wanted!r}; known horizons: {', '.join(known)}.",
        )
    return wanted


def resolve_status(value: str | None) -> str | None:
    """A status filter (`STATUS_FILTERS`), None for "all"; 422 for anything else."""
    if value is None or not value.strip():
        return None
    wanted = value.strip().lower()
    if wanted not in STATUS_FILTERS:
        raise HTTPException(
            422,
            f"Unknown status {wanted!r}; one of {', '.join(STATUS_FILTERS)}.",
        )
    return wanted


def resolve_setup(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip().lower()


def known_setups(db: Session) -> list[str]:
    """The strategies' setups in their order, then any other setup found in the ledger."""
    out = list(dict.fromkeys(SETUPS.values()))
    stored = db.scalars(select(Idea.setup).distinct().order_by(Idea.setup)).all()
    out.extend(s for s in stored if s not in out)
    return out


def setup_label(setup: str) -> str:
    return SETUP_WORDS.get(setup, setup.replace("_", " ").capitalize())


def strategy_label(strategy: str) -> str:
    return SETUP_LABELS.get(strategy, strategy.replace("_", " "))


# ----------------------------------------------------------------------------- rows -> schemas


def outcome_out(o: IdeaOutcome | None) -> IdeaOutcomeOut:
    """The outcome row as the API shows it; a pending idea has every field None."""
    if o is None:
        return IdeaOutcomeOut(
            status="pending",
            entry_date=None,
            entry_price=None,
            exit_date=None,
            exit_price=None,
            gross_return_pct=None,
            costs_pct=None,
            net_return_pct=None,
            r_multiple=None,
            days_held=None,
            max_favorable_pct=None,
            max_adverse_pct=None,
            graded_at=None,
            source=None,
            notes=None,
            error_category=None,
        )
    return IdeaOutcomeOut(
        status=o.status,
        entry_date=o.entry_date,
        entry_price=o.entry_price,
        exit_date=o.exit_date,
        exit_price=o.exit_price,
        gross_return_pct=o.gross_return_pct,
        costs_pct=o.costs_pct,
        net_return_pct=o.net_return_pct,
        r_multiple=o.r_multiple,
        days_held=o.days_held,
        max_favorable_pct=o.max_favorable_pct,
        max_adverse_pct=o.max_adverse_pct,
        graded_at=o.graded_at,
        source=o.source,
        notes=o.notes,
        error_category=o.error_category,
    )


def track_out(t: IdeaTrack, settings: Settings) -> IdeaTrackOut:
    return IdeaTrackOut(
        horizon=t.horizon,
        setup=t.setup,
        n_graded=t.n_graded,
        hit_rate=t.hit_rate,
        win_rate=t.win_rate,
        expectancy_pct=t.expectancy_pct,
        sample_warning=t.sample_warning,
        min_sample_size=settings.min_sample_size,
    )


def tracks_for(
    db: Session, ideas: Iterable[Idea], settings: Settings
) -> dict[tuple[str, str], IdeaTrackOut]:
    """One live track record per (horizon, setup) present in `ideas` (a few queries, not
    one per row)."""
    pairs = sorted({(i.horizon, i.setup) for i in ideas})
    return {(h, s): track_out(idea_track(db, h, s, settings=settings), settings) for h, s in pairs}


def versions_for(db: Session, ideas: Iterable[Idea]) -> dict[int, str]:
    ids = sorted({i.model_version_id for i in ideas if i.model_version_id is not None})
    if not ids:
        return {}
    rows = db.execute(
        select(ModelVersion.id, ModelVersion.version).where(ModelVersion.id.in_(ids))
    ).all()
    return {int(mv_id): str(version) for mv_id, version in rows}


def corrections_for(db: Session, ideas: Iterable[Idea]) -> dict[int, int]:
    """idea id -> seq of the later record that corrects it (only for superseded ideas)."""
    ids = [i.id for i in ideas]
    if not ids:
        return {}
    rows = db.execute(
        select(Idea.corrects_idea_id, Idea.seq).where(Idea.corrects_idea_id.in_(ids))
    ).all()
    return {int(original): int(seq) for original, seq in rows if original is not None}


def idea_out(
    i: Idea,
    o: IdeaOutcome | None,
    *,
    track: IdeaTrackOut,
    model_version: str | None,
    corrected_by_seq: int | None,
    is_previous: bool = False,
) -> IdeaOut:
    return IdeaOut(
        id=i.id,
        seq=i.seq,
        session_date=i.session_date,
        as_of=i.as_of,
        is_late=i.is_late,
        instrument_id=i.instrument_id,
        symbol=i.symbol,
        exchange=i.exchange,
        horizon=i.horizon,
        horizon_days=i.horizon_days,
        setup=i.setup,
        setup_label=setup_label(i.setup),
        strategy=i.strategy,
        strategy_label=strategy_label(i.strategy),
        direction=i.direction,
        entry_low=i.entry_low,
        entry_high=i.entry_high,
        stop=i.stop,
        target_1=i.target_1,
        target_2=i.target_2,
        p_target=i.p_target,
        expected_return_low_pct=i.expected_return_low_pct,
        expected_return_high_pct=i.expected_return_high_pct,
        risk_reward=i.risk_reward,
        ev_pct_after_costs=i.ev_pct_after_costs,
        position_qty=i.position_qty,
        position_value=i.position_value,
        risk_amount=i.risk_amount,
        rank=i.rank,
        surveillance_unknown=i.surveillance_unknown,
        surveillance_note=SURVEILLANCE_NOTE,
        model_version_id=i.model_version_id,
        model_version=model_version,
        corrects_idea_id=i.corrects_idea_id,
        corrected_by_seq=corrected_by_seq,
        note=i.note,
        outcome=outcome_out(o),
        track=track,
        is_previous=is_previous,
    )


def ideas_out(
    db: Session,
    rows: Sequence[tuple[Idea, IdeaOutcome | None]],
    settings: Settings,
    *,
    is_previous: bool = False,
) -> list[IdeaOut]:
    """`IdeaOut` for every (idea, outcome) pair, with the lookups batched."""
    ideas = [i for i, _ in rows]
    tracks = tracks_for(db, ideas, settings)
    versions = versions_for(db, ideas)
    corrections = corrections_for(db, ideas)
    return [
        idea_out(
            i,
            o,
            track=tracks[(i.horizon, i.setup)],
            model_version=versions.get(i.model_version_id) if i.model_version_id else None,
            corrected_by_seq=corrections.get(i.id),
            is_previous=is_previous,
        )
        for i, o in rows
    ]


# ----------------------------------------------------------------------------- factors


def _driver_sort_key(d: DriverOut) -> tuple[bool, float]:
    """Largest |contribution| first; drivers without a contribution last."""
    return (d.contribution is None, -abs(d.contribution) if d.contribution is not None else 0.0)


def factors_out(
    drivers: object,
    snapshot: object,
    label_of: Callable[[str], str],
    top_n: int,
) -> FactorsOut:
    """The full factor list behind a record from its stored drivers and feature snapshot.

    `drivers` is the ledger's JSON list of driver dicts (whatever the engine recorded:
    the top few or every non-zero contribution), `snapshot` the feature row. The drivers
    come first, largest |contribution| first, then every other feature of the snapshot in
    its stored order with `contribution=None`: a contribution the ledger did not record is
    not invented. `drivers` in the result is the top `top_n` of that ranking.
    """
    stored: list[DriverOut] = []
    if isinstance(drivers, Sequence) and not isinstance(drivers, str | bytes):
        for d in drivers:
            if isinstance(d, Mapping) and "feature" in d:
                stored.append(
                    DriverOut(
                        feature=str(d.get("feature")),
                        value=finite(d.get("value")),
                        contribution=finite(d.get("contribution")),
                        text=str(d.get("text") or ""),
                    )
                )
    ranked = sorted(stored, key=_driver_sort_key)
    snap: dict[str, float | None] = {}
    if isinstance(snapshot, Mapping):
        snap = {str(k): finite(v) for k, v in snapshot.items()}
    factors: list[FactorOut] = []
    seen: set[str] = set()
    for d in ranked:
        if d.feature in seen:
            continue
        seen.add(d.feature)
        factors.append(
            FactorOut(
                feature=d.feature,
                label=label_of(d.feature),
                value=d.value if d.value is not None else snap.get(d.feature),
                contribution=d.contribution,
                text=d.text or None,
            )
        )
    for name, value in snap.items():
        if name in seen:
            continue
        factors.append(
            FactorOut(feature=name, label=label_of(name), value=value, contribution=None, text=None)
        )
    top = max(0, int(top_n))
    return FactorsOut(
        drivers=ranked[:top],
        all_factors=factors,
        feature_snapshot=snap,
        n_features=len(factors),
        n_with_contribution=sum(1 for f in factors if f.contribution is not None),
        drivers_top_n=top,
    )


def idea_factors(i: Idea, settings: Settings) -> FactorsOut:
    """An idea's factors, labelled with the ideas feature pipeline's names."""
    return factors_out(
        i.drivers, i.feature_snapshot, ideas_models.feature_label, settings.drivers_top_n
    )


# ----------------------------------------------------------------------------- queries


def ideas_query(
    *, status_filter: str | None = None, horizon: str | None = None, setup: str | None = None
) -> Select[Any]:
    """Current (non-superseded) ideas with their outcome (None while pending), newest
    session first, without the feature snapshot or the drivers."""
    stmt = (
        select(Idea, IdeaOutcome)
        .options(defer(Idea.feature_snapshot), defer(Idea.drivers))
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .where(not_superseded_idea())
    )
    if horizon is not None:
        stmt = stmt.where(Idea.horizon == horizon)
    if setup is not None:
        stmt = stmt.where(Idea.setup == setup)
    if status_filter is not None:
        wanted = STATUS_GROUPS.get(status_filter, (status_filter,))
        if "pending" in wanted:
            stmt = stmt.where(or_(IdeaOutcome.id.is_(None), IdeaOutcome.status.in_(wanted)))
        else:
            stmt = stmt.where(IdeaOutcome.status.in_(wanted))
    return stmt.order_by(Idea.session_date.desc(), Idea.seq.desc())


def ideas_rows(db: Session, stmt: Select[Any]) -> list[tuple[Idea, IdeaOutcome | None]]:
    return [(i, o) for i, o in db.execute(stmt).all()]


# ----------------------------------------------------------------------------- endpoints


@router.get("", response_model=IdeasListOut)
def list_ideas(
    status: str | None = Query(default=None, max_length=16),
    horizon: str | None = Query(default=None, max_length=16),
    setup: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> IdeasListOut:
    """Current ideas, newest session first, filtered by outcome status, horizon and setup.

    Superseded ideas (ones a correction points at) are left out; the ledger itself keeps
    them. An unknown horizon is a 404, an unknown status a 422.
    """
    s = get_settings()
    h = resolve_horizon(horizon, s)
    st = resolve_status(status)
    su = resolve_setup(setup)
    rows = ideas_rows(db, ideas_query(status_filter=st, horizon=h, setup=su).limit(limit))
    return IdeasListOut(
        ideas=ideas_out(db, rows, s),
        n=len(rows),
        status=st,
        horizon=h,
        setup=su,
        limit=limit,
        statuses=list(STATUS_FILTERS),
        horizons=list(horizons(s)),
        setups=known_setups(db),
        min_sample_size=s.min_sample_size,
        surveillance_note=SURVEILLANCE_NOTE,
    )


@router.get("/{seq}", response_model=IdeaDetailOut)
def idea_detail(
    seq: int,
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> IdeaDetailOut:
    """One idea by ledger `seq` with its full reasoning: reasons, invalidation, every
    factor (drivers + feature snapshot), outcome, paper trade and track record. A
    superseded idea is still returned, with `corrected_by_seq` pointing at its correction."""
    s = get_settings()
    row = db.execute(
        select(Idea, IdeaOutcome)
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .where(Idea.seq == seq)
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No idea with seq {seq} in the ledger.")
    i, o = row
    [base] = ideas_out(db, [(i, o)], s)
    trade = paper.trade_for_idea(db, i)
    paper_note = None
    if trade is None and o is not None and o.entry_price is not None and o.status != "void":
        paper_note = (
            "Filled, but the paper book had no cash left that day: no paper position was "
            "opened (the idea is still graded against its levels)."
        )
    return IdeaDetailOut(
        **base.model_dump(),
        reasons=[str(r) for r in (i.reasons or [])],
        invalidation=i.invalidation,
        factors=idea_factors(i, s),
        paper_trade=PaperTradeOut.model_validate(paper.trade_dict(trade)) if trade else None,
        paper_note=paper_note,
        hash=i.hash,
        prev_hash=i.prev_hash,
    )

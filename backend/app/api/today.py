"""Today page data: market clock, the market call per target, global cues, flows, Kite.

The market call shown is the ledger's current call for the session `core.session_for(now)`
is heading into. Once the market closes (15:30 IST) that session moves to the next trading
day, which has no call until the 07:30 job runs; the page then falls back to the latest
earlier call for the target, flagged `is_previous`, so its 15:45 close grade is visible
rather than an empty card. The "no call" note is only for a target with no call at all.

The stock ideas follow the same rule (`today_ideas`): the current ideas of that session per
horizon, else the latest earlier session's ideas with their outcomes, flagged `is_previous`.
A horizon without an idea says why in words: "No edge today" only when the engine ran for
that session (an ideas champion exists and the session's market call is in the ledger: the
07:30 job generates ideas right after the calls) and recorded none; otherwise that there is
no ideas model yet, or that the job has not run. Nothing is ever shown that is not in the
ledger. `/api/today/market-call/{id}/factors` serves a call's full factor list (every
feature of its snapshot with the recorded contribution) for the expandable "All factors".
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, defer

from app.api.deps import current_user, kv_dep
from app.api.ideas import SURVEILLANCE_NOTE, factors_out, ideas_out
from app.api.kite import kite_status
from app.cache import KV
from app.config import IST, Settings, get_settings
from app.db import get_db
from app.jobs.definitions import get_spec
from app.jobs.runner import run_job
from app.ml import ideas_models, models, ranking
from app.ml.core import MODEL_NAME, horizons, session_for
from app.models import AppUser, Idea, IdeaOutcome, ModelVersion, Prediction, PredictionGrade
from app.schemas import (
    CueOut,
    DriverOut,
    FactorsOut,
    FlowDayOut,
    FlowsOut,
    GradeOut,
    HorizonIdeasOut,
    IdeaOut,
    MarketCallOut,
    MarketClockOut,
    TodayIdeasOut,
    TodayOut,
    TrackOut,
)
from app.services import flows, global_cues
from app.services.calendar import TradingCalendar
from app.services.ledger import not_superseded_idea
from app.services.scorecard import MIN_PAIRED_N, Scorecard, not_superseded, scorecard

router = APIRouter(prefix="/api/today", tags=["today"])


def track_out(sc: Scorecard, settings: Settings) -> TrackOut:
    """The one-line track record shown beside a call, from the full (paired) scorecard."""
    best = sc.best_baseline
    return TrackOut(
        n_graded=sc.n_graded,
        accuracy=sc.accuracy,
        best_baseline=best["name"] if best else None,
        best_baseline_label=best["label"] if best else None,
        best_baseline_accuracy=best["accuracy"] if best else None,
        paired_n=best["n"] if best else None,
        model_paired_accuracy=best["model_accuracy"] if best else None,
        min_paired_n=MIN_PAIRED_N,
        comparison=sc.comparison,
        sample_warning=sc.sample_warning,
        min_sample_size=settings.min_sample_size,
    )


def _calls(target: str) -> Select[tuple[Prediction]]:
    """Current (non-superseded) market calls for a target, without the feature snapshot.

    The card shows the drivers but never the snapshot (the full feature row, the largest
    column of the ledger), so that one is deferred.
    """
    return (
        select(Prediction)
        .options(defer(Prediction.feature_snapshot))
        .where(Prediction.target == target, Prediction.kind == MODEL_NAME, not_superseded())
    )


def current_call(db: Session, target: str, session: date) -> Prediction | None:
    """The ledger's current call for a target and session (same rule as `ledger.latest_call`)."""
    stmt = _calls(target).where(Prediction.session_date == session)
    return db.scalars(stmt.order_by(Prediction.seq.desc()).limit(1)).first()


def previous_call(db: Session, target: str, through: date) -> Prediction | None:
    """The latest recorded call for a target with a session on or before `through`.

    The fallback when the upcoming session has no call yet: usually the session that just
    closed, with its grade.
    """
    stmt = _calls(target).where(Prediction.session_date <= through)
    order = (Prediction.session_date.desc(), Prediction.seq.desc())
    return db.scalars(stmt.order_by(*order).limit(1)).first()


def market_call_out(
    db: Session, p: Prediction, settings: Settings, *, is_previous: bool = False
) -> MarketCallOut:
    """One ledger record as the Today card shows it: call + grade (if any) + track record."""
    grade = db.scalars(select(PredictionGrade).where(PredictionGrade.prediction_id == p.id)).first()
    version = (
        db.scalar(select(ModelVersion.version).where(ModelVersion.id == p.model_version_id))
        if p.model_version_id is not None
        else None
    )
    return MarketCallOut(
        id=p.id,
        seq=p.seq,
        target=p.target,
        session_date=p.session_date,
        as_of=p.as_of,
        is_late=p.is_late,
        reference_date=p.reference_date,
        reference_close=p.reference_close,
        predicted_class=p.predicted_class,
        probs={"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down},
        range_p10_pct=p.range_p10_pct,
        range_p90_pct=p.range_p90_pct,
        gap_call=p.gap_call,
        gap_expected_pct=p.gap_expected_pct,
        confidence=p.confidence,
        no_edge=p.no_edge,
        drivers=[DriverOut.model_validate(d) for d in (p.drivers or [])],
        timing_guidance=p.timing_guidance,
        baselines=dict(p.baselines or {}),
        model_version=version,
        note=p.note,
        grade=(
            GradeOut(
                actual_open=grade.actual_open,
                actual_close=grade.actual_close,
                actual_pct=grade.actual_pct,
                actual_class=grade.actual_class,
                correct=grade.correct,
                brier=grade.brier,
                log_loss=grade.log_loss,
                gap_actual_pct=grade.gap_actual_pct,
                gap_correct=grade.gap_correct,
                graded_at=grade.graded_at,
            )
            if grade is not None
            else None
        ),
        track=track_out(scorecard(db, p.target, settings=settings), settings),
        is_previous=is_previous,
    )


def market_calls(
    db: Session, cal: TradingCalendar, now: datetime, settings: Settings
) -> tuple[list[MarketCallOut], str, date]:
    """The call to show for every configured target, the note, and the session it is for.

    A target with a call for the upcoming session shows that call; one without shows its
    latest earlier call (session on or before the IST date of `now`) flagged `is_previous`,
    so the close grade stays on the page after 15:30 IST. The note names only the targets
    with no call at all, and is empty when every target has one.
    """
    session = session_for(cal, now)
    today = now.astimezone(IST).date()
    calls: list[MarketCallOut] = []
    missing: list[str] = []
    for target in settings.prediction_target_symbols:
        p = current_call(db, target, session)
        is_previous = False
        if p is None:
            p = previous_call(db, target, today)
            is_previous = p is not None
        if p is None:
            missing.append(target)
        else:
            calls.append(market_call_out(db, p, settings, is_previous=is_previous))
    if not calls:
        note = (
            f"No call recorded for {session.isoformat()}: the 07:30 pre-market job hasn't run "
            "yet, or there is no trained model (run `python -m app.cli train`)."
        )
    elif missing:
        note = f"No call in the ledger for {', '.join(missing)} (none for any session yet)."
    else:
        note = ""
    return calls, note, session


# ----------------------------------------------------------------------------- ideas


def ideas_for(db: Session, session: date) -> list[tuple[Idea, IdeaOutcome | None]]:
    """The current (non-superseded) ideas of a session with their outcomes, by horizon and
    rank, without the feature snapshot or the drivers."""
    stmt = (
        select(Idea, IdeaOutcome)
        .options(defer(Idea.feature_snapshot), defer(Idea.drivers))
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .where(Idea.session_date == session, not_superseded_idea())
        .order_by(Idea.horizon.asc(), Idea.rank.asc(), Idea.seq.asc())
    )
    return [(i, o) for i, o in db.execute(stmt).all()]


def latest_ideas_session(db: Session, through: date) -> date | None:
    """The latest session on or before `through` with an idea in the ledger."""
    return db.scalar(select(func.max(Idea.session_date)).where(Idea.session_date <= through))


def call_recorded(db: Session, session: date, settings: Settings) -> bool:
    """Whether the 07:30 job recorded a market call for the session (it generates the
    ideas right after the calls, so a recorded call means the ideas engine ran)."""
    return any(
        current_call(db, target, session) is not None
        for target in settings.prediction_target_symbols
    )


def horizon_note(
    horizon: str, session: date, *, has_ideas: bool, has_model: bool, ran: bool, settings: Settings
) -> tuple[bool, str | None]:
    """(no_edge, note) for one horizon of the Today page (module doc)."""
    if has_ideas:
        return False, None
    if not has_model:
        return False, (
            f"No ideas model for {horizon} yet: train one with `python -m app.cli train-ideas`. "
            f"Until then no {horizon} idea can be generated."
        )
    if ran:
        return True, ranking.no_edge_text(horizon, settings)
    return False, (
        f"No {horizon} idea recorded for {session.isoformat()}: the 07:30 IST pre-market job "
        "hasn't run yet (ideas are generated right after the market calls)."
    )


def today_ideas(
    db: Session, cal: TradingCalendar, now: datetime, settings: Settings
) -> TodayIdeasOut:
    """The ideas to show per horizon, the session they are for, and the notes (module doc)."""
    session = session_for(cal, now)
    today = now.astimezone(IST).date()
    shown = session
    is_previous = False
    rows = ideas_for(db, session)
    ran = call_recorded(db, session, settings)
    if not rows and not ran:
        # The 07:30 run has not happened for this session: show the last session's ideas,
        # clearly marked. Once it has run, an empty session means "No edge today" and the
        # previous session's ideas are never shown in its place.
        previous = latest_ideas_session(db, today)
        if previous is not None:
            shown, is_previous, rows = previous, True, ideas_for(db, previous)
            ran = call_recorded(db, shown, settings)
    outs: list[IdeaOut] = ideas_out(db, rows, settings, is_previous=is_previous)
    groups: list[HorizonIdeasOut] = []
    any_model = False
    for name, spec in horizons(settings).items():
        mine = [o for o in outs if o.horizon == name]
        champion = ideas_models.ideas_champion_light(db, name, settings)
        any_model = any_model or champion is not None
        no_edge, note = horizon_note(
            name,
            shown,
            has_ideas=bool(mine),
            has_model=champion is not None,
            ran=ran,
            settings=settings,
        )
        groups.append(
            HorizonIdeasOut(
                horizon=name,
                horizon_days=spec.days,
                target_atr=spec.target_atr,
                stop_atr=spec.stop_atr,
                ideas=mine,
                no_edge=no_edge,
                note=note,
                champion_version=champion.version if champion is not None else None,
            )
        )
    if outs:
        note = ""
    elif not any_model:
        note = (
            "No ideas model yet: train the swing and positional models with "
            "`python -m app.cli train-ideas`. Until then no idea can be generated."
        )
    elif ran:
        note = f"{ranking.NO_EDGE_TEXT}: no candidate cleared the bar for any horizon."
    else:
        note = (
            f"No ideas recorded for {shown.isoformat()}: the 07:30 IST pre-market job hasn't "
            "run yet (ideas are generated right after the market calls)."
        )
    return TodayIdeasOut(
        session=shown,
        is_previous=is_previous,
        by_horizon=groups,
        n_ideas=len(outs),
        note=note,
        surveillance_note=SURVEILLANCE_NOTE,
        min_p_target=settings.ideas_min_p_target,
        min_ev_pct=settings.ideas_min_ev_pct,
        max_per_horizon=settings.ideas_max_per_horizon,
    )


@router.get("/market-call/{prediction_id}/factors", response_model=FactorsOut)
def market_call_factors(
    prediction_id: int, db: Session = Depends(get_db), _: AppUser = Depends(current_user)
) -> FactorsOut:
    """A market call's full factor list: its recorded drivers (largest |contribution|
    first, the top `drivers_top_n` as `drivers`) and every feature of its snapshot, with
    `contribution=None` where the ledger holds no contribution. 404 for an unknown call."""
    p = db.get(Prediction, prediction_id)
    if p is None or p.kind != MODEL_NAME:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No market call with id {prediction_id}.")
    return factors_out(
        p.drivers, p.feature_snapshot, models.feature_label, get_settings().drivers_top_n
    )


@router.get("", response_model=TodayOut)
def today(db: Session = Depends(get_db), _: AppUser = Depends(current_user)) -> TodayOut:
    settings = get_settings()
    cal = TradingCalendar.load(db)
    now = datetime.now(UTC)
    clock = cal.clock(now)
    days = flows.recent(db, 10)
    expected = flows.expected_latest(cal)
    if not days:
        flow_status, note = "unavailable", "No FII/DII data stored yet."
    elif days[0].trade_date >= expected:
        flow_status, note = "ok", None
    else:
        flow_status, note = "stale", f"Latest FII/DII data is for {days[0].trade_date}."
    calls, call_note, call_session = market_calls(db, cal, now, settings)
    return TodayOut(
        clock=MarketClockOut(
            now_ist=clock.now_ist,
            session=clock.session,
            is_trading_day=clock.today.is_trading_day,
            day_reason=clock.today.reason,
            calendar_verified=clock.today.calendar_verified,
            next_trading_day=clock.next_trading_day,
            previous_trading_day=clock.previous_trading_day,
        ),
        cues=[CueOut(**asdict(c)) for c in global_cues.latest(db)],
        flows=FlowsOut(
            days=[FlowDayOut(**asdict(d)) for d in days],
            status=flow_status,
            expected_latest=expected,
            note=note,
        ),
        kite=kite_status(db),
        market_call_session=call_session,
        market_call=calls,
        market_call_note=call_note,
        ideas=today_ideas(db, cal, now, settings),
    )


@router.post("/refresh-cues", status_code=202)
def refresh_cues(r: KV = Depends(kv_dep), _: AppUser = Depends(current_user)) -> dict[str, str]:
    """Run the global-cue refresh now. Limited to once a minute."""
    if not r.set("ratelimit:manual_cues", "1", nx=True, ex=60):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Refreshed less than a minute ago.")
    return {"status": run_job(get_spec("global_cues_refresh"))}

"""Today page data: market clock, the market call per target, global cues, flows, Kite."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, kv_dep
from app.api.kite import kite_status
from app.cache import KV
from app.config import Settings, get_settings
from app.db import get_db
from app.jobs.definitions import get_spec
from app.jobs.runner import run_job
from app.ml.core import session_for
from app.models import AppUser, ModelVersion, Prediction, PredictionGrade
from app.schemas import (
    CueOut,
    DriverOut,
    FlowDayOut,
    FlowsOut,
    GradeOut,
    MarketCallOut,
    MarketClockOut,
    TodayOut,
    TrackOut,
)
from app.services import flows, global_cues, ledger
from app.services.calendar import TradingCalendar
from app.services.scorecard import Scorecard, scorecard

router = APIRouter(prefix="/api/today", tags=["today"])


def track_out(sc: Scorecard, settings: Settings) -> TrackOut:
    """The one-line track record shown beside a call, from the full scorecard."""
    with_calls = [b for b in sc.baselines if b.get("accuracy") is not None]
    best = max(with_calls, key=lambda b: (b["accuracy"], b["n"])) if with_calls else None
    return TrackOut(
        n_graded=sc.n_graded,
        accuracy=sc.accuracy,
        best_baseline=best["name"] if best else None,
        best_baseline_label=best["label"] if best else None,
        best_baseline_accuracy=best["accuracy"] if best else None,
        sample_warning=sc.sample_warning,
        min_sample_size=settings.min_sample_size,
    )


def market_call_out(db: Session, p: Prediction, settings: Settings) -> MarketCallOut:
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
    )


def market_calls(
    db: Session, cal: TradingCalendar, now: datetime, settings: Settings
) -> tuple[list[MarketCallOut], str]:
    """The ledger's current call for every configured target for the session `now` is in.

    The note names what is missing and why; it is empty when every target has a call.
    """
    session = session_for(cal, now)
    calls: list[MarketCallOut] = []
    missing: list[str] = []
    for target in settings.prediction_target_symbols:
        p = ledger.latest_call(db, target, session)
        if p is None:
            missing.append(target)
        else:
            calls.append(market_call_out(db, p, settings))
    if not calls:
        note = (
            f"No call recorded for {session.isoformat()}: the 07:30 pre-market job hasn't run "
            "yet, or there is no trained model (run `python -m app.cli train`)."
        )
    elif missing:
        note = f"No call recorded for {', '.join(missing)} for {session.isoformat()}."
    else:
        note = ""
    return calls, note


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
    calls, call_note = market_calls(db, cal, now, settings)
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
        market_call=calls,
        market_call_note=call_note,
    )


@router.post("/refresh-cues", status_code=202)
def refresh_cues(r: KV = Depends(kv_dep), _: AppUser = Depends(current_user)) -> dict[str, str]:
    """Run the global-cue refresh now. Limited to once a minute."""
    if not r.set("ratelimit:manual_cues", "1", nx=True, ex=60):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Refreshed less than a minute ago.")
    return {"status": run_job(get_spec("global_cues_refresh"))}

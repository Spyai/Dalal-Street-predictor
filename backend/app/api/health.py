"""Health: last run of every job, data freshness per source, champion models."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import redis
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import current_user, kv_dep
from app.cache import HEARTBEAT_KEY, KV
from app.config import IST, get_settings
from app.db import get_db
from app.jobs.definitions import job_specs
from app.ml.core import MODEL_NAME, finite, session_for
from app.models import (
    AppUser,
    DailyBar,
    FiiDiiFlow,
    GlobalCueSnapshot,
    Instrument,
    JobRun,
    ModelVersion,
    Prediction,
)
from app.schemas import FreshnessOut, HealthOut, JobHealthOut, ModelHealthOut, SourceHealthOut
from app.services import calendar, flows
from app.services.calendar import TradingCalendar
from app.services.scorecard import climatology_brier
from app.services.source_status import SOURCE_INFO, all_statuses

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz(db: Session = Depends(get_db)) -> dict[str, bool]:
    """Unauthenticated liveness probe for Docker. Reveals nothing."""
    db.execute(text("SELECT 1"))
    return {"ok": True}


def external_scheduler_alive(last_run: datetime | None, now: datetime) -> bool:
    """GitHub Actions has no heartbeat; judge it by the last job run.

    Cues run every 30 minutes from 05:30 to 23:30 IST, so a run older than 75 minutes
    in that window means the schedule stopped. Overnight there is a ~6 hour gap.
    """
    if last_run is None:
        return False
    local = now.astimezone(IST).time()
    limit = timedelta(minutes=75) if time(6, 45) <= local <= time(23, 59) else timedelta(hours=7)
    return now - last_run <= limit


def _jobs(db: Session) -> list[JobHealthOut]:
    out: list[JobHealthOut] = []
    for spec in job_specs():
        last = db.scalar(
            select(JobRun).where(JobRun.job_name == spec.name).order_by(JobRun.id.desc()).limit(1)
        )
        last_ok = db.scalar(
            select(func.max(JobRun.finished_at)).where(
                JobRun.job_name == spec.name, JobRun.status == "success"
            )
        )
        out.append(
            JobHealthOut(
                name=spec.name,
                label=spec.label,
                schedule=spec.schedule,
                phase=spec.phase,
                built=spec.fn is not None,
                last_status=last.status if last else None,
                last_run_at=last.started_at if last else None,
                last_finished_at=last.finished_at if last else None,
                last_message=last.message if last else None,
                last_success_at=last_ok,
            )
        )
    return out


def _freshness(db: Session, cal: TradingCalendar) -> list[FreshnessOut]:
    now = datetime.now(UTC)
    expected_session = cal.last_completed_session(now)
    out: list[FreshnessOut] = []

    cov = calendar.coverage(db)
    year = datetime.now(UTC).year
    out.append(
        FreshnessOut(
            dataset="NSE holiday calendar",
            latest=None,
            expected=None,
            status="ok" if cov.get(year) else "missing",
            detail=f"{cov.get(year, 0)} holidays stored for {year}",
        )
    )

    nifty = db.scalar(
        select(func.max(DailyBar.trade_date))
        .join(Instrument, Instrument.id == DailyBar.instrument_id)
        .where(Instrument.symbol == "NIFTY 50", Instrument.exchange == "INDEX")
    )
    out.append(_fresh("Nifty 50 daily bars", nifty, expected_session))

    uni_total = db.scalar(
        select(func.count()).select_from(Instrument).where(Instrument.in_universe)
    )
    uni_fresh = db.scalar(
        select(func.count(func.distinct(DailyBar.instrument_id)))
        .join(Instrument, Instrument.id == DailyBar.instrument_id)
        .where(Instrument.in_universe, DailyBar.trade_date >= expected_session)
    )
    uni_latest = db.scalar(
        select(func.max(DailyBar.trade_date))
        .join(Instrument, Instrument.id == DailyBar.instrument_id)
        .where(Instrument.in_universe)
    )
    f = _fresh("Universe daily bars", uni_latest, expected_session)
    f.detail = (
        f"{uni_fresh or 0}/{uni_total or 0} universe stocks have a bar for {expected_session}"
    )
    if uni_total and (uni_fresh or 0) < uni_total and f.status == "ok":
        f.status = "partial"
    out.append(f)

    fii = db.scalar(select(func.max(FiiDiiFlow.trade_date)))
    out.append(_fresh("FII/DII flows", fii, flows.expected_latest(cal, now)))

    cues = db.scalar(
        select(func.max(GlobalCueSnapshot.fetched_at)).where(GlobalCueSnapshot.last.is_not(None))
    )
    cue_status = (
        "missing" if cues is None else ("ok" if (now - cues).total_seconds() < 3600 else "stale")
    )
    out.append(
        FreshnessOut(
            dataset="Global cues (last good fetch)", latest=cues, expected=None, status=cue_status
        )
    )

    last_call = db.scalar(
        select(func.max(Prediction.session_date)).where(Prediction.kind == MODEL_NAME)
    )
    n_calls = db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.kind == MODEL_NAME)
    )
    lock = get_settings().ledger_lock_time
    out.append(market_call_freshness(cal, now, last_call, int(n_calls or 0), lock))
    return out


def call_due_session(cal: TradingCalendar, now: datetime, lock: time) -> date:
    """The latest session whose call is already due at `now`.

    A call for session S is written by the 07:30 IST job and locked at `lock` (09:15 IST)
    on S. So once the lock time has passed on a trading day, that day's call is due;
    before it, and on non-trading days, only the previous session's call is.
    """
    local = now.astimezone(IST)
    today = local.date()
    if cal.is_trading_day(today) and local.time() >= lock:
        return today
    return cal.previous_trading_day(today)


def market_call_freshness(
    cal: TradingCalendar, now: datetime, last_call: date | None, n_calls: int, lock: time
) -> FreshnessOut:
    """The "Last market call" freshness row, session-aware.

    `ok` when the ledger holds a call for every session already due (`call_due_session`):
    the upcoming session's call, or, before the lock time on a trading day and on
    non-trading days, the previous session's. `stale` once a due session has no call,
    `missing` with an empty ledger. `expected` is the session we are heading into.
    """
    session = session_for(cal, now)
    due = call_due_session(cal, now, lock)
    records = f"{n_calls} ledger record{'s' if n_calls != 1 else ''}"
    lock_text = f"{lock:%H:%M} IST"
    if last_call is None:
        status, detail = "missing", "no market call in the ledger yet"
    elif last_call >= session:
        status, detail = "ok", f"{records}; the call for {session} is recorded"
    elif last_call >= due:
        status = "ok"
        detail = (
            f"{records}; latest call is for {last_call}; the next is due for {session} "
            f"by {lock_text}"
        )
    else:
        status = "stale"
        detail = (
            f"{records}; latest call is for {last_call}; the call for {due} was due by "
            f"{lock_text} and is missing"
        )
    return FreshnessOut(
        dataset="Last market call", latest=last_call, expected=session, status=status, detail=detail
    )


def _models(db: Session) -> list[ModelHealthOut]:
    """Every champion in production with its out-of-sample metrics (None when absent).

    Selects columns explicitly so the model artifact (about 1.3 MB each) is never read.
    """
    rows = db.execute(
        select(
            ModelVersion.name,
            ModelVersion.target,
            ModelVersion.version,
            ModelVersion.trained_through,
            ModelVersion.metrics,
            ModelVersion.promoted_at,
        )
        .where(ModelVersion.is_champion.is_(True))
        .order_by(ModelVersion.name, ModelVersion.target)
    ).all()
    out: list[ModelHealthOut] = []
    for row in rows:
        metrics: dict[str, object] = dict(row.metrics or {})
        n_oos = metrics.get("n_oos")
        out.append(
            ModelHealthOut(
                name=row.name,
                target=row.target,
                version=row.version,
                trained_through=row.trained_through,
                oos_brier=finite(metrics.get("brier")),
                climatology_brier=climatology_brier(metrics),
                oos_accuracy=finite(metrics.get("accuracy")),
                n_oos=int(n_oos) if isinstance(n_oos, int | float) else None,
                promoted_at=row.promoted_at,
            )
        )
    return out


def _models_note(models: list[ModelHealthOut]) -> str:
    if not models:
        return (
            "No champion model yet, so no market call can be made. Train one with "
            "`python -m app.cli train` (the Saturday retrain then challenges it weekly)."
        )
    n = len(models)
    return (
        f"{n} champion model{'s' if n != 1 else ''} in production. A challenger replaces a "
        "champion only when it beats it out of sample in the Saturday retrain."
    )


def _fresh(name: str, latest: object, expected: object) -> FreshnessOut:
    status = "missing" if latest is None else "ok" if latest >= expected else "stale"  # type: ignore[operator]
    return FreshnessOut(dataset=name, latest=latest, expected=expected, status=status)  # type: ignore[arg-type]


@router.get("/api/health", response_model=HealthOut)
def health(
    db: Session = Depends(get_db),
    r: KV = Depends(kv_dep),
    _: AppUser = Depends(current_user),
) -> HealthOut:
    try:
        hb_raw = r.get(HEARTBEAT_KEY)
        redis_state = "ok"
    except redis.RedisError as exc:
        hb_raw, redis_state = None, f"error: {exc.__class__.__name__}"
    hb = datetime.fromisoformat(str(hb_raw)) if hb_raw else None
    mode = get_settings().scheduler_mode
    if mode == "external":
        last_run = db.scalar(select(func.max(JobRun.started_at)))
        alive = external_scheduler_alive(last_run, datetime.now(UTC))
        hb = last_run
    else:
        alive = hb is not None and (datetime.now(UTC) - hb).total_seconds() < 180
    statuses = all_statuses(db)
    sources = [
        SourceHealthOut(
            source=name,
            label=info["label"],
            status=statuses[name].status if name in statuses else "never_used",
            last_success_at=statuses[name].last_success_at if name in statuses else None,
            last_error_at=statuses[name].last_error_at if name in statuses else None,
            last_error=statuses[name].last_error if name in statuses else None,
            data_as_of=statuses[name].data_as_of if name in statuses else None,
            hint=(statuses[name].hint if name in statuses else None) or info["hint"],
        )
        for name, info in SOURCE_INFO.items()
    ]
    cal = TradingCalendar.load(db)
    models = _models(db)
    return HealthOut(
        now=datetime.now(UTC),
        scheduler_mode=mode,
        scheduler_alive=alive,
        scheduler_heartbeat=hb,
        database="ok",
        redis=redis_state,
        cache_backend=get_settings().kv_backend,
        jobs=_jobs(db),
        sources=sources,
        freshness=_freshness(db, cal),
        models=models,
        models_note=_models_note(models),
        holiday_coverage=calendar.coverage(db),
    )

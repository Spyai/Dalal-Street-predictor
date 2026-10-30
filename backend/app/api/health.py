"""Health: last run of every job, data freshness per source, model versions."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import redis
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import current_user, kv_dep
from app.cache import HEARTBEAT_KEY, KV
from app.config import IST, get_settings
from app.db import get_db
from app.jobs.definitions import job_specs
from app.models import AppUser, DailyBar, FiiDiiFlow, GlobalCueSnapshot, Instrument, JobRun
from app.schemas import FreshnessOut, HealthOut, JobHealthOut, SourceHealthOut
from app.services import calendar, flows
from app.services.calendar import TradingCalendar
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
    return out


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
    statuses = all_statuses()
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
        models=[],
        models_note="No models in production. The market model arrives in Phase 2.",
        holiday_coverage=calendar.coverage(db),
    )

"""Job execution: trading-day gating, retries with exponential backoff, run records, alerts.

Every scheduled job leaves a job_run row: success, partial, failed, or skipped with the
reason ("weekend", "NSE holiday: Diwali", "Kite not logged in"). Nothing is skipped
silently.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.config import IST, Settings, get_settings
from app.db import session_scope
from app.logging import get_logger
from app.models import JobRun
from app.services import telegram
from app.services.calendar import TradingCalendar

log = get_logger(__name__)


@dataclass
class JobContext:
    db: Session
    now: datetime
    today: date  # IST
    calendar: TradingCalendar
    settings: Settings


@dataclass
class JobResult:
    status: str = "success"  # success | partial | skipped
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class JobSkipped(Exception):
    """Raise from a job to record a deliberate, explained skip (no retry, no alert)."""


JobFn = Callable[[JobContext], JobResult]


@dataclass(frozen=True)
class JobSpec:
    name: str
    label: str
    schedule: str  # human-readable, shown on /health
    cron: dict[str, Any]  # APScheduler CronTrigger kwargs (IST)
    phase: int
    fn: JobFn | None = None  # None => not built yet in this phase
    trading_days_only: bool = True
    max_attempts: int | None = None  # None => settings.job_max_attempts
    alert_on_partial: bool = True
    alert_on_failure: bool = True


def _record(
    *,
    run_id: int | None,
    name: str,
    today: date,
    status: str,
    attempts: int,
    message: str,
    details: dict[str, Any] | None,
    finished: bool,
) -> int:
    with session_scope() as db:
        if run_id is None:
            row = JobRun(job_name=name, run_date=today, status=status, attempts=attempts)
            db.add(row)
        else:
            row = db.get(JobRun, run_id)  # type: ignore[assignment]
            assert row is not None
            row.status = status
            row.attempts = attempts
        row.message = message[:4000]
        row.details = details
        if finished:
            row.finished_at = datetime.now(UTC)
        db.flush()
        return row.id


def run_job(
    spec: JobSpec,
    *,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    settings: Settings | None = None,
    force: bool = False,
) -> str:
    """Run a job with retries. Returns the final status.

    `force=True` skips the trading-day check (`python -m app.cli predict --force` on a
    weekend, or re-running a job by hand); the run record says the check was skipped.
    """
    settings = settings or get_settings()
    now = now or datetime.now(UTC)
    today = now.astimezone(IST).date()
    if spec.fn is None:
        raise RuntimeError(f"job {spec.name} is not built yet (phase {spec.phase})")

    with session_scope() as db:
        cal = TradingCalendar.load(db)
    info = cal.info(today)
    notes: list[str] = []
    if spec.trading_days_only and force and not info.is_trading_day:
        notes.append(f"forced run on a non-trading day ({info.reason})")
    elif spec.trading_days_only:
        if not info.is_trading_day:
            _record(
                run_id=None,
                name=spec.name,
                today=today,
                status="skipped",
                attempts=0,
                message=f"skipped: {info.reason}",
                details=None,
                finished=True,
            )
            log.info("job_skipped", job=spec.name, reason=info.reason)
            return "skipped"
        if not info.calendar_verified:
            notes.append(f"NSE holiday calendar for {today.year} not loaded; assumed a trading day")

    max_attempts = spec.max_attempts or settings.job_max_attempts
    run_id = _record(
        run_id=None,
        name=spec.name,
        today=today,
        status="running",
        attempts=0,
        message="running",
        details=None,
        finished=False,
    )
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            with session_scope() as db:
                result = spec.fn(
                    JobContext(db=db, now=now, today=today, calendar=cal, settings=settings)
                )
        except JobSkipped as skip:
            _record(
                run_id=run_id,
                name=spec.name,
                today=today,
                status="skipped",
                attempts=attempt,
                message=f"skipped: {skip}",
                details=None,
                finished=True,
            )
            log.info("job_skipped", job=spec.name, reason=str(skip))
            return "skipped"
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            log.error(
                "job_attempt_failed",
                job=spec.name,
                attempt=attempt,
                error=last_error,
                tb=traceback.format_exc(limit=5),
            )
            _record(
                run_id=run_id,
                name=spec.name,
                today=today,
                status="running",
                attempts=attempt,
                message=f"attempt {attempt} failed: {last_error}",
                details=None,
                finished=False,
            )
            if attempt < max_attempts:
                sleep(settings.job_backoff_seconds * 2 ** (attempt - 1))
            continue

        message = "; ".join(filter(None, [result.message, *notes]))
        _record(
            run_id=run_id,
            name=spec.name,
            today=today,
            status=result.status,
            attempts=attempt,
            message=message,
            details=result.details or None,
            finished=True,
        )
        log.info("job_finished", job=spec.name, status=result.status, attempts=attempt)
        if result.status == "partial" and spec.alert_on_partial:
            telegram.send(f"⚠️ <b>{spec.label}</b> finished with gaps:\n{_esc(message)}")
        return result.status

    _record(
        run_id=run_id,
        name=spec.name,
        today=today,
        status="failed",
        attempts=max_attempts,
        message=last_error,
        details=None,
        finished=True,
    )
    if spec.alert_on_failure:
        telegram.send(
            f"❌ <b>{spec.label}</b> failed after {max_attempts} attempts.\n{_esc(last_error)}"
        )
    return "failed"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

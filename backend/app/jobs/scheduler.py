"""Scheduler process: `python -m app.jobs.scheduler`. All triggers in Asia/Kolkata."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.cache import HEARTBEAT_KEY, get_kv
from app.config import IST, get_settings
from app.db import session_scope
from app.jobs.definitions import get_spec, job_specs
from app.jobs.runner import run_job
from app.logging import configure_logging, get_logger
from app.services import calendar, instruments

log = get_logger(__name__)


def heartbeat() -> None:
    try:
        get_kv().set(HEARTBEAT_KEY, datetime.now(UTC).isoformat(), ex=300)
    except Exception as exc:
        log.warning("heartbeat_failed", error=str(exc))


def startup_checks() -> None:
    """Make sure the basics exist before the first scheduled run."""
    with session_scope() as db:
        instruments.ensure_reference_instruments(db)
        cov = calendar.coverage(db)
    year = datetime.now(IST).year
    if not cov.get(year):
        log.info("startup_holiday_refresh", year=year)
        run_job(get_spec("holiday_refresh"))
    run_job(get_spec("global_cues_refresh"))


def _on_error(event: JobExecutionEvent) -> None:
    log.error("scheduler_job_crashed", job=event.job_id, error=str(event.exception))


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    sched = BlockingScheduler(
        timezone=IST,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 15 * 60},
    )
    for spec in job_specs():
        if spec.fn is None:
            log.info("job_not_built_yet", job=spec.name, phase=spec.phase)
            continue
        sched.add_job(
            run_job,
            CronTrigger(timezone=IST, **spec.cron),
            args=[spec],
            id=spec.name,
            name=spec.label,
        )
        log.info("job_registered", job=spec.name, schedule=spec.schedule)
    now = datetime.now(IST)
    sched.add_job(heartbeat, "interval", seconds=60, id="heartbeat", next_run_time=now)
    # Run once inside the scheduler so slow retries (e.g. NSE down) never delay startup.
    sched.add_job(startup_checks, id="startup_checks", next_run_time=now)
    sched.add_listener(_on_error, EVENT_JOB_ERROR)
    log.info("scheduler_started", tz=str(IST))
    sched.start()


if __name__ == "__main__":
    main()

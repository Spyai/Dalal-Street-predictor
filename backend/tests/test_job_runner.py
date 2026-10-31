"""Job runner: trading-day gating, retries with backoff, run records, alerts."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import IST
from app.jobs import runner
from app.jobs.runner import JobContext, JobResult, JobSkipped, JobSpec, run_job
from app.models import JobRun
from app.providers.base import HolidayRecord
from app.services.calendar import store_holidays

FRIDAY_0730 = datetime(2026, 10, 30, 7, 30, tzinfo=IST)
SATURDAY = datetime(2026, 10, 31, 10, 0, tzinfo=IST)


@pytest.fixture()
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []
    monkeypatch.setattr(runner.telegram, "send", lambda text, **kw: sent.append(text) or True)
    return sent


def _runs(db: Session, name: str) -> list[JobRun]:
    db.expire_all()
    return list(db.scalars(select(JobRun).where(JobRun.job_name == name).order_by(JobRun.id)))


def _spec(fn: runner.JobFn, **kw: object) -> JobSpec:
    return JobSpec("t", "Test job", "now", {}, phase=1, fn=fn, **kw)  # type: ignore[arg-type]


def test_weekend_is_recorded_as_skipped(db: Session, alerts: list[str]) -> None:
    called: list[int] = []
    status = run_job(_spec(lambda ctx: called.append(1) or JobResult()), now=SATURDAY)
    assert status == "skipped" and not called
    (run,) = _runs(db, "t")
    assert run.status == "skipped" and run.message == "skipped: weekend"
    assert not alerts


def test_holiday_is_recorded_with_name(db: Session, alerts: list[str]) -> None:
    store_holidays(db, [HolidayRecord(date(2026, 10, 30), "Synthetic Holiday")], "test")
    db.commit()
    run_job(_spec(lambda ctx: JobResult()), now=FRIDAY_0730)
    (run,) = _runs(db, "t")
    assert run.message == "skipped: NSE holiday: Synthetic Holiday"


def test_unverified_calendar_is_noted(db: Session, alerts: list[str]) -> None:
    status = run_job(_spec(lambda ctx: JobResult("success", "did the thing")), now=FRIDAY_0730)
    assert status == "success"
    (run,) = _runs(db, "t")
    assert "did the thing" in (run.message or "")
    assert "holiday calendar for 2026 not loaded" in (run.message or "")


def test_retries_with_exponential_backoff_then_alerts(db: Session, alerts: list[str]) -> None:
    sleeps: list[float] = []

    def always_fails(ctx: JobContext) -> JobResult:
        raise RuntimeError("source down")

    status = run_job(_spec(always_fails), now=FRIDAY_0730, sleep=sleeps.append)
    assert status == "failed"
    assert sleeps == [30.0, 60.0]  # 3 attempts, backoff doubles
    (run,) = _runs(db, "t")
    assert run.status == "failed" and run.attempts == 3
    assert "source down" in (run.message or "")
    assert len(alerts) == 1 and "failed after 3 attempts" in alerts[0]


def test_recovers_on_retry(db: Session, alerts: list[str]) -> None:
    attempts: list[int] = []

    def flaky(ctx: JobContext) -> JobResult:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("blip")
        return JobResult("success", "ok")

    assert run_job(_spec(flaky), now=FRIDAY_0730, sleep=lambda s: None) == "success"
    (run,) = _runs(db, "t")
    assert run.attempts == 2 and run.status == "success"
    assert not alerts


def test_explicit_skip_and_partial_alert(db: Session, alerts: list[str]) -> None:
    def skip(ctx: JobContext) -> JobResult:
        raise JobSkipped("Kite not logged in")

    assert run_job(_spec(skip), now=FRIDAY_0730) == "skipped"
    assert _runs(db, "t")[-1].message == "skipped: Kite not logged in"

    assert (
        run_job(_spec(lambda ctx: JobResult("partial", "FII/DII unavailable")), now=FRIDAY_0730)
        == "partial"
    )
    assert alerts and "FII/DII unavailable" in alerts[-1]

    alerts.clear()
    run_job(_spec(lambda ctx: JobResult("partial", "x"), alert_on_partial=False), now=FRIDAY_0730)
    assert not alerts


def test_all_days_job_runs_on_weekend(db: Session, alerts: list[str]) -> None:
    assert (
        run_job(_spec(lambda ctx: JobResult(), trading_days_only=False), now=SATURDAY) == "success"
    )


def test_schedule_covers_the_brief() -> None:
    from app.jobs.definitions import job_specs

    specs = {s.name: s for s in job_specs()}
    expected = {
        "premarket_prediction": (7, 30),
        "open_check": (9, 20),
        "close_grading": (15, 45),
        "evening_data_pull": (19, 30),
        "nightly_reflection": (20, 30),
        "weekly_review": (10, 0),
    }
    for name, (hh, mm) in expected.items():
        assert specs[name].cron["hour"] == hh and specs[name].cron["minute"] == mm, name
    assert specs["weekly_review"].cron["day_of_week"] == "sat"
    retrain = specs["weekly_retrain"]
    assert retrain.cron == {"day_of_week": "sat", "hour": 10, "minute": 0}
    assert retrain.phase == 2 and not retrain.trading_days_only
    built = {n for n, s in specs.items() if s.fn is not None}
    assert built == {
        "holiday_refresh",
        "kite_login_reminder",
        "global_cues_refresh",
        "instrument_sync",
        "evening_data_pull",
        "premarket_prediction",
        "open_check",
        "close_grading",
        "weekly_retrain",
    }


def test_force_runs_a_trading_days_job_on_a_weekend(db: Session, alerts: list[str]) -> None:
    called: list[int] = []
    spec = _spec(lambda ctx: called.append(1) or JobResult("success", "ran"))
    assert run_job(spec, now=SATURDAY) == "skipped" and not called
    assert run_job(spec, now=SATURDAY, force=True) == "success" and called == [1]
    forced = _runs(db, "t")[-1]
    assert forced.status == "success"
    assert "ran" in (forced.message or "") and "forced run on a non-trading day" in (
        forced.message or ""
    )
    assert not alerts

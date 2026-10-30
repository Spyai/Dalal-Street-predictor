"""The daily schedule (Section 3 of the brief). All times IST.

Jobs with fn=None are part of the plan but not built yet; /health lists them with the
phase that delivers them, and the scheduler does not register them.
"""

from __future__ import annotations

from app.config import get_settings
from app.jobs import tasks
from app.jobs.runner import JobSpec


def job_specs() -> list[JobSpec]:
    s = get_settings()
    rt = s.kite_login_reminder_time
    return [
        JobSpec(
            "holiday_refresh",
            "Holiday calendar refresh",
            "06:05 daily",
            {"hour": 6, "minute": 5},
            phase=1,
            fn=tasks.holiday_refresh,
            trading_days_only=False,
        ),
        JobSpec(
            "kite_login_reminder",
            "Kite login reminder",
            f"{rt:%H:%M} trading days",
            {"day_of_week": "mon-fri", "hour": rt.hour, "minute": rt.minute},
            phase=1,
            fn=tasks.kite_login_reminder,
            alert_on_partial=False,
            max_attempts=1,
        ),
        JobSpec(
            "global_cues_refresh",
            "Global cues refresh",
            f"every {s.global_cues_refresh_minutes} min",
            {"minute": f"*/{s.global_cues_refresh_minutes}"},
            phase=1,
            fn=tasks.global_cues_refresh,
            trading_days_only=False,
            max_attempts=1,
            alert_on_partial=False,
            alert_on_failure=False,
        ),
        JobSpec(
            "premarket_prediction",
            "Pre-market prediction run",
            "07:30 trading days",
            {"day_of_week": "mon-fri", "hour": 7, "minute": 30},
            phase=2,
        ),
        JobSpec(
            "instrument_sync",
            "Instrument + universe sync",
            "08:45 trading days",
            {"day_of_week": "mon-fri", "hour": 8, "minute": 45},
            phase=1,
            fn=tasks.instrument_sync,
        ),
        JobSpec(
            "open_check",
            "Open check (gap grading)",
            "09:20 trading days",
            {"day_of_week": "mon-fri", "hour": 9, "minute": 20},
            phase=2,
        ),
        JobSpec(
            "intraday_monitor",
            "Intraday monitoring",
            "09:15-15:30 trading days",
            {"day_of_week": "mon-fri", "hour": "9-15", "minute": "*"},
            phase=3,
        ),
        JobSpec(
            "close_grading",
            "Close grading",
            "15:45 trading days",
            {"day_of_week": "mon-fri", "hour": 15, "minute": 45},
            phase=2,
        ),
        JobSpec(
            "evening_data_pull",
            "Evening data pull",
            "19:30 trading days",
            {"day_of_week": "mon-fri", "hour": 19, "minute": 30},
            phase=1,
            fn=tasks.evening_data_pull,
        ),
        JobSpec(
            "nightly_reflection",
            "Nightly reflection",
            "20:30 trading days",
            {"day_of_week": "mon-fri", "hour": 20, "minute": 30},
            phase=4,
        ),
        JobSpec(
            "weekly_review",
            "Weekly review + retrain",
            "Saturday 10:00",
            {"day_of_week": "sat", "hour": 10, "minute": 0},
            phase=4,
            trading_days_only=False,
        ),
    ]


def get_spec(name: str) -> JobSpec:
    for spec in job_specs():
        if spec.name == name:
            return spec
    raise KeyError(name)

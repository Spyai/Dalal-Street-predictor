"""Today page data: market clock, global cues, FII/DII flows, Kite status."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import current_user, kv_dep
from app.api.kite import kite_status
from app.cache import KV
from app.db import get_db
from app.jobs.definitions import get_spec
from app.jobs.runner import run_job
from app.models import AppUser
from app.schemas import CueOut, FlowDayOut, FlowsOut, MarketClockOut, TodayOut
from app.services import flows, global_cues
from app.services.calendar import TradingCalendar

router = APIRouter(prefix="/api/today", tags=["today"])


@router.get("", response_model=TodayOut)
def today(db: Session = Depends(get_db), _: AppUser = Depends(current_user)) -> TodayOut:
    cal = TradingCalendar.load(db)
    clock = cal.clock()
    days = flows.recent(db, 10)
    expected = flows.expected_latest(cal)
    if not days:
        flow_status, note = "unavailable", "No FII/DII data stored yet."
    elif days[0].trade_date >= expected:
        flow_status, note = "ok", None
    else:
        flow_status, note = "stale", f"Latest FII/DII data is for {days[0].trade_date}."
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
    )


@router.post("/refresh-cues", status_code=202)
def refresh_cues(r: KV = Depends(kv_dep), _: AppUser = Depends(current_user)) -> dict[str, str]:
    """Run the global-cue refresh now. Limited to once a minute."""
    if not r.set("ratelimit:manual_cues", "1", nx=True, ex=60):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Refreshed less than a minute ago.")
    return {"status": run_job(get_spec("global_cues_refresh"))}

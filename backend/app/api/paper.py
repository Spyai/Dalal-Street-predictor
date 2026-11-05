"""Paper portfolio page data (brief §7.10, §6.3): the book's summary, trades and equity curve.

Everything comes from `app.services.paper`: simulated fills at recorded prices with the
configured cost model, never real orders. Every metric carries its sample size and is None
when there is nothing to compute it from; the sample-size warning is on below
`Settings.min_sample_size` closed trades. The benchmark is the Nifty 50 *price* index
rebased to the paper capital (the total-return index is not available), and the summary's
`note` says so.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.config import get_settings
from app.db import get_db
from app.ml.core import CostModel
from app.models import AppUser
from app.schemas import PaperOut, PaperTradeOut
from app.services import paper

router = APIRouter(prefix="/api/paper", tags=["paper"])

TRADE_STATUSES: tuple[str, ...] = ("open", "closed")


@router.get("", response_model=PaperOut)
def get_paper(
    recent_limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> PaperOut:
    """The paper book: trade statistics (n closed trades), curve statistics (n snapshots /
    daily returns), the equity curve beside the rebased Nifty 50, open positions and the
    most recent closed trades."""
    s = get_settings()
    summary = paper.portfolio_summary(db, settings=s, recent_limit=recent_limit)
    return PaperOut(
        **summary.as_dict(),
        min_sample_size=s.min_sample_size,
        risk_per_trade_pct=s.risk_per_trade_pct,
        max_position_pct=s.paper_max_position_pct,
        cost_round_trip_pct=CostModel.from_settings(s).round_trip_pct(),
    )


@router.get("/trades", response_model=list[PaperTradeOut])
def list_trades(
    status: str | None = Query(default=None, max_length=8),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> list[PaperTradeOut]:
    """Every paper trade, newest entry first; `status` = open | closed filters."""
    wanted: str | None = None
    if status is not None and status.strip():
        wanted = status.strip().lower()
        if wanted not in TRADE_STATUSES:
            raise HTTPException(
                422, f"Unknown status {wanted!r}; one of {', '.join(TRADE_STATUSES)}."
            )
    rows = paper.list_trades(db, status=wanted, limit=limit)
    return [PaperTradeOut.model_validate(paper.trade_dict(t)) for t in rows]

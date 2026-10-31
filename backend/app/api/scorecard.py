"""Scorecard page data (brief §6.3): the live track record, the ledger rows, verify-ledger.

Everything here is read from the ledger and its grades through `app.services.scorecard`
and `app.services.ledger`; the API never imports LightGBM. A metric with nothing behind
it is None and the UI shows "—".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.config import get_settings
from app.db import get_db
from app.models import AppUser, Prediction
from app.schemas import LedgerVerifyOut, PredictionRowOut, ScorecardOut
from app.services import ledger
from app.services.scorecard import row_dict, scorecard

router = APIRouter(prefix="/api/scorecard", tags=["scorecard"])


def resolve_target(target: str | None) -> str:
    """The requested target (upper-cased), or the first configured prediction target."""
    if target is not None and target.strip():
        return target.strip().upper()
    configured = get_settings().prediction_target_symbols
    return configured[0] if configured else "NIFTY 50"


def known_targets(db: Session) -> list[str]:
    """Configured targets first, then any other target that has records in the ledger."""
    out = list(get_settings().prediction_target_symbols)
    stored = db.scalars(select(Prediction.target).distinct().order_by(Prediction.target)).all()
    out.extend(t for t in stored if t not in out)
    return out


@router.get("", response_model=ScorecardOut)
def get_scorecard(
    target: str | None = Query(default=None, max_length=32),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> ScorecardOut:
    s = get_settings()
    sc = scorecard(db, resolve_target(target), settings=s)
    return ScorecardOut(
        targets=known_targets(db), min_sample_size=s.min_sample_size, **sc.as_dict()
    )


@router.get("/ledger", response_model=list[PredictionRowOut])
def ledger_rows(
    target: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> list[PredictionRowOut]:
    """Recent ledger records with their grades, newest first. No target = every target."""
    wanted = target.strip().upper() if target is not None and target.strip() else None
    rows = ledger.history(db, wanted, limit)
    return [PredictionRowOut.model_validate(row_dict(p, g)) for p, g in rows]


@router.post("/verify-ledger", response_model=LedgerVerifyOut)
def verify_ledger(
    db: Session = Depends(get_db), _: AppUser = Depends(current_user)
) -> LedgerVerifyOut:
    """Walk the whole hash chain and report the first bad record, if any."""
    v = ledger.verify_ledger(db)
    return LedgerVerifyOut(
        ok=v.ok, checked=v.checked, first_bad_seq=v.first_bad_seq, message=v.message
    )

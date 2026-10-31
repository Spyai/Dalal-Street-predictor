"""Scorecard page data (brief §6.3): the live track record, the ledger rows, verify-ledger.

Everything here is read from the ledger and its grades through `app.services.scorecard`
and `app.services.ledger`; the API never imports LightGBM. A metric with nothing behind
it is None and the UI shows "—". A target that is neither configured nor in the ledger
is a 404, not an empty scorecard that looks like a real one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.config import get_settings
from app.db import get_db
from app.models import AppUser, Prediction
from app.schemas import LedgerVerifyOut, PredictionRowOut, ScorecardOut
from app.services import ledger
from app.services.scorecard import MIN_PAIRED_N, row_dict, scorecard

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


def known_target(db: Session, target: str | None) -> tuple[str, list[str]]:
    """(resolved target, known targets); 404 when the target is unknown.

    Unknown = not configured and without a single ledger record.
    """
    wanted = resolve_target(target)
    known = known_targets(db)
    if wanted not in known:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown target {wanted!r}; known targets: {', '.join(known)}.",
        )
    return wanted, known


@router.get("", response_model=ScorecardOut)
def get_scorecard(
    target: str | None = Query(default=None, max_length=32),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> ScorecardOut:
    s = get_settings()
    wanted, known = known_target(db, target)
    sc = scorecard(db, wanted, settings=s)
    return ScorecardOut(
        targets=known, min_sample_size=s.min_sample_size, min_paired_n=MIN_PAIRED_N, **sc.as_dict()
    )


@router.get("/ledger", response_model=list[PredictionRowOut])
def ledger_rows(
    target: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> list[PredictionRowOut]:
    """Recent ledger records with their grades, newest first. No target = every target."""
    wanted: str | None = None
    if target is not None and target.strip():
        wanted, _known = known_target(db, target)
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

"""Scorecard page data (brief §6.3): the live track record, the ledger rows, verify-ledger,
and the stock-ideas section (`/api/scorecard/ideas`).

Everything here is read from the ledgers and their grades through `app.services.scorecard`,
`app.services.ideas_scorecard` and `app.services.ledger`; the API never imports LightGBM. A
metric with nothing behind it is None and the UI shows "—". A target that is neither
configured nor in the ledger is a 404, not an empty scorecard that looks like a real one;
so is a horizon that is not one of `core.horizons()`.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.api.ideas import resolve_horizon
from app.config import get_settings
from app.db import get_db
from app.ml.core import horizons
from app.models import AppUser, Prediction
from app.schemas import (
    IdeaChampionOut,
    IdeaRowOut,
    IdeasScorecardOut,
    LedgerVerifyOut,
    PredictionRowOut,
    ScorecardOut,
)
from app.services import ledger
from app.services.ideas_scorecard import ideas_scorecard
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
    chain: str = Query(default="prediction", max_length=16),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> LedgerVerifyOut:
    """Walk one whole hash chain and report the first bad record, if any.

    `chain` is "prediction" (the market-call ledger, the default) or "idea" (the stock-idea
    ledger); anything else is a 404.
    """
    wanted = chain.strip().lower()
    if wanted not in ledger.CHAIN_NAMES:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown ledger chain {wanted!r}; one of {', '.join(ledger.CHAIN_NAMES)}.",
        )
    v = ledger.verify_ledger(db, wanted)
    return LedgerVerifyOut(
        ok=v.ok, checked=v.checked, first_bad_seq=v.first_bad_seq, message=v.message
    )


# ----------------------------------------------------------------------------- ideas


def json_safe(value: Any) -> Any:
    """`value` with every non-finite float replaced by None, recursively, so a stored
    metric that is NaN reaches the browser as a missing number, not as invalid JSON."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


@router.get("/ideas", response_model=IdeasScorecardOut)
def get_ideas_scorecard(
    horizon: str | None = Query(default=None, max_length=16),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> IdeasScorecardOut:
    """The stock-ideas track record (brief §6.3): hit rate, average win / loss, expectancy
    and profit factor after costs with their sample sizes, by setup and horizon, the
    calibration of p_target, the champion model per horizon and the recent ideas. No
    horizon = every horizon; an unknown one is a 404."""
    s = get_settings()
    wanted = resolve_horizon(horizon, s)
    sc = ideas_scorecard(db, wanted, settings=s)
    d = sc.as_dict()
    champions = {
        h: IdeaChampionOut.model_validate(json_safe(c)) if c is not None else None
        for h, c in d.pop("champions").items()
    }
    recent = [IdeaRowOut.model_validate(r) for r in d.pop("recent")]
    return IdeasScorecardOut(
        horizons=list(horizons(s)),
        min_sample_size=s.min_sample_size,
        champions=champions,
        recent=recent,
        **json_safe(d),
    )

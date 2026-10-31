"""Grading market calls against what the market did (brief §3 09:20 / 15:45, §6.1).

Grades live in `prediction_grade`, one row per prediction, and are mutable: the gap
grade arrives at 09:20 and the close grade at 15:45; re-grading overwrites. The ledger
row itself is never touched.

Scores (3 classes in `core.CLASSES` order) use the same functions as the walk-forward
metrics (`app.ml.metrics`), so a live grade and an out-of-sample metric are directly
comparable. `actual_pct` is rounded through `core.finite()` *before* it is classified,
so the number shown and the class it produced always agree; with the default threshold a
move of exactly +0.30 % is FLAT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import (
    BASELINE_NAMES,
    CLASS_INDEX,
    CLASSES,
    class_of,
    finite,
    gap_class_of,
    pct_change,
)
from app.ml.metrics import brier, log_loss
from app.models import Prediction, PredictionGrade
from app.services.ledger import calls_for_session


@dataclass
class GradingReport:
    """What `grade_session` did: how many it graded and which it had to skip, with reasons."""

    graded: int = 0
    skipped: list[str] = field(default_factory=list)


def probs_vector(p: Prediction) -> np.ndarray:
    """The recorded probabilities as a (1, 3) array in CLASSES order (DOWN, FLAT, UP)."""
    by_class = {"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down}
    return np.asarray([[by_class[c] for c in CLASSES]], dtype=float)


def score_call(p: Prediction, actual_class: str) -> tuple[bool, float | None, float | None]:
    """(correct, brier, log_loss) of one recorded call against the realised class."""
    y = np.asarray([CLASS_INDEX[actual_class]])
    probs = probs_vector(p)
    return (
        p.predicted_class == actual_class,
        finite(brier(probs, y)),
        finite(log_loss(probs, y)),
    )


def baseline_correctness(p: Prediction, actual_class: str) -> dict[str, bool | None]:
    """Per baseline: right, wrong, or None when the baseline had no call that day."""
    calls: dict[str, Any] = dict(p.baselines or {})
    out: dict[str, bool | None] = {}
    for name in BASELINE_NAMES:
        call = calls.get(name)
        out[name] = None if call is None else bool(call == actual_class)
    return out


def gap_outcome(p: Prediction, actual_open: float) -> tuple[float | None, bool | None]:
    """(gap_actual_pct, gap_correct). gap_correct is None when no gap call was recorded."""
    gap_pct = finite(pct_change(actual_open, p.reference_close))
    if gap_pct is None:
        return None, None
    if p.gap_call is None:
        return gap_pct, None
    return gap_pct, gap_class_of(gap_pct) == p.gap_call


def _existing(db: Session, prediction_id: int) -> PredictionGrade | None:
    return db.scalars(
        select(PredictionGrade).where(PredictionGrade.prediction_id == prediction_id)
    ).first()


def _upsert(db: Session, prediction_id: int, values: dict[str, Any]) -> PredictionGrade:
    stmt = insert(PredictionGrade).values(prediction_id=prediction_id, **values)
    stmt = stmt.on_conflict_do_update(constraint="uq_prediction_grade_prediction", set_=values)
    db.execute(stmt)
    db.flush()
    return db.execute(
        select(PredictionGrade)
        .where(PredictionGrade.prediction_id == prediction_id)
        .execution_options(populate_existing=True)
    ).scalar_one()


def grade_gap(
    db: Session, p: Prediction, actual_open: float, *, now: datetime, source: str
) -> PredictionGrade:
    """Grade the opening-gap call against the actual open (09:20 job).

    Only the gap fields are written; an existing close grade is left untouched.
    """
    open_ = finite(actual_open)
    if open_ is None:
        raise ValueError(f"actual_open is not a finite number: {actual_open!r}")
    gap_pct, gap_correct = gap_outcome(p, open_)
    existing = _existing(db, p.id)
    values: dict[str, Any] = {
        "graded_at": existing.graded_at if existing is not None else now,
        "actual_open": open_,
        "gap_actual_pct": gap_pct,
        "gap_correct": gap_correct,
        "gap_graded_at": now,
        "source": source,
    }
    return _upsert(db, p.id, values)


def grade_close(
    db: Session,
    p: Prediction,
    actual_open: float | None,
    actual_close: float,
    *,
    now: datetime,
    source: str,
    settings: Settings | None = None,
) -> PredictionGrade:
    """Grade the market call against the session close (15:45 job).

    Writes actual_pct, actual_class, correct, Brier, log loss and every baseline's
    correctness. If `actual_open` is given and the gap was not graded yet, the gap is
    graded too; a gap grade already in place is kept as it was.
    """
    close = finite(actual_close)
    if close is None:
        raise ValueError(f"actual_close is not a finite number: {actual_close!r}")
    s = settings or get_settings()
    actual_pct = finite(pct_change(close, p.reference_close))
    if actual_pct is None:
        raise ValueError(f"cannot compute a move from reference_close {p.reference_close!r}")
    actual_class = class_of(actual_pct, s.market_flat_threshold_pct)
    correct, brier_score, ll = score_call(p, actual_class)

    existing = _existing(db, p.id)
    values: dict[str, Any] = {
        "graded_at": now,
        "actual_close": close,
        "actual_pct": actual_pct,
        "actual_class": actual_class,
        "correct": correct,
        "brier": brier_score,
        "log_loss": ll,
        "baseline_correct": baseline_correctness(p, actual_class),
        "source": source,
    }
    open_ = finite(actual_open)
    if existing is not None and existing.gap_graded_at is not None:
        pass  # the 09:20 grade stands
    elif open_ is not None:
        gap_pct, gap_correct = gap_outcome(p, open_)
        values.update(
            actual_open=open_, gap_actual_pct=gap_pct, gap_correct=gap_correct, gap_graded_at=now
        )
    return _upsert(db, p.id, values)


def grade_session(
    db: Session,
    session_date: date,
    closes: dict[str, tuple[float | None, float]],
    *,
    now: datetime,
    source: str,
) -> GradingReport:
    """Close-grade every record of a session. `closes` maps target -> (open or None, close).

    A target without a close is skipped with a reason, never graded against a guess.
    """
    report = GradingReport()
    for p in calls_for_session(db, session_date):
        pair = closes.get(p.target)
        if pair is None or finite(pair[1]) is None:
            report.skipped.append(
                f"{p.target} seq {p.seq}: no close for {session_date.isoformat()}"
            )
            continue
        actual_open, actual_close = pair
        grade_close(db, p, actual_open, actual_close, now=now, source=source)
        report.graded += 1
    return report

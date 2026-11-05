"""Grading market calls against what the market did (brief §3 09:20 / 15:45, §6.1).

Grades live in `prediction_grade`, one row per prediction, and are mutable: the gap
grade arrives at 09:20 and the close grade at 15:45; re-grading overwrites. The ledger
row itself is never touched.

Scores (3 classes in `core.CLASSES` order) use the same functions as the walk-forward
metrics (`app.ml.metrics`), so a live grade and an out-of-sample metric are directly
comparable. `actual_pct` is rounded through `core.finite()` *before* it is classified,
so the number shown and the class it produced always agree; with the default threshold a
move of exactly +0.30 % is FLAT.

Provenance: `source` names where the close came from once the close is graded, and the
gap's own source and time are kept in `notes` ("gap graded from kite at 2026-11-02 09:20
IST; close from nse_bhavcopy"), so neither grade overwrites the other's origin.

A call that can never be graded (its session never happened, or the data is unusable)
is *voided* with `void_grade`: `actual_close` stays NULL, `error_category` says why and
`notes` carry the operator's reason; the ledger row itself is untouched and
`ledger.ungraded` stops listing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import IST, Settings, get_settings
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

# Why a call was voided instead of graded (stored in PredictionGrade.error_category).
VOID_CATEGORIES: tuple[str, ...] = ("no_session", "data_error")
VOID_SOURCE = "manual"


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


def _ist(at: datetime) -> str:
    return at.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")


def gap_provenance(source: str, at: datetime) -> str:
    """The note that records where and when the gap was graded."""
    return f"gap graded from {source} at {_ist(at)}"


def _gap_provenance_of(existing: PredictionGrade) -> str | None:
    """The gap's own provenance from an existing grade row, or None if the gap is ungraded.

    Taken from `notes` when the row already carries it; otherwise the row was written by
    `grade_gap` alone, so its `source` is the gap's source.
    """
    if existing.gap_graded_at is None:
        return None
    notes = existing.notes or ""
    if notes.startswith("gap graded from "):
        return notes.split("; close from ", 1)[0]
    return gap_provenance(existing.source, existing.gap_graded_at)


def grade_gap(
    db: Session, p: Prediction, actual_open: float, *, now: datetime, source: str
) -> PredictionGrade:
    """Grade the opening-gap call against the actual open (09:20 job).

    Only the gap fields are written. An existing close grade is left untouched, including
    its `source`: the gap's source then goes into `notes` instead.
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
    }
    if existing is not None and existing.actual_close is not None:
        values["source"] = existing.source  # the close's; the gap's goes into notes
        values["notes"] = f"{gap_provenance(source, now)}; close from {existing.source}"
    else:
        values["source"] = source
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
    correctness; `source` becomes the close's source. If `actual_open` is given and the
    gap was not graded yet, the gap is graded too; a gap grade already in place is kept
    as it was, and its own source and time are preserved in `notes`. Grading a voided
    call clears the void.
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
    gap_note = _gap_provenance_of(existing) if existing is not None else None
    if gap_note is not None:
        pass  # the 09:20 grade stands
    elif open_ is not None:
        gap_pct, gap_correct = gap_outcome(p, open_)
        values.update(
            actual_open=open_, gap_actual_pct=gap_pct, gap_correct=gap_correct, gap_graded_at=now
        )
        gap_note = gap_provenance(source, now)
    if existing is not None and existing.actual_close is None and existing.error_category:
        values["error_category"] = None  # a real close supersedes the void
    values["notes"] = f"{gap_note}; close from {source}" if gap_note is not None else None
    return _upsert(db, p.id, values)


def void_grade(
    db: Session,
    p: Prediction,
    *,
    now: datetime,
    reason: str,
    category: str = "no_session",
    source: str = VOID_SOURCE,
) -> PredictionGrade:
    """Record that a call cannot be graded: no close will come (`no_session`) or the data
    is unusable (`data_error`). `actual_close` stays NULL; an existing gap grade is kept.

    Refuses to void a call that already has a close grade, and requires a reason: a
    voided call shows in the history with the reason next to it, never as a silent gap.
    """
    if category not in VOID_CATEGORIES:
        raise ValueError(f"category must be one of {', '.join(VOID_CATEGORIES)}: {category!r}")
    if not reason or not reason.strip():
        raise ValueError("a reason is required to void a call")
    existing = _existing(db, p.id)
    if existing is not None and existing.actual_close is not None:
        raise ValueError(f"seq {p.seq} already has a close grade; nothing to void")
    keep_gap_source = existing is not None and existing.gap_graded_at is not None
    values: dict[str, Any] = {
        "graded_at": now,
        "error_category": category,
        "notes": f"voided ({category}): {reason.strip()}",
        "source": existing.source if keep_gap_source and existing is not None else source,
    }
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

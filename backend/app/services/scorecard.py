"""The market-call scorecard (brief §6.3): the live track record, always beside the model.

Everything here is derived from the ledger and its grades; nothing is estimated. A metric
with no graded calls behind it is None (the UI shows "—"), the sample-size warning is on
below `Settings.min_sample_size` graded calls, and the verdict always names the best
naive baseline so the dashboard says in plain words whether the model is beating it.

Superseded records (ones a correction points at) are left out of the aggregates so a
corrected call counts once; `app.services.ledger.history` still lists every record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased

from app.config import Settings, get_settings
from app.ml.core import BASELINE_NAMES, MODEL_NAME, finite
from app.ml.metrics import by_confidence, calibration_bins
from app.models import ModelVersion, Prediction, PredictionGrade

BASELINE_LABELS: dict[str, str] = {
    "always_up": "always UP",
    "same_as_yesterday": "same as yesterday",
    "overnight_gap": "overnight gap direction",
    "random": "random with historical class frequencies",
}
RECENT_LIMIT = 20
OOS_KEYS: tuple[str, ...] = ("n_oos", "accuracy", "brier", "log_loss")


@dataclass
class Scorecard:
    """One target's live track record. `n_graded` is the sample size behind every metric."""

    target: str
    n_graded: int
    accuracy: float | None
    brier: float | None
    log_loss: float | None
    gap_n: int
    gap_accuracy: float | None
    by_confidence: list[dict[str, Any]]
    calibration: list[dict[str, Any]]
    baselines: list[dict[str, Any]]  # {"name", "label", "n", "accuracy"}
    verdict: str
    sample_warning: bool
    champion: dict[str, Any] | None  # {"version", "trained_through", "oos", "created_at", ...}
    recent: list[dict[str, Any]]  # last RECENT_LIMIT records + grades, as_of desc

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _not_superseded() -> Any:
    later = aliased(Prediction)
    return ~exists().where(later.corrects_prediction_id == Prediction.id)


def _mean(values: list[float]) -> float | None:
    return finite(float(np.mean(values))) if values else None


def baseline_label(name: str) -> str:
    return BASELINE_LABELS.get(name, name.replace("_", " "))


def verdict_text(
    n: int,
    accuracy: float | None,
    best_baseline_name: str | None,
    best_baseline_acc: float | None,
    sample_warning: bool,
    min_sample_size: int | None = None,
) -> str:
    """The scorecard's plain-words verdict, honest about sample size and baselines."""
    min_n = get_settings().min_sample_size if min_sample_size is None else min_sample_size
    if n == 0 or accuracy is None:
        return (
            "No graded market calls yet: the model is unproven. "
            f"Treat it as unproven until it has {min_n} graded calls."
        )
    calls = "call" if n == 1 else "calls"
    parts = [f"After {n} graded {calls} the model is right {accuracy:.0%} of the time"]
    if best_baseline_name is not None and best_baseline_acc is not None:
        parts[0] += (
            f"; the best naive baseline ({baseline_label(best_baseline_name)}) scores "
            f"{best_baseline_acc:.0%}."
        )
        edge = accuracy - best_baseline_acc
        if edge > 0.005:
            parts.append("The model is ahead of the baselines.")
        elif edge < -0.005:
            parts.append("The model is NOT beating the naive baselines.")
        else:
            parts.append("The model is level with the naive baselines: no edge shown yet.")
    else:
        parts[0] += "; no baseline has a call to compare with yet."
    if sample_warning:
        parts.append(f"Fewer than {min_n} graded calls: treat as unproven.")
    return " ".join(parts)


def champion_summary(db: Session, target: str, name: str = MODEL_NAME) -> dict[str, Any] | None:
    """The champion model's identity and out-of-sample metrics, or None if there is none."""
    mv = db.scalars(
        select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.target == target,
            ModelVersion.is_champion.is_(True),
        )
    ).first()
    if mv is None:
        return None
    metrics: dict[str, Any] = dict(mv.metrics or {})
    return {
        "id": mv.id,
        "version": mv.version,
        "kind": mv.kind,
        "trained_through": mv.trained_through.isoformat(),
        "train_rows": mv.train_rows,
        "oos": {k: metrics.get(k) for k in OOS_KEYS},
        "promoted_at": _iso(mv.promoted_at),
        "created_at": _iso(mv.created_at),
    }


def _iso(dt: datetime | date | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def row_dict(p: Prediction, g: PredictionGrade | None) -> dict[str, Any]:
    """One prediction plus its grade (fields None when ungraded), for tables and the API."""
    return {
        "id": p.id,
        "seq": p.seq,
        "target": p.target,
        "session_date": p.session_date.isoformat(),
        "as_of": _iso(p.as_of),
        "is_late": p.is_late,
        "predicted_class": p.predicted_class,
        "probs": {"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down},
        "range_p10_pct": p.range_p10_pct,
        "range_p90_pct": p.range_p90_pct,
        "confidence": p.confidence,
        "no_edge": p.no_edge,
        "gap_call": p.gap_call,
        "gap_expected_pct": p.gap_expected_pct,
        "model_version_id": p.model_version_id,
        "corrects_prediction_id": p.corrects_prediction_id,
        "actual_open": g.actual_open if g else None,
        "actual_close": g.actual_close if g else None,
        "actual_pct": g.actual_pct if g else None,
        "actual_class": g.actual_class if g else None,
        "correct": g.correct if g else None,
        "brier": g.brier if g else None,
        "log_loss": g.log_loss if g else None,
        "gap_actual_pct": g.gap_actual_pct if g else None,
        "gap_correct": g.gap_correct if g else None,
        "graded_at": _iso(g.graded_at) if g else None,
    }


def scorecard(db: Session, target: str, *, settings: Settings | None = None) -> Scorecard:
    """Aggregate every graded, non-superseded market call for `target`."""
    s = settings or get_settings()
    base = (
        select(Prediction, PredictionGrade)
        .outerjoin(PredictionGrade, PredictionGrade.prediction_id == Prediction.id)
        .where(Prediction.target == target, Prediction.kind == MODEL_NAME, _not_superseded())
    )
    rows = db.execute(base.order_by(Prediction.seq.asc())).all()
    graded = [(p, g) for p, g in rows if g is not None and g.actual_class is not None]

    correct = np.asarray([bool(g.correct) for _, g in graded], dtype=bool)
    labels = [p.confidence for p, _ in graded]
    max_prob = np.asarray(
        [max(p.prob_up, p.prob_flat, p.prob_down) for p, _ in graded], dtype=float
    )
    briers = [g.brier for _, g in graded if g.brier is not None]
    lls = [g.log_loss for _, g in graded if g.log_loss is not None]
    gap_flags = [g.gap_correct for _, g in rows if g is not None and g.gap_correct is not None]

    baselines: list[dict[str, Any]] = []
    for name in BASELINE_NAMES:
        flags = [
            bool((g.baseline_correct or {}).get(name))
            for _, g in graded
            if (g.baseline_correct or {}).get(name) is not None
        ]
        baselines.append(
            {
                "name": name,
                "label": baseline_label(name),
                "n": len(flags),
                "accuracy": _mean([float(f) for f in flags]),
            }
        )
    with_calls = [b for b in baselines if b["accuracy"] is not None]
    best = max(with_calls, key=lambda b: (b["accuracy"], b["n"])) if with_calls else None

    n = len(graded)
    accuracy = _mean([float(c) for c in correct]) if n else None
    sample_warning = n < s.min_sample_size
    recent = db.execute(
        base.order_by(Prediction.as_of.desc(), Prediction.seq.desc()).limit(RECENT_LIMIT)
    ).all()
    return Scorecard(
        target=target,
        n_graded=n,
        accuracy=accuracy,
        brier=_mean(briers),
        log_loss=_mean(lls),
        gap_n=len(gap_flags),
        gap_accuracy=_mean([float(f) for f in gap_flags]),
        by_confidence=[dict(r) for r in by_confidence(labels, correct)] if n else [],
        calibration=[dict(r) for r in calibration_bins(max_prob, correct)] if n else [],
        baselines=baselines,
        verdict=verdict_text(
            n,
            accuracy,
            best["name"] if best else None,
            best["accuracy"] if best else None,
            sample_warning,
            s.min_sample_size,
        ),
        sample_warning=sample_warning,
        champion=champion_summary(db, target),
        recent=[row_dict(p, g) for p, g in recent],
    )

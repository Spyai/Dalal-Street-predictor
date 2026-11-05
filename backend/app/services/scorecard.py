"""The market-call scorecard (brief §6.3): the live track record, always beside the model.

Everything here is derived from the ledger and its grades; nothing is estimated. A metric
with no graded calls behind it is None (the UI shows "—"), the sample-size warning is on
below `Settings.min_sample_size` graded calls, and the verdict always names a naive
baseline so the dashboard says in plain words whether the model is beating it.

Baseline comparisons are PAIRED: a baseline that had no call on some sessions (the
overnight-gap rule needs US futures; `random` needs class frequencies) is compared with
the model on exactly the sessions where it did have a call, and both sides' accuracies
are reported for that subset with its own `n`. The "toughest" baseline, the one the
model leads by the least on paired calls, is the one the verdict names; the verdict only
says ahead/behind once that paired sample has `MIN_PAIRED_N` calls, and says "too early
to tell" before that.

Superseded records (ones a correction points at) are left out of the aggregates so a
corrected call counts once; `app.services.ledger.history` still lists every record.
Aggregates never load `Prediction.feature_snapshot` or `Prediction.drivers` (large JSON
columns nothing here reads), and the champion box never loads the model artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased, defer

from app.config import Settings, get_settings
from app.ml.core import BASELINE_NAMES, CLASSES, MODEL_NAME, finite
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
MIN_PAIRED_N = 30  # paired graded calls before the verdict says ahead / behind
EDGE_TOLERANCE = 0.005  # an accuracy difference inside this is "level", not an edge
# The walk-forward scores `random` as the class-frequency (climatology) forecast, so its
# out-of-sample Brier is the climatology Brier (see app.ml.baselines.baseline_probs).
CLIMATOLOGY_BASELINE = "random"
COMPARISONS: tuple[str, ...] = ("ahead", "behind", "level", "too_early", "no_baseline", "no_calls")


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
    # {"name", "label", "n", "accuracy", "model_accuracy"}: paired on the baseline's sessions
    baselines: list[dict[str, Any]]
    best_baseline: dict[str, Any] | None  # the toughest paired comparison (same keys)
    comparison: str  # one of COMPARISONS
    verdict: str
    sample_warning: bool
    champion: dict[str, Any] | None  # {"version", "trained_through", "oos", "baselines", ...}
    recent: list[dict[str, Any]]  # last RECENT_LIMIT records + grades, as_of desc

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def not_superseded() -> Any:
    """SQL predicate: no correction points at this `Prediction` row."""
    later = aliased(Prediction)
    return ~exists().where(later.corrects_prediction_id == Prediction.id)


def _mean(values: list[float]) -> float | None:
    return finite(float(np.mean(values))) if values else None


def baseline_label(name: str) -> str:
    return BASELINE_LABELS.get(name, name.replace("_", " "))


def paired_baselines(
    graded: list[tuple[Prediction, PredictionGrade]],
) -> list[dict[str, Any]]:
    """Every baseline against the model on exactly the sessions where that baseline had a call.

    One row per `BASELINE_NAMES` entry: `n` is the size of that paired subset, `accuracy`
    the baseline's and `model_accuracy` the model's hit rate on those same sessions
    (both None when the baseline never had a call).
    """
    out: list[dict[str, Any]] = []
    for name in BASELINE_NAMES:
        pairs = [
            (bool(flag), bool(g.correct))
            for _, g in graded
            if (flag := (g.baseline_correct or {}).get(name)) is not None
        ]
        out.append(
            {
                "name": name,
                "label": baseline_label(name),
                "n": len(pairs),
                "accuracy": _mean([float(b) for b, _ in pairs]),
                "model_accuracy": _mean([float(m) for _, m in pairs]),
            }
        )
    return out


def best_baseline(
    baselines: list[dict[str, Any]], min_paired_n: int = MIN_PAIRED_N
) -> dict[str, Any] | None:
    """The toughest paired comparison: the baseline the model leads by the least.

    Only baselines that had at least one call take part. Those with `min_paired_n` or more
    paired calls are preferred; when none has that many, the toughest of the rest is named
    and `comparison_state` says it is too early to tell. Ties go to the larger sample, then
    to `BASELINE_NAMES` order. None when no baseline had a call.
    """
    pool = [
        b
        for b in baselines
        if b["n"] > 0 and b["accuracy"] is not None and b["model_accuracy"] is not None
    ]
    if not pool:
        return None
    enough = [b for b in pool if b["n"] >= min_paired_n]
    return min(enough or pool, key=lambda b: (b["model_accuracy"] - b["accuracy"], -b["n"]))


def comparison_state(
    best: Mapping[str, Any] | None, n_graded: int, min_paired_n: int = MIN_PAIRED_N
) -> str:
    """Where the model stands against its toughest baseline, as one of `COMPARISONS`."""
    if n_graded == 0:
        return "no_calls"
    if best is None:
        return "no_baseline"
    if best["n"] < min_paired_n:
        return "too_early"
    edge = float(best["model_accuracy"]) - float(best["accuracy"])
    if edge > EDGE_TOLERANCE:
        return "ahead"
    if edge < -EDGE_TOLERANCE:
        return "behind"
    return "level"


def verdict_text(
    n: int,
    accuracy: float | None,
    best_baseline_name: str | None,
    best_baseline_acc: float | None,
    sample_warning: bool,
    min_sample_size: int | None = None,
    *,
    paired_n: int | None = None,
    paired_model_acc: float | None = None,
    min_paired_n: int = MIN_PAIRED_N,
) -> str:
    """The scorecard's plain-words verdict, paired against the toughest baseline.

    `best_baseline_acc` is the baseline's accuracy on the `paired_n` sessions where it had
    a call and `paired_model_acc` the model's on those same sessions; both default to the
    whole graded sample (`n`, `accuracy`) when the comparison covers every graded call.
    Ahead/behind is only claimed with at least `min_paired_n` paired calls; the sample-size
    warning is appended whenever `sample_warning` is on.
    """
    min_n = get_settings().min_sample_size if min_sample_size is None else min_sample_size
    if n == 0 or accuracy is None:
        return (
            "No graded market calls yet: the model is unproven. "
            f"Treat it as unproven until it has {min_n} graded calls."
        )
    calls = "call" if n == 1 else "calls"
    parts = [f"After {n} graded {calls} the model is right {accuracy:.0%} of the time."]
    if best_baseline_name is not None and best_baseline_acc is not None:
        pn = n if paired_n is None else paired_n
        pm = accuracy if paired_model_acc is None else paired_model_acc
        label = baseline_label(best_baseline_name)
        if pn >= n:
            parts.append(
                f"The toughest naive baseline ({label}) scores {best_baseline_acc:.0%} "
                f"on the same {n} {calls}."
            )
        else:
            parts.append(
                f"On the {pn} of them where the toughest naive baseline ({label}) had a call, "
                f"the model scores {pm:.0%} and the baseline {best_baseline_acc:.0%}."
            )
        if pn < min_paired_n:
            parts.append(
                "Too early to tell whether the model beats the baselines: that needs "
                f"{min_paired_n} paired calls."
            )
        else:
            edge = pm - best_baseline_acc
            if edge > EDGE_TOLERANCE:
                parts.append("The model is ahead of the baselines.")
            elif edge < -EDGE_TOLERANCE:
                parts.append("The model is NOT beating the naive baselines.")
            else:
                parts.append("The model is level with the naive baselines: no edge shown yet.")
    else:
        parts.append("No baseline has a call to compare with yet.")
    if sample_warning:
        parts.append(f"Fewer than {min_n} graded calls: treat as unproven.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- champion


def oos_baselines(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The walk-forward baselines stored with a model, one row per `BASELINE_NAMES` entry.

    Each row: {"name", "label", "n", "accuracy", "brier"}; a number the training run did
    not store is None. An empty list when the model was stored without baselines at all.
    """
    raw = metrics.get("baselines")
    if not isinstance(raw, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for name in BASELINE_NAMES:
        entry = raw.get(name)
        e: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
        n = finite(e.get("n"))
        out.append(
            {
                "name": name,
                "label": baseline_label(name),
                "n": int(n) if n is not None else 0,
                "accuracy": finite(e.get("accuracy")),
                "brier": finite(e.get("brier")),
            }
        )
    return out


def climatology_brier(metrics: Mapping[str, Any]) -> float | None:
    """The out-of-sample Brier of the class-frequency forecast, or None if not stored."""
    for b in oos_baselines(metrics):
        if b["name"] == CLIMATOLOGY_BASELINE:
            return b["brier"]
    return None


def class_frequencies(metrics: Mapping[str, Any]) -> dict[str, float | None] | None:
    """Training class frequencies in UP/FLAT/DOWN order, or None if not stored."""
    raw = metrics.get("class_frequencies")
    if not isinstance(raw, Mapping):
        return None
    return {c: finite(raw.get(c)) for c in reversed(CLASSES)}


def oos_verdict_text(
    oos: Mapping[str, Any], baselines: list[dict[str, Any]], tolerance: float
) -> str:
    """One plain-words line about the walk-forward period.

    The model's probabilities against the class-frequency forecast (Brier; `tolerance` is
    the difference that counts, `Settings.retrain_min_brier_gain`), then its accuracy
    against the best baseline that had a call on every out-of-sample session (so the two
    accuracies cover the same sessions). Says so when a number is missing.
    """
    brier = finite(oos.get("brier"))
    clim = next((b["brier"] for b in baselines if b["name"] == CLIMATOLOGY_BASELINE), None)
    if brier is None:
        return "No out-of-sample metrics were stored with this model."
    if clim is None:
        return (
            f"Out of sample the model's Brier score was {brier:.3f}; no class-frequency "
            "Brier was stored with it, so there is nothing to compare that with."
        )
    diff = brier - clim
    if diff < -tolerance:
        word = "better than"
    elif diff > tolerance:
        word = "worse than"
    else:
        word = "no better than"
    text = (
        f"Out of sample the model's probabilities were {word} class frequencies "
        f"(Brier {brier:.3f} vs {clim:.3f})."
    )
    acc = finite(oos.get("accuracy"))
    n_oos = finite(oos.get("n_oos"))
    paired = [
        b for b in baselines if b["accuracy"] is not None and n_oos is not None and b["n"] == n_oos
    ]
    if acc is not None and paired:
        best = max(paired, key=lambda b: float(b["accuracy"]))
        text += (
            f" Accuracy {acc:.0%} vs the best naive baseline ({best['label']}) "
            f"{float(best['accuracy']):.0%} on the same {best['n']} sessions."
        )
    return text


def champion_summary(
    db: Session, target: str, name: str = MODEL_NAME, *, settings: Settings | None = None
) -> dict[str, Any] | None:
    """The champion model's identity and out-of-sample metrics, or None if there is none.

    Beside the model's OOS numbers come the walk-forward baselines (`baselines`), the
    training class frequencies, `range_coverage` when the training run stored it, the
    climatology Brier and a one-line `oos_verdict`. The artifact column is never read.
    """
    s = settings or get_settings()
    row = db.execute(
        select(
            ModelVersion.id,
            ModelVersion.version,
            ModelVersion.kind,
            ModelVersion.trained_through,
            ModelVersion.train_rows,
            ModelVersion.metrics,
            ModelVersion.promoted_at,
            ModelVersion.created_at,
        ).where(
            ModelVersion.name == name,
            ModelVersion.target == target,
            ModelVersion.is_champion.is_(True),
        )
    ).first()
    if row is None:
        return None
    metrics: dict[str, Any] = dict(row.metrics or {})
    oos = {k: metrics.get(k) for k in OOS_KEYS}
    baselines = oos_baselines(metrics)
    return {
        "id": row.id,
        "version": row.version,
        "kind": row.kind,
        "trained_through": row.trained_through.isoformat(),
        "train_rows": row.train_rows,
        "oos": oos,
        "baselines": baselines,
        "class_frequencies": class_frequencies(metrics),
        "range_coverage": finite(metrics.get("range_coverage")),
        "climatology_brier": climatology_brier(metrics),
        "oos_verdict": oos_verdict_text(oos, baselines, s.retrain_min_brier_gain),
        "promoted_at": _iso(row.promoted_at),
        "created_at": _iso(row.created_at),
    }


def _iso(dt: datetime | date | None) -> str | None:
    return dt.isoformat() if dt is not None else None


# ----------------------------------------------------------------------------- rows


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
        # Nothing here reads the feature snapshot or the drivers; keep the big JSON out.
        .options(defer(Prediction.feature_snapshot), defer(Prediction.drivers))
        .outerjoin(PredictionGrade, PredictionGrade.prediction_id == Prediction.id)
        .where(Prediction.target == target, Prediction.kind == MODEL_NAME, not_superseded())
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

    baselines = paired_baselines(graded)
    best = best_baseline(baselines)
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
        best_baseline=dict(best) if best else None,
        comparison=comparison_state(best, n),
        verdict=verdict_text(
            n,
            accuracy,
            best["name"] if best else None,
            best["accuracy"] if best else None,
            sample_warning,
            s.min_sample_size,
            paired_n=best["n"] if best else None,
            paired_model_acc=best["model_accuracy"] if best else None,
        ),
        sample_warning=sample_warning,
        champion=champion_summary(db, target, settings=s),
        recent=[row_dict(p, g) for p, g in recent],
    )

"""Model registry in Postgres: versions, the champion per target, promotion rules (brief §6.2).

The artifact lives in `model_version.artifact` because the job runners keep no files.
Exactly one champion per (name, target) is enforced by the partial unique index
`uq_model_version_champion`; `promote()` demotes the previous champion first.
`should_promote` compares champion and challenger on a common recent window of their
walk-forward folds (see its docstring). MLflow is optional (`Settings.mlflow_tracking_uri`),
imported lazily, and never allowed to break a training run. Nothing here commits: callers
own the transaction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, defer

from app.config import Settings, get_settings
from app.logging import get_logger
from app.ml.core import MODEL_NAME, TrainResult, WalkForwardFold, WalkForwardMetrics, finite
from app.models import ModelVersion

log = get_logger(__name__)

ACCURACY_TOLERANCE = 0.01  # full-sample fallback: a challenger may lose at most one accuracy point
WINDOW_DAYS = 182  # champion and challenger are compared on the champion's last ~6 months
# Chosen out of sample by the walk-forward (see app.ml.models), so two models with the same
# recipe may legitimately differ in them; the data-refresh rule ignores them.
SELECTED_PARAM_KEYS: frozenset[str] = frozenset({"n_estimators", "shrink"})


def save_model(
    db: Session,
    target: str,
    result: TrainResult,
    *,
    promote: bool,
    note: str | None = None,
    name: str = MODEL_NAME,
) -> ModelVersion:
    """Store a trained model; version = f"lgbm-{utcnow:%Y%m%dT%H%MZ}-{sha256(artifact)[:6]}".

    With `promote=True` the new row becomes the champion (the previous one is demoted).
    The row is flushed, not committed.
    """
    sha = hashlib.sha256(result.artifact.encode("utf-8")).hexdigest()
    base = f"lgbm-{datetime.now(UTC):%Y%m%dT%H%MZ}-{sha[:6]}"
    version = base
    taken = set(
        db.scalars(
            select(ModelVersion.version).where(
                ModelVersion.name == name,
                ModelVersion.target == target,
                ModelVersion.version.like(f"{base}%"),
            )
        ).all()
    )
    suffix = 2
    while version in taken:  # the same artifact saved twice within one minute
        version = f"{base}-{suffix}"
        suffix += 1
    mv = ModelVersion(
        name=name,
        target=target,
        version=version,
        kind="lightgbm",
        params=dict(result.params),
        feature_names=list(result.feature_names),
        metrics=result.metrics.as_dict(),
        artifact=result.artifact,
        trained_through=result.trained_through,
        train_rows=result.train_rows,
        is_champion=False,
    )
    db.add(mv)
    db.flush()
    if promote:
        _make_champion(db, mv, note or "promoted on save")
    return mv


def champion(db: Session, target: str, name: str = MODEL_NAME) -> ModelVersion | None:
    """The current champion for (name, target), or None before the first training."""
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.target == target,
            ModelVersion.is_champion.is_(True),
        )
    ).first()


def champion_light(db: Session, target: str, name: str = MODEL_NAME) -> ModelVersion | None:
    """`champion()` without reading the artifact column (about 1.3 MB per model).

    For API callers that need the version, metrics, params or feature names only.
    `mv.artifact` is deferred: touching it issues one more query instead of failing.
    """
    return db.scalars(
        select(ModelVersion)
        .options(defer(ModelVersion.artifact))
        .where(
            ModelVersion.name == name,
            ModelVersion.target == target,
            ModelVersion.is_champion.is_(True),
        )
    ).first()


def promote(db: Session, mv: ModelVersion, note: str) -> None:
    """Make `mv` the champion for its (name, target); the previous champion is demoted."""
    _make_champion(db, mv, note)


def _make_champion(db: Session, mv: ModelVersion, note: str) -> None:
    previous = db.scalars(
        select(ModelVersion).where(
            ModelVersion.name == mv.name,
            ModelVersion.target == mv.target,
            ModelVersion.is_champion.is_(True),
            ModelVersion.id != mv.id,
        )
    ).all()
    for old in previous:
        old.is_champion = False
    db.flush()  # the partial unique index allows one champion per (name, target) at a time
    mv.is_champion = True
    mv.promoted_at = datetime.now(UTC)
    mv.promotion_note = note
    db.flush()
    log.info(
        "model_promoted",
        name=mv.name,
        target=mv.target,
        version=mv.version,
        demoted=[old.version for old in previous],
    )


def champion_count(db: Session, target: str, name: str = MODEL_NAME) -> int:
    """How many champions the table holds for (name, target); the index keeps this <= 1."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(ModelVersion)
            .where(
                ModelVersion.name == name,
                ModelVersion.target == target,
                ModelVersion.is_champion.is_(True),
            )
        )
        or 0
    )


@dataclass(frozen=True)
class _FoldRow:
    test_start: date
    test_end: date
    n_test: int
    brier: float


@dataclass(frozen=True)
class _Candidate:
    """What the promotion rule knows about one side (champion or challenger)."""

    brier: float | None
    accuracy: float | None
    folds: tuple[_FoldRow, ...]
    trained_through: date | None
    feature_names: tuple[str, ...] | None
    params: Mapping[str, Any] | None
    train_rows: int | None

    @property
    def last_test_end(self) -> date | None:
        return max((f.test_end for f in self.folds), default=None)


def _to_date(x: object) -> date | None:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        try:
            return date.fromisoformat(x)
        except ValueError:
            return None
    return None


def _fold_rows(folds: Iterable[object] | None) -> tuple[_FoldRow, ...]:
    """Per-fold (test_start, test_end, n_test, brier) from `WalkForwardFold`s or their dicts.

    Folds without a usable date, size or Brier are left out rather than guessed.
    """
    out: list[_FoldRow] = []
    start: date | None
    end: date | None
    brier: float | None
    for f in folds or ():
        if isinstance(f, _FoldRow):
            out.append(f)
            continue
        if isinstance(f, WalkForwardFold):
            start, end, n, brier = f.test_start, f.test_end, f.n_test, finite(f.brier)
        elif isinstance(f, Mapping):
            start, end = _to_date(f.get("test_start")), _to_date(f.get("test_end"))
            n_raw = f.get("n_test")
            n = int(n_raw) if isinstance(n_raw, int | float) else 0
            brier = finite(f.get("brier"))
        else:
            continue
        if start is None or end is None or n <= 0 or brier is None:
            continue
        out.append(_FoldRow(start, end, n, brier))
    return tuple(out)


def _from_metrics(m: Mapping[str, Any]) -> tuple[float | None, float | None, tuple[_FoldRow, ...]]:
    folds = m.get("folds")
    return (
        finite(m.get("brier")),
        finite(m.get("accuracy")),
        _fold_rows(folds if isinstance(folds, list) else None),
    )


def _champion_side(champion: ModelVersion | Mapping[str, Any]) -> _Candidate:
    if isinstance(champion, ModelVersion):
        brier, acc, folds = _from_metrics(champion.metrics or {})
        names = champion.feature_names
        return _Candidate(
            brier=brier,
            accuracy=acc,
            folds=folds,
            trained_through=champion.trained_through,
            feature_names=tuple(str(f) for f in names) if names is not None else None,
            params=champion.params,
            train_rows=champion.train_rows,
        )
    brier, acc, folds = _from_metrics(champion)
    return _Candidate(brier, acc, folds, None, None, None, None)


def _challenger_side(challenger: TrainResult | WalkForwardMetrics) -> _Candidate:
    if isinstance(challenger, TrainResult):
        m = challenger.metrics
        return _Candidate(
            brier=finite(m.brier),
            accuracy=finite(m.accuracy),
            folds=_fold_rows(m.folds),
            trained_through=challenger.trained_through,
            feature_names=tuple(challenger.feature_names),
            params=challenger.params,
            train_rows=challenger.train_rows,
        )
    return _Candidate(
        finite(challenger.brier),
        finite(challenger.accuracy),
        _fold_rows(challenger.folds),
        None,
        None,
        None,
        None,
    )


def window_brier(folds: Iterable[object], start: date, end: date) -> tuple[float, int] | None:
    """(n-weighted Brier, n) over the folds tested entirely inside [start, end]; None if none.

    `folds` are `WalkForwardFold`s or the dicts `WalkForwardMetrics.as_dict()` stores.
    """
    rows = _fold_rows(folds)
    inside = [f for f in rows if f.test_start >= start and f.test_end <= end]
    n = sum(f.n_test for f in inside)
    if n <= 0:
        return None
    return sum(f.brier * f.n_test for f in inside) / n, n


def _same_recipe(champ: _Candidate, chal: _Candidate) -> list[str]:
    """Why the data-refresh rule does not apply; empty when it does."""
    why: list[str] = []
    if champ.feature_names is None or champ.params is None or champ.train_rows is None:
        why.append("champion row not given, only its metrics")
        return why
    if chal.feature_names is None or chal.params is None or chal.train_rows is None:
        why.append("challenger TrainResult not given, only its metrics")
        return why
    if tuple(champ.feature_names) != tuple(chal.feature_names):
        why.append("different feature set")
    champ_shape = {k: v for k, v in champ.params.items() if k not in SELECTED_PARAM_KEYS}
    chal_shape = {k: v for k, v in chal.params.items() if k not in SELECTED_PARAM_KEYS}
    if champ_shape != chal_shape:
        why.append("different params")
    if chal.train_rows <= champ.train_rows:
        why.append(f"not more training rows ({chal.train_rows} vs {champ.train_rows})")
    return why


def should_promote(
    champion: ModelVersion | Mapping[str, Any] | None,
    challenger: TrainResult | WalkForwardMetrics,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    """Champion/challenger rule (brief §6.2) with the reason in plain words.

    No champion: promote. Otherwise the two are compared on a COMMON RECENT WINDOW, the
    `WINDOW_DAYS` before the champion's `trained_through` (its last fold's test_end when
    only its metrics dict is given): the n-weighted Brier of the walk-forward folds tested
    entirely inside that window, on both sides. Two rules can promote there:

    1. beat: the challenger's window Brier is at least `retrain_min_brier_gain` lower;
    2. data refresh: same feature set, same params apart from the out-of-sample choices
       (`SELECTED_PARAM_KEYS`), strictly more training rows, and a window Brier no worse
       than the champion's plus the gain. Needs the champion `ModelVersion` and the
       challenger `TrainResult` (metrics alone cannot show the recipe).

    Without a common window (a champion stored without fold metrics, or no fold of either
    side inside it) the whole-sample rule applies: Brier better by the gain AND accuracy
    within `ACCURACY_TOLERANCE`. A champion without comparable metrics at all cannot defend
    its place. The reason text names the rule that fired and the window numbers.
    """
    s = settings or get_settings()
    gain = s.retrain_min_brier_gain
    if champion is None:
        return True, "no champion yet: the first trained model is promoted"
    champ = _champion_side(champion)
    chal = _challenger_side(challenger)
    if chal.brier is None or chal.accuracy is None:
        return False, "challenger has no out-of-sample Brier/accuracy: not promoted"

    window_end = champ.trained_through or champ.last_test_end
    if window_end is not None:
        window_start = window_end - timedelta(days=WINDOW_DAYS)
        champ_w = window_brier(champ.folds, window_start, window_end)
        chal_w = window_brier(chal.folds, window_start, window_end)
        if champ_w is not None and chal_w is not None:
            (pb, pn), (cb, cn) = champ_w, chal_w
            detail = (
                f"common window {window_start.isoformat()}..{window_end.isoformat()}: "
                f"challenger Brier {cb:.4f} (n={cn}) vs champion {pb:.4f} (n={pn}), "
                f"needs <= {pb - gain:.4f}"
            )
            if cb <= pb - gain:
                return True, f"promote (beat the champion on the common window): {detail}"
            not_refresh = _same_recipe(champ, chal)
            if not not_refresh and cb <= pb + gain:
                assert champ.train_rows is not None and chal.train_rows is not None
                return True, (
                    "promote (data refresh: same features and params, "
                    f"{chal.train_rows} vs {champ.train_rows} training rows, not worse than "
                    f"the champion by more than {gain:.4f} on the common window): {detail}"
                )
            if not not_refresh:
                not_refresh.append(f"worse by more than {gain:.4f}")
            return False, (
                f"reject (Brier gain too small on the common window; data refresh not "
                f"applicable: {'; '.join(not_refresh)}): {detail}"
            )

    if champ.brier is None or champ.accuracy is None:
        return True, "champion has no comparable out-of-sample Brier/accuracy: challenger promoted"
    brier_ok = chal.brier <= champ.brier - gain
    acc_ok = chal.accuracy >= champ.accuracy - ACCURACY_TOLERANCE
    detail = (
        f"no common window (champion folds: {len(champ.folds)}, challenger folds: "
        f"{len(chal.folds)}), comparing the whole out-of-sample run: challenger Brier "
        f"{chal.brier:.4f} vs champion {champ.brier:.4f} (needs <= {champ.brier - gain:.4f}); "
        f"accuracy {chal.accuracy:.3f} vs {champ.accuracy:.3f} "
        f"(needs >= {champ.accuracy - ACCURACY_TOLERANCE:.3f})"
    )
    if brier_ok and acc_ok:
        return True, f"promote (whole-sample rule): {detail}"
    why = []
    if not brier_ok:
        why.append("Brier gain too small")
    if not acc_ok:
        why.append("accuracy fell more than a point")
    return False, f"reject ({'; '.join(why)}): {detail}"


def log_to_mlflow(
    result: TrainResult, target: str, promoted: bool, settings: Settings | None = None
) -> None:
    """Record the run in MLflow when `mlflow_tracking_uri` is set. Never raises."""
    s = settings or get_settings()
    if not s.mlflow_tracking_uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(f"dalal-{MODEL_NAME}")
        m = result.metrics
        numbers: dict[str, float] = {}
        for key, val in (
            ("n_oos", m.n_oos),
            ("accuracy", m.accuracy),
            ("brier", m.brier),
            ("log_loss", m.log_loss),
            ("range_coverage", getattr(m, "range_coverage", None)),
            ("train_rows", result.train_rows),
        ):
            f = finite(val)
            if f is not None:
                numbers[key] = f
        for bname, entry in m.baselines.items():
            for mname, val in entry.items():
                f = finite(val)
                if f is not None:
                    numbers[f"baseline_{bname}_{mname}"] = f
        with mlflow.start_run(run_name=f"{target} through {result.trained_through.isoformat()}"):
            mlflow.set_tags(
                {
                    "model": MODEL_NAME,
                    "target": target,
                    "promoted": "true" if promoted else "false",
                    "trained_through": result.trained_through.isoformat(),
                }
            )
            mlflow.log_params({k: str(v) for k, v in result.params.items()})
            mlflow.log_metrics(numbers)
            mlflow.log_dict(m.as_dict(), "walk_forward_metrics.json")
            mlflow.log_text(result.artifact, "model_artifact.json")
    except Exception as exc:  # optional integration: log and move on
        log.warning("mlflow_log_failed", target=target, error=f"{exc.__class__.__name__}: {exc}")

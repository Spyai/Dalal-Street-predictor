"""Model registry in Postgres: versions, the champion per target, promotion rules (brief §6.2).

The artifact lives in `model_version.artifact` because the job runners keep no files.
Exactly one champion per (name, target) is enforced by the partial unique index
`uq_model_version_champion`; `promote()` demotes the previous champion first.
MLflow is optional (`Settings.mlflow_tracking_uri`), imported lazily, and never allowed
to break a training run. Nothing here commits: callers own the transaction.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.logging import get_logger
from app.ml.core import MODEL_NAME, TrainResult, WalkForwardMetrics, finite
from app.models import ModelVersion

log = get_logger(__name__)

ACCURACY_TOLERANCE = 0.01  # a challenger may lose at most one accuracy point


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


def should_promote(
    champion_metrics: dict | None,
    challenger: WalkForwardMetrics,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    """Champion/challenger rule (brief §6.2) with the reason in plain words.

    No champion: promote. Otherwise promote only if the challenger's out-of-sample Brier
    is at least `retrain_min_brier_gain` better AND its accuracy is within one point of
    the champion's. A champion without comparable metrics cannot defend its place.
    """
    s = settings or get_settings()
    gain = s.retrain_min_brier_gain
    if champion_metrics is None:
        return True, "no champion yet: the first trained model is promoted"
    champ_brier = finite(champion_metrics.get("brier"))
    champ_acc = finite(champion_metrics.get("accuracy"))
    ch_brier, ch_acc = finite(challenger.brier), finite(challenger.accuracy)
    if ch_brier is None or ch_acc is None:
        return False, "challenger has no out-of-sample Brier/accuracy: not promoted"
    if champ_brier is None or champ_acc is None:
        return True, "champion has no comparable out-of-sample Brier/accuracy: challenger promoted"
    brier_ok = ch_brier <= champ_brier - gain
    acc_ok = ch_acc >= champ_acc - ACCURACY_TOLERANCE
    detail = (
        f"challenger Brier {ch_brier:.4f} vs champion {champ_brier:.4f} "
        f"(needs <= {champ_brier - gain:.4f}); accuracy {ch_acc:.3f} vs {champ_acc:.3f} "
        f"(needs >= {champ_acc - ACCURACY_TOLERANCE:.3f})"
    )
    if brier_ok and acc_ok:
        return True, f"promote: {detail}"
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

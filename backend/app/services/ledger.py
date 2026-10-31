"""The immutable, hash-chained prediction ledger (brief §5).

Every market call is one `Prediction` row. Rows are never updated or deleted (a database
trigger from migration 0003 rejects both); corrections are new rows that point at the
original through `corrects_prediction_id`. Each row carries a SHA-256 over its own
content plus the previous row's hash, so the chain (ordered by `seq`) makes any later
edit, insertion or removal detectable by `verify_ledger`.

What exactly is hashed is fixed by `hashed_fields` and `canonical_json`: sorted keys, no
whitespace, ASCII only, NaN/inf forbidden. Every float goes through `core.finite()`
(10 decimals) before it is stored *and* before it is hashed, so a row re-read from
Postgres (double precision columns, JSON columns) hashes to exactly the same digest.
Datetimes are hashed as UTC ISO strings, dates as ISO dates.

One current call per (kind, target, session): `record_market_call` checks for an existing,
non-superseded record *inside* the ledger lock and raises `LedgerConflict` instead of
appending a duplicate, so two overlapping 07:30 runs cannot both record. A correction
must name a record of the same target and session that nobody has corrected yet.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import exists, select, text
from sqlalchemy.orm import Session, aliased

from app.config import Settings
from app.ml.core import CLASSES, MarketCallRecordIn, finite, is_late
from app.models import Prediction, PredictionGrade

GENESIS = "0" * 64
LOCK_SQL = text("SELECT pg_advisory_xact_lock(hashtext('prediction_ledger'))")

# The record fields covered by the hash, in the order they are documented (the digest
# itself is order-independent because canonical_json sorts keys).
HASHED_FIELDS: tuple[str, ...] = (
    "seq",
    "kind",
    "target",
    "session_date",
    "as_of",
    "is_late",
    "model_version_id",
    "reference_date",
    "reference_close",
    "prob_up",
    "prob_flat",
    "prob_down",
    "predicted_class",
    "range_p10_pct",
    "range_p90_pct",
    "gap_call",
    "gap_expected_pct",
    "confidence",
    "no_edge",
    "drivers",
    "timing_guidance",
    "baselines",
    "feature_snapshot",
    "corrects_prediction_id",
    "note",
    "prev_hash",
)
_DATE_FIELDS = {"session_date", "reference_date"}
_DATETIME_FIELDS = {"as_of"}


class LedgerConflict(Exception):
    """A current (non-superseded) record already exists for this kind, target and session.

    Raised under the ledger lock, so the caller can trust `existing` is the row the
    other writer committed; the caller reports "already recorded" and moves on.
    """

    def __init__(self, existing: Prediction) -> None:
        super().__init__(
            f"{existing.kind} for {existing.target} on {existing.session_date.isoformat()} "
            f"already recorded as seq {existing.seq}"
        )
        self.existing = existing


@dataclass(frozen=True)
class LedgerVerification:
    """Result of walking the whole chain. `first_bad_seq` is None when the ledger is intact."""

    ok: bool
    checked: int
    first_bad_seq: int | None
    message: str


def canonical_json(obj: object) -> str:
    """The one serialisation that is hashed: sorted keys, compact, ASCII, no NaN/inf."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def record_hash(fields: Mapping[str, object]) -> str:
    """SHA-256 hex digest of the canonical JSON of `fields`."""
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()


def normalise(value: object) -> Any:
    """Make a value storable in a JSON column *and* identical after a round trip.

    Floats (and numpy numbers) go through `core.finite()`; bools and ints are kept as they
    are (JSON distinguishes `1` from `1.0`, so the two must never be mixed up); dates and
    datetimes become ISO strings (datetimes in UTC); containers are normalised recursively.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float | numbers.Number):
        return finite(value)
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [normalise(v) for v in value]
    raise TypeError(f"cannot store a {type(value).__name__} in the ledger")


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("ledger timestamps must be timezone-aware")
    return dt.astimezone(UTC).isoformat()


def hashed_fields(p: Prediction | Mapping[str, object]) -> dict[str, Any]:
    """Exactly the fields covered by the hash, normalised as they are stored.

    Accepts an ORM row or any mapping with the same attribute names (for example a
    modified copy of a previous `hashed_fields` result). Idempotent: feeding the result
    back in returns an equal dict.
    """
    get = p.get if isinstance(p, Mapping) else lambda k, _d=None: getattr(p, k)
    out: dict[str, Any] = {}
    for name in HASHED_FIELDS:
        v = get(name)
        if name in _DATE_FIELDS and isinstance(v, datetime):
            v = v.date()
        if name in _DATETIME_FIELDS and isinstance(v, str):
            # Already an ISO string (a re-hashed copy); keep it exactly as given.
            out[name] = v
            continue
        out[name] = normalise(v)
    return out


def _check_probs(probs: Mapping[str, float]) -> dict[str, float]:
    """Validate the model's probabilities instead of silently fixing them."""
    out: dict[str, float] = {}
    for c in CLASSES:
        v = finite(probs.get(c))
        if v is None or v < 0.0 or v > 1.0:
            raise ValueError(f"probability for {c} missing or out of range: {probs.get(c)!r}")
        out[c] = v
    if abs(sum(out.values()) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {sum(out.values())!r}")
    return out


def record_market_call(
    db: Session,
    rec: MarketCallRecordIn,
    *,
    kind: str = "market_call",
    settings: Settings | None = None,
) -> Prediction:
    """Append one record to the ledger and return the flushed row (the caller commits).

    Takes a transaction-scoped advisory lock so two writers cannot pick the same `seq`
    or the same `prev_hash`; the second one waits until the first commits or rolls back.
    Under the same lock, a non-correction is refused with `LedgerConflict` when a current
    record for (kind, target, session_date) already exists, and a correction is validated
    against the record it names (ValueError). Never touches an existing row.
    """
    if rec.as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    reference_close = finite(rec.reference_close)
    if reference_close is None or reference_close <= 0:
        raise ValueError(f"reference_close missing or not positive: {rec.reference_close!r}")
    probs = _check_probs(rec.output.probs)
    if rec.output.predicted_class not in CLASSES:
        raise ValueError(f"unknown predicted_class {rec.output.predicted_class!r}")

    db.execute(LOCK_SQL)
    if rec.corrects_prediction_id is None:
        existing = _current(db, kind, rec.target, rec.session_date)
        if existing is not None:
            raise LedgerConflict(existing)
    else:
        _check_correction(db, rec)
    last = db.scalars(select(Prediction).order_by(Prediction.seq.desc()).limit(1)).first()
    seq = (last.seq if last is not None else 0) + 1
    prev_hash = last.hash if last is not None else GENESIS

    row = Prediction(
        seq=seq,
        kind=kind,
        target=rec.target,
        session_date=rec.session_date,
        as_of=rec.as_of,
        is_late=is_late(rec.as_of, rec.session_date, settings),
        model_version_id=rec.model_version_id,
        reference_date=rec.reference_date,
        reference_close=reference_close,
        prob_up=probs["UP"],
        prob_flat=probs["FLAT"],
        prob_down=probs["DOWN"],
        predicted_class=rec.output.predicted_class,
        range_p10_pct=finite(rec.output.range_p10_pct),
        range_p90_pct=finite(rec.output.range_p90_pct),
        gap_call=rec.gap_call,
        gap_expected_pct=finite(rec.gap_expected_pct),
        confidence=rec.output.confidence,
        no_edge=bool(rec.output.no_edge),
        drivers=normalise([d.as_dict() for d in rec.output.drivers]),
        timing_guidance=rec.timing_guidance,
        baselines=normalise(dict(rec.baselines)),
        feature_snapshot=normalise(dict(rec.feature_snapshot)),
        corrects_prediction_id=rec.corrects_prediction_id,
        note=rec.note,
        prev_hash=prev_hash,
    )
    row.hash = record_hash(hashed_fields(row))
    db.add(row)
    db.flush()
    return row


def _superseded() -> Any:
    """SQL predicate: another row corrects this one."""
    later = aliased(Prediction)
    return exists().where(later.corrects_prediction_id == Prediction.id)


def _current(db: Session, kind: str, target: str, session_date: date) -> Prediction | None:
    """The non-superseded record of one kind for a target and session (highest `seq`)."""
    return db.scalars(
        select(Prediction)
        .where(
            Prediction.kind == kind,
            Prediction.target == target,
            Prediction.session_date == session_date,
            ~_superseded(),
        )
        .order_by(Prediction.seq.desc())
        .limit(1)
    ).first()


def _check_correction(db: Session, rec: MarketCallRecordIn) -> None:
    """A correction must point at an existing record of the same target and session that
    has not been corrected already; anything else is a caller bug, not a ledger entry."""
    original = db.get(Prediction, rec.corrects_prediction_id)
    if original is None:
        raise ValueError(f"corrects_prediction_id {rec.corrects_prediction_id} does not exist")
    if original.target != rec.target or original.session_date != rec.session_date:
        raise ValueError(
            f"correction targets {rec.target} {rec.session_date.isoformat()} but seq "
            f"{original.seq} is {original.target} {original.session_date.isoformat()}"
        )
    superseded = db.scalar(
        select(Prediction.seq).where(Prediction.corrects_prediction_id == original.id).limit(1)
    )
    if superseded is not None:
        raise ValueError(f"seq {original.seq} is already corrected by seq {superseded}")


def latest_call(db: Session, target: str, session_date: date) -> Prediction | None:
    """The current call for a target and session: highest `seq` that no correction supersedes."""
    return db.scalars(
        select(Prediction)
        .where(
            Prediction.target == target,
            Prediction.session_date == session_date,
            ~_superseded(),
        )
        .order_by(Prediction.seq.desc())
        .limit(1)
    ).first()


def head(db: Session) -> Prediction | None:
    """The last record in the chain (highest `seq`), or None for an empty ledger."""
    return db.scalars(select(Prediction).order_by(Prediction.seq.desc()).limit(1)).first()


def calls_for_session(db: Session, session_date: date) -> list[Prediction]:
    """Every record for a session (corrections and originals alike), in ledger order."""
    return list(
        db.scalars(
            select(Prediction)
            .where(Prediction.session_date == session_date)
            .order_by(Prediction.seq.asc())
        ).all()
    )


def ungraded(db: Session, *, through: date | None = None) -> list[Prediction]:
    """Records still waiting for a close grade: no grade row, or one without `actual_close`.

    Leaves out records a correction superseded (the correction is graded instead) and
    records that were voided (`grading.void_grade`: `actual_close` NULL with an
    `error_category` saying why no close will ever come).
    """
    stmt = (
        select(Prediction)
        .outerjoin(PredictionGrade, PredictionGrade.prediction_id == Prediction.id)
        .where(
            PredictionGrade.actual_close.is_(None),
            PredictionGrade.error_category.is_(None),
            ~_superseded(),
        )
        .order_by(Prediction.seq.asc())
    )
    if through is not None:
        stmt = stmt.where(Prediction.session_date <= through)
    return list(db.scalars(stmt).all())


def history(
    db: Session, target: str | None = None, limit: int = 50
) -> list[tuple[Prediction, PredictionGrade | None]]:
    """Recent records with their grade (if any), newest `as_of` first."""
    stmt = (
        select(Prediction, PredictionGrade)
        .outerjoin(PredictionGrade, PredictionGrade.prediction_id == Prediction.id)
        .order_by(Prediction.as_of.desc(), Prediction.seq.desc())
        .limit(limit)
    )
    if target is not None:
        stmt = stmt.where(Prediction.target == target)
    return [(p, g) for p, g in db.execute(stmt).all()]


def verify_ledger(db: Session) -> LedgerVerification:
    """Walk every row by `seq` and re-derive the chain.

    A row is bad when its `seq` is not the previous one plus 1, its `prev_hash` is not
    the previous row's `hash` (GENESIS for seq 1), or its stored `hash` differs from the
    digest recomputed from its content. The walk stops at the first bad row.
    """
    # populate_existing: an audit must read the database, not objects cached in the session.
    rows = db.scalars(
        select(Prediction).order_by(Prediction.seq.asc()).execution_options(populate_existing=True)
    ).all()
    prev_hash = GENESIS
    prev_seq = 0
    for row in rows:
        if row.seq != prev_seq + 1:
            return LedgerVerification(
                False, prev_seq, row.seq, f"seq {row.seq} follows seq {prev_seq}: gap or duplicate"
            )
        if row.prev_hash != prev_hash:
            return LedgerVerification(
                False, prev_seq, row.seq, f"seq {row.seq}: prev_hash does not match seq {prev_seq}"
            )
        if record_hash(hashed_fields(row)) != row.hash:
            return LedgerVerification(
                False, prev_seq, row.seq, f"seq {row.seq}: stored hash does not match its content"
            )
        prev_hash, prev_seq = row.hash, row.seq
    n = len(rows)
    msg = "ledger is empty" if n == 0 else f"{n} record{'s' if n != 1 else ''} verified"
    return LedgerVerification(True, n, None, msg)

"""The immutable, hash-chained ledgers (brief §5): market calls and stock ideas.

Two chains, one set of rules. Every market call is one `Prediction` row on the chain
named "prediction"; every stock idea is one `Idea` row on the chain named "idea". Rows
are never updated or deleted (database triggers from migrations 0003 and 0004 reject
both); corrections are new rows that point at the original through
`corrects_prediction_id` / `corrects_idea_id`. Each row carries a SHA-256 over its own
content plus the previous row's hash on the same chain, so each chain (ordered by its own
`seq`) makes any later edit, insertion or removal detectable by `verify_ledger`.

What exactly is hashed is fixed per chain (`HASHED_FIELDS`, `IDEA_HASHED_FIELDS`) and by
`canonical_json`: sorted keys, no whitespace, ASCII only, NaN/inf forbidden. Every float
goes through `core.finite()` (10 decimals) before it is stored *and* before it is hashed,
so a row re-read from Postgres (double precision columns, JSON columns) hashes to exactly
the same digest. Datetimes are hashed as UTC ISO strings, dates as ISO dates.

One current call per (kind, target, session) and one current idea per (symbol, horizon,
session): `record_market_call` and `record_idea` check for an existing, non-superseded
record *inside* their chain's advisory lock (`hashtext('prediction_ledger')` and
`hashtext('idea_ledger')`) and raise `LedgerConflict` instead of appending a duplicate,
so two overlapping 07:30 runs cannot both record. A correction must name a record of the
same call (or the same symbol and horizon) and session that nobody has corrected yet.

The "prediction" chain's functions keep their Phase 2 signatures and behaviour exactly;
the idea chain adds `record_idea`, `latest_idea`, `ideas_for_session`, `open_ideas`,
`idea_history`, `head_idea` and `verify_ledger(db, chain="idea")`.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import exists, or_, select, text
from sqlalchemy.orm import Session, aliased

from app.config import Settings
from app.ml.core import CLASSES, IdeaRecordIn, MarketCallRecordIn, finite, horizons, is_late
from app.models import Idea, IdeaOutcome, Prediction, PredictionGrade

GENESIS = "0" * 64
LOCK_SQL = text("SELECT pg_advisory_xact_lock(hashtext('prediction_ledger'))")
IDEA_LOCK_SQL = text("SELECT pg_advisory_xact_lock(hashtext('idea_ledger'))")
CHAIN_NAMES: tuple[str, ...] = ("prediction", "idea")

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
# Every column of `idea` except id, hash and created_at: the whole record is signed.
IDEA_HASHED_FIELDS: tuple[str, ...] = (
    "seq",
    "session_date",
    "as_of",
    "is_late",
    "instrument_id",
    "symbol",
    "exchange",
    "horizon",
    "horizon_days",
    "setup",
    "direction",
    "entry_low",
    "entry_high",
    "stop",
    "target_1",
    "target_2",
    "p_target",
    "expected_return_low_pct",
    "expected_return_high_pct",
    "risk_reward",
    "ev_pct_after_costs",
    "position_qty",
    "position_value",
    "risk_amount",
    "reasons",
    "invalidation",
    "drivers",
    "feature_snapshot",
    "model_version_id",
    "strategy",
    "rank",
    "surveillance_unknown",
    "corrects_idea_id",
    "note",
    "prev_hash",
)
_DATE_FIELDS = {"session_date", "reference_date"}
_DATETIME_FIELDS = {"as_of"}
# Pending ideas have no outcome row at all, or one with status "pending".
OPEN_STATUSES: tuple[str, ...] = ("pending", "open")


@dataclass(frozen=True)
class Chain:
    """One hash chain: its table, lock and the fields its hash covers."""

    name: str
    model: type[Prediction] | type[Idea]
    lock_sql: Any
    hashed_fields: tuple[str, ...]


CHAINS: dict[str, Chain] = {
    "prediction": Chain("prediction", Prediction, LOCK_SQL, HASHED_FIELDS),
    "idea": Chain("idea", Idea, IDEA_LOCK_SQL, IDEA_HASHED_FIELDS),
}


def chain_spec(chain: str) -> Chain:
    """The `Chain` named `chain`, or ValueError for a name that is not a ledger."""
    try:
        return CHAINS[chain]
    except KeyError:
        raise ValueError(f"unknown ledger chain {chain!r}; one of {', '.join(CHAINS)}") from None


class LedgerConflict[R: (Prediction, Idea)](Exception):
    """A current (non-superseded) record already exists for this call or idea and session.

    Raised under the chain's lock, so the caller can trust `existing` is the row the
    other writer committed (a `Prediction` from `record_market_call`, an `Idea` from
    `record_idea`); the caller reports "already recorded" and moves on.
    """

    def __init__(self, existing: R) -> None:
        if isinstance(existing, Idea):
            what = f"idea for {existing.symbol} ({existing.horizon})"
        else:
            what = f"{existing.kind} for {existing.target}"
        super().__init__(
            f"{what} on {existing.session_date.isoformat()} already recorded as seq {existing.seq}"
        )
        self.existing: R = existing


@dataclass(frozen=True)
class LedgerVerification:
    """Result of walking one whole chain. `first_bad_seq` is None when the chain is intact."""

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


def chain_of(p: Prediction | Idea | Mapping[str, object]) -> str:
    """Which chain a row or a hashed-fields mapping belongs to.

    A mapping is an idea's when it carries the idea-only key "horizon" and not the
    call-only key "kind"; a mapping with neither is treated as a prediction's (the Phase 2
    behaviour).
    """
    if isinstance(p, Idea):
        return "idea"
    if isinstance(p, Prediction):
        return "prediction"
    return "idea" if "horizon" in p and "kind" not in p else "prediction"


def hashed_fields(
    p: Prediction | Idea | Mapping[str, object], chain: str | None = None
) -> dict[str, Any]:
    """Exactly the fields covered by the hash, normalised as they are stored.

    Accepts an ORM row of either chain or any mapping with the same attribute names (for
    example a modified copy of a previous `hashed_fields` result); the chain is inferred
    (`chain_of`) unless named. Idempotent: feeding the result back in returns an equal dict.
    """
    spec = chain_spec(chain or chain_of(p))
    get = p.get if isinstance(p, Mapping) else lambda k, _d=None: getattr(p, k)
    out: dict[str, Any] = {}
    for name in spec.hashed_fields:
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


# ============================================================================= predictions


def record_market_call(
    db: Session,
    rec: MarketCallRecordIn,
    *,
    kind: str = "market_call",
    settings: Settings | None = None,
) -> Prediction:
    """Append one record to the prediction ledger and return the flushed row (the caller
    commits).

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
    """The last record in the prediction chain (highest `seq`), or None for an empty ledger."""
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


# ============================================================================= ideas


def _positive(value: object, name: str) -> float:
    v = finite(value)
    if v is None or v <= 0.0:
        raise ValueError(f"{name} missing or not positive: {value!r}")
    return v


def _non_negative(value: object, name: str) -> float:
    v = finite(value)
    if v is None or v < 0.0:
        raise ValueError(f"{name} missing or negative: {value!r}")
    return v


def _finite_or_fail(value: object, name: str) -> float:
    v = finite(value)
    if v is None:
        raise ValueError(f"{name} missing or not finite: {value!r}")
    return v


def _check_idea(rec: IdeaRecordIn, settings: Settings | None) -> dict[str, Any]:
    """Validate an idea's numbers instead of silently fixing them; the checked values."""
    out = rec.output
    if rec.as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if out.horizon not in horizons(settings):
        raise ValueError(f"unknown horizon {out.horizon!r}")
    if out.horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {out.horizon_days!r}")
    if not out.symbol or not out.exchange:
        raise ValueError("an idea needs a symbol and an exchange")
    if not out.setup or not out.strategy:
        raise ValueError("an idea needs a setup and a strategy")
    entry_low = _positive(out.entry_low, "entry_low")
    entry_high = _positive(out.entry_high, "entry_high")
    if entry_low > entry_high:
        raise ValueError(f"entry zone is inverted: {entry_low} > {entry_high}")
    stop = _positive(out.stop, "stop")
    if stop >= entry_low:
        raise ValueError(f"stop {stop} must be below the entry zone [{entry_low}, {entry_high}]")
    target_1 = _positive(out.target_1, "target_1")
    if target_1 <= entry_high:
        raise ValueError(f"target_1 {target_1} must be above the entry zone")
    target_2 = finite(out.target_2)
    if out.target_2 is not None and (target_2 is None or target_2 <= target_1):
        raise ValueError(f"target_2 {out.target_2!r} must be above target_1 {target_1}")
    p_target = finite(out.p_target)
    if p_target is None or p_target < 0.0 or p_target > 1.0:
        raise ValueError(f"p_target missing or out of range: {out.p_target!r}")
    if not isinstance(rec.position_qty, int) or isinstance(rec.position_qty, bool):
        raise ValueError(f"position_qty must be an int, got {rec.position_qty!r}")
    if rec.position_qty < 0:
        raise ValueError(f"position_qty must not be negative, got {rec.position_qty!r}")
    if not isinstance(rec.rank, int) or isinstance(rec.rank, bool) or rec.rank < 1:
        raise ValueError(f"rank must be a positive int, got {rec.rank!r}")
    if not isinstance(rec.instrument_id, int) or isinstance(rec.instrument_id, bool):
        raise ValueError(f"instrument_id must be an int, got {rec.instrument_id!r}")
    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "p_target": p_target,
        "expected_return_low_pct": _finite_or_fail(
            out.expected_return_low_pct, "expected_return_low_pct"
        ),
        "expected_return_high_pct": _finite_or_fail(
            out.expected_return_high_pct, "expected_return_high_pct"
        ),
        "risk_reward": _positive(out.risk_reward, "risk_reward"),
        "ev_pct_after_costs": _finite_or_fail(out.ev_pct_after_costs, "ev_pct_after_costs"),
        "position_value": _non_negative(rec.position_value, "position_value"),
        "risk_amount": _non_negative(rec.risk_amount, "risk_amount"),
    }


def record_idea(db: Session, rec: IdeaRecordIn, *, settings: Settings | None = None) -> Idea:
    """Append one stock idea to the idea ledger and return the flushed row (the caller
    commits).

    Same rules as `record_market_call`, on the idea chain: the `hashtext('idea_ledger')`
    advisory lock, `seq` and `prev_hash` continue the idea chain only, every float goes
    through `finite()`, and the hash covers every column (`IDEA_HASHED_FIELDS`). Under
    the lock a non-correction is refused with `LedgerConflict` when a current idea for
    (symbol, horizon, session_date) already exists; a correction must name a current idea
    for the same symbol, horizon and session (ValueError otherwise). Prices are checked
    for sense (stop below the entry zone, target above it, p_target in [0, 1]); a missing
    number is an error, never a default. `surveillance_unknown` is stored as given: the
    engine sets True because no ASM/GSM source is reachable, and the UI says so.
    """
    checked = _check_idea(rec, settings)
    out = rec.output

    db.execute(IDEA_LOCK_SQL)
    if rec.corrects_idea_id is None:
        existing = latest_idea(db, out.symbol, out.horizon, rec.session_date)
        if existing is not None:
            raise LedgerConflict(existing)
    else:
        _check_idea_correction(db, rec)
    last = db.scalars(select(Idea).order_by(Idea.seq.desc()).limit(1)).first()
    seq = (last.seq if last is not None else 0) + 1
    prev_hash = last.hash if last is not None else GENESIS

    row = Idea(
        seq=seq,
        session_date=rec.session_date,
        as_of=rec.as_of,
        is_late=is_late(rec.as_of, rec.session_date, settings),
        instrument_id=rec.instrument_id,
        symbol=out.symbol,
        exchange=out.exchange,
        horizon=out.horizon,
        horizon_days=int(out.horizon_days),
        setup=out.setup,
        direction="long",
        entry_low=checked["entry_low"],
        entry_high=checked["entry_high"],
        stop=checked["stop"],
        target_1=checked["target_1"],
        target_2=checked["target_2"],
        p_target=checked["p_target"],
        expected_return_low_pct=checked["expected_return_low_pct"],
        expected_return_high_pct=checked["expected_return_high_pct"],
        risk_reward=checked["risk_reward"],
        ev_pct_after_costs=checked["ev_pct_after_costs"],
        position_qty=rec.position_qty,
        position_value=checked["position_value"],
        risk_amount=checked["risk_amount"],
        reasons=[str(r) for r in out.reasons],
        invalidation=str(out.invalidation),
        drivers=normalise([d.as_dict() for d in out.drivers]),
        feature_snapshot=normalise(dict(rec.feature_snapshot)),
        model_version_id=rec.model_version_id,
        strategy=out.strategy,
        rank=rec.rank,
        surveillance_unknown=bool(rec.surveillance_unknown),
        corrects_idea_id=rec.corrects_idea_id,
        note=rec.note,
        prev_hash=prev_hash,
    )
    row.hash = record_hash(hashed_fields(row))
    db.add(row)
    db.flush()
    return row


def _idea_superseded() -> Any:
    """SQL predicate: another idea corrects this one."""
    later = aliased(Idea)
    return exists().where(later.corrects_idea_id == Idea.id)


def not_superseded_idea() -> Any:
    """SQL predicate: no correction points at this `Idea` row (for aggregates)."""
    return ~_idea_superseded()


def _check_idea_correction(db: Session, rec: IdeaRecordIn) -> None:
    original = db.get(Idea, rec.corrects_idea_id)
    if original is None:
        raise ValueError(f"corrects_idea_id {rec.corrects_idea_id} does not exist")
    out = rec.output
    if (
        original.symbol != out.symbol
        or original.horizon != out.horizon
        or original.session_date != rec.session_date
    ):
        raise ValueError(
            f"correction targets {out.symbol} {out.horizon} {rec.session_date.isoformat()} "
            f"but idea seq {original.seq} is {original.symbol} {original.horizon} "
            f"{original.session_date.isoformat()}"
        )
    superseded = db.scalar(select(Idea.seq).where(Idea.corrects_idea_id == original.id).limit(1))
    if superseded is not None:
        raise ValueError(f"idea seq {original.seq} is already corrected by seq {superseded}")


def latest_idea(db: Session, symbol: str, horizon: str, session_date: date) -> Idea | None:
    """The current idea for a symbol, horizon and session: highest `seq` that no correction
    supersedes."""
    return db.scalars(
        select(Idea)
        .where(
            Idea.symbol == symbol,
            Idea.horizon == horizon,
            Idea.session_date == session_date,
            ~_idea_superseded(),
        )
        .order_by(Idea.seq.desc())
        .limit(1)
    ).first()


def head_idea(db: Session) -> Idea | None:
    """The last record in the idea chain (highest `seq`), or None for an empty ledger."""
    return db.scalars(select(Idea).order_by(Idea.seq.desc()).limit(1)).first()


def ideas_for_session(db: Session, session_date: date, *, current_only: bool = False) -> list[Idea]:
    """Every idea recorded for a session, in ledger order (by `seq`).

    By default corrections and the originals they supersede are both listed (the ledger
    is complete); `current_only=True` keeps only the ideas no correction supersedes, the
    set the Today page shows.
    """
    stmt = select(Idea).where(Idea.session_date == session_date)
    if current_only:
        stmt = stmt.where(~_idea_superseded())
    return list(db.scalars(stmt.order_by(Idea.seq.asc())).all())


def open_ideas(db: Session, *, through: date | None = None) -> list[Idea]:
    """Current ideas whose outcome is still pending or open, in ledger order.

    "Pending" is an idea with no `idea_outcome` row yet or one whose status is "pending";
    "open" is a filled position. Ideas a correction superseded are left out (the
    correction is graded instead), as are invalidated, voided and graded ones. `through`
    keeps ideas whose `session_date` is on or before that date (the open check fills the
    session's own ideas; the close grading walks everything up to today).
    """
    stmt = (
        select(Idea)
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .where(
            or_(IdeaOutcome.id.is_(None), IdeaOutcome.status.in_(OPEN_STATUSES)),
            ~_idea_superseded(),
        )
        .order_by(Idea.seq.asc())
    )
    if through is not None:
        stmt = stmt.where(Idea.session_date <= through)
    return list(db.scalars(stmt).all())


def idea_history(
    db: Session, horizon: str | None = None, limit: int = 50
) -> list[tuple[Idea, IdeaOutcome | None]]:
    """Recent ideas with their outcome (if any), newest `as_of` first, then highest seq."""
    stmt = (
        select(Idea, IdeaOutcome)
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .order_by(Idea.as_of.desc(), Idea.seq.desc())
        .limit(limit)
    )
    if horizon is not None:
        stmt = stmt.where(Idea.horizon == horizon)
    return [(i, o) for i, o in db.execute(stmt).all()]


# ============================================================================= verification


def verify_ledger(db: Session, chain: str = "prediction") -> LedgerVerification:
    """Walk every row of one chain by `seq` and re-derive it.

    A row is bad when its `seq` is not the previous one plus 1, its `prev_hash` is not
    the previous row's `hash` (GENESIS for seq 1), or its stored `hash` differs from the
    digest recomputed from its content. The walk stops at the first bad row. `chain` is
    "prediction" (the default: the market-call ledger, exactly as in Phase 2) or "idea".
    """
    spec = chain_spec(chain)
    # populate_existing: an audit must read the database, not objects cached in the session.
    rows: Sequence[Prediction | Idea]
    if spec.name == "idea":
        rows = db.scalars(
            select(Idea).order_by(Idea.seq.asc()).execution_options(populate_existing=True)
        ).all()
    else:
        rows = db.scalars(
            select(Prediction)
            .order_by(Prediction.seq.asc())
            .execution_options(populate_existing=True)
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
        if record_hash(hashed_fields(row, spec.name)) != row.hash:
            return LedgerVerification(
                False, prev_seq, row.seq, f"seq {row.seq}: stored hash does not match its content"
            )
        prev_hash, prev_seq = row.hash, row.seq
    n = len(rows)
    msg = "ledger is empty" if n == 0 else f"{n} record{'s' if n != 1 else ''} verified"
    return LedgerVerification(True, n, None, msg)


def verify_all(db: Session) -> dict[str, LedgerVerification]:
    """`verify_ledger` for every chain, keyed by chain name."""
    return {name: verify_ledger(db, name) for name in CHAIN_NAMES}

"""The hash-chained prediction ledger: chain, round trip, tamper detection, immutability."""

from __future__ import annotations

import math
import threading
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.config import IST
from app.db import get_sessionmaker
from app.ml.core import Driver, MarketCallOutput, MarketCallRecordIn
from app.models import Prediction
from app.services import ledger
from app.services.ledger import (
    GENESIS,
    canonical_json,
    hashed_fields,
    record_hash,
    record_market_call,
    verify_ledger,
)

SESSION = date(2026, 11, 2)  # a Monday
REF = date(2026, 10, 30)
AS_OF = datetime(2026, 11, 2, 7, 31, 12, 345678, tzinfo=IST)

PROBS_UP = {"UP": 0.52, "FLAT": 0.3, "DOWN": 0.18}
SNAPSHOT = {
    "ret_1d": 0.1 + 0.2,  # 0.30000000000000004 in binary; stored and hashed as 0.3
    "tiny": 1e-7,
    "neg": -12345.678901234567,
    "big": 24999.95,
    "vix": 13.37,
    "one": 1.0,
    "nan": math.nan,  # must become None, never a number
    "missing": None,
}


def make_record(
    target: str = "NIFTY 50",
    session_date: date = SESSION,
    as_of: datetime = AS_OF,
    probs: dict[str, float] | None = None,
    predicted_class: str | None = None,
    *,
    confidence: str = "medium",
    no_edge: bool = False,
    gap_call: str | None = "gap_up",
    gap_expected_pct: float | None = 0.4,
    baselines: dict[str, str | None] | None = None,
    reference_close: float = 1000.0,
    corrects: int | None = None,
    note: str | None = None,
    snapshot: dict[str, float | None] | None = None,
) -> MarketCallRecordIn:
    p = probs or PROBS_UP
    out = MarketCallOutput(
        probs=p,
        predicted_class=predicted_class or max(p, key=lambda c: p[c]),
        range_p10_pct=-0.9,
        range_p90_pct=1.1,
        drivers=[
            Driver("es_fut_ret_1d", 0.8, 0.12, "US futures up 0.8 %"),
            Driver("vix", 13.37, -0.05, "VIX at 13.4"),
        ],
        confidence=confidence,
        no_edge=no_edge,
    )
    return MarketCallRecordIn(
        target=target,
        session_date=session_date,
        as_of=as_of,
        model_version_id=None,
        reference_date=REF,
        reference_close=reference_close,
        output=out,
        gap_call=gap_call,
        gap_expected_pct=gap_expected_pct,
        timing_guidance="Gap-up likely; avoid buying the first 15 minutes.",
        baselines=baselines
        or {
            "always_up": "UP",
            "same_as_yesterday": "DOWN",
            "overnight_gap": "UP",
            "random": None,
        },
        feature_snapshot=snapshot if snapshot is not None else dict(SNAPSHOT),
        corrects_prediction_id=corrects,
        note=note,
    )


# ------------------------------------------------------------------ pure helpers


def test_canonical_json_is_sorted_compact_and_rejects_nan() -> None:
    assert canonical_json({"b": 1, "a": [1.5, None, "x"]}) == '{"a":[1.5,null,"x"],"b":1}'
    assert canonical_json({"s": "é"}) == '{"s":"\\u00e9"}'
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_normalise_keeps_ints_and_bools_but_rounds_floats() -> None:
    out = ledger.normalise({"i": 3, "b": True, "f": 0.1 + 0.2, "n": math.nan, "d": REF})
    assert out == {"i": 3, "b": True, "f": 0.3, "n": None, "d": "2026-10-30"}
    assert isinstance(out["i"], int) and isinstance(out["b"], bool)
    with pytest.raises(ValueError):
        ledger.normalise(datetime(2026, 1, 1))  # naive datetime is ambiguous


def test_hashed_fields_is_idempotent_and_covers_exactly_the_contract(db: Session) -> None:
    row = record_market_call(db, make_record())
    fields = hashed_fields(row)
    assert tuple(fields) == ledger.HASHED_FIELDS
    assert fields["as_of"] == AS_OF.astimezone(UTC).isoformat()
    assert fields["session_date"] == "2026-11-02"
    assert fields["feature_snapshot"]["ret_1d"] == 0.3
    assert fields["feature_snapshot"]["nan"] is None
    assert hashed_fields(fields) == fields
    assert "hash" not in fields and "created_at" not in fields and "id" not in fields


# ------------------------------------------------------------------ the chain


def test_hash_chain_over_three_records(db: Session) -> None:
    rows = [
        record_market_call(db, make_record()),
        record_market_call(db, make_record(target="NIFTY BANK")),
        record_market_call(db, make_record(session_date=SESSION + timedelta(days=1))),
    ]
    db.commit()
    assert [r.seq for r in rows] == [1, 2, 3]
    assert rows[0].prev_hash == GENESIS
    assert rows[1].prev_hash == rows[0].hash
    assert rows[2].prev_hash == rows[1].hash
    assert len({r.hash for r in rows}) == 3
    for r in rows:
        assert len(r.hash) == 64
        assert record_hash(hashed_fields(r)) == r.hash
    v = verify_ledger(db)
    assert v.ok and v.checked == 3 and v.first_bad_seq is None
    assert "3 records" in v.message


def test_empty_ledger_verifies(db: Session) -> None:
    v = verify_ledger(db)
    assert v.ok and v.checked == 0 and v.first_bad_seq is None


def test_round_trip_through_postgres_hashes_identically(db: Session) -> None:
    """Record, commit, re-read in a brand-new session: the stored hash must recompute."""
    written = record_market_call(db, make_record())
    written_hash = written.hash
    written_fields = hashed_fields(written)
    db.commit()
    db.close()

    fresh = get_sessionmaker()()
    try:
        row = fresh.get(Prediction, written.id)
        assert row is not None
        assert row.hash == written_hash
        assert hashed_fields(row) == written_fields
        assert record_hash(hashed_fields(row)) == row.hash
        # The JSON columns came back with the exact values that were hashed.
        assert row.feature_snapshot["ret_1d"] == 0.3
        assert row.feature_snapshot["tiny"] == 1e-7
        assert row.feature_snapshot["neg"] == -12345.6789012346
        assert row.feature_snapshot["nan"] is None and row.feature_snapshot["missing"] is None
        assert row.baselines["random"] is None
        assert row.drivers[0]["contribution"] == 0.12
        # Timestamps survive with microseconds and the right instant.
        assert row.as_of == AS_OF
        assert verify_ledger(fresh).ok
    finally:
        fresh.close()


def test_tamper_detected_by_recomputing_a_modified_copy(db: Session) -> None:
    row = record_market_call(db, make_record())
    fields = hashed_fields(row)
    for key, value in (
        ("prob_up", fields["prob_up"] + 0.01),
        ("predicted_class", "DOWN"),
        ("feature_snapshot", {**fields["feature_snapshot"], "vix": 13.38}),
        ("prev_hash", "f" * 64),
        ("note", "edited"),
    ):
        tampered = {**fields, key: value}
        assert record_hash(tampered) != row.hash, key
    assert record_hash(dict(fields)) == row.hash


# ------------------------------------------------------------------ immutability


def test_db_trigger_rejects_update_and_delete(db: Session) -> None:
    row = record_market_call(db, make_record())
    db.commit()

    with pytest.raises(DBAPIError, match="append-only"):
        db.execute(text("UPDATE prediction SET note = 'edited' WHERE seq = 1"))
    db.rollback()
    with pytest.raises(DBAPIError, match="append-only"):
        db.execute(text("DELETE FROM prediction WHERE seq = 1"))
    db.rollback()

    # The ORM path is rejected too.
    loaded = db.get(Prediction, row.id)
    assert loaded is not None
    loaded.note = "edited through the ORM"
    with pytest.raises(DBAPIError):
        db.flush()
    db.rollback()
    reloaded = db.get(Prediction, row.id)
    assert reloaded is not None
    db.delete(reloaded)
    with pytest.raises(DBAPIError):
        db.flush()
    db.rollback()

    db.expire_all()
    kept = db.get(Prediction, row.id)
    assert kept is not None and kept.note is None
    assert verify_ledger(db).ok


def test_verify_detects_an_out_of_band_edit(db: Session) -> None:
    """Someone with enough privilege disables the trigger and edits a row: the chain tells."""
    for _ in range(3):
        record_market_call(db, make_record())
    db.commit()

    db.execute(text("ALTER TABLE prediction DISABLE TRIGGER prediction_immutable"))
    db.execute(
        text(
            "UPDATE prediction SET prob_up = prob_up + 0.02, prob_flat = prob_flat - 0.02 "
            "WHERE seq = 2"
        )
    )
    db.execute(text("ALTER TABLE prediction ENABLE TRIGGER prediction_immutable"))
    db.commit()

    v = verify_ledger(db)
    assert not v.ok
    assert v.first_bad_seq == 2 and v.checked == 1
    assert "seq 2" in v.message and "hash" in v.message


def test_verify_detects_a_broken_link_and_a_seq_gap(db: Session) -> None:
    first = record_market_call(db, make_record())
    db.commit()

    # A row appended without going through record_market_call, pointing at the wrong parent.
    rogue = Prediction(
        **{k: getattr(first, k) for k in ledger.HASHED_FIELDS if k not in ("seq", "prev_hash")},
        seq=2,
        prev_hash="a" * 64,
    )
    rogue.hash = record_hash(hashed_fields(rogue))
    db.add(rogue)
    db.commit()
    v = verify_ledger(db)
    assert not v.ok and v.first_bad_seq == 2 and "prev_hash" in v.message

    # Remove it (TRUNCATE is the only way to empty the table) and leave a gap instead.
    db.execute(text("ALTER TABLE prediction DISABLE TRIGGER prediction_immutable"))
    db.execute(text("DELETE FROM prediction WHERE seq = 2"))
    db.execute(text("ALTER TABLE prediction ENABLE TRIGGER prediction_immutable"))
    db.commit()
    gap = Prediction(
        **{k: getattr(first, k) for k in ledger.HASHED_FIELDS if k not in ("seq", "prev_hash")},
        seq=4,
        prev_hash=first.hash,
    )
    gap.hash = record_hash(hashed_fields(gap))
    db.add(gap)
    db.commit()
    v = verify_ledger(db)
    assert not v.ok and v.first_bad_seq == 4 and "gap" in v.message


# ------------------------------------------------------------------ concurrency


def test_concurrent_writers_get_consecutive_seq_and_a_linked_chain(db: Session) -> None:
    """Writer A holds the ledger lock until told to commit; writer B must wait, then follow."""
    make_session = get_sessionmaker()
    a_locked = threading.Event()
    release_a = threading.Event()
    results: dict[str, tuple[int, str, str]] = {}
    errors: list[BaseException] = []

    def writer(name: str, hold: bool) -> None:
        s = make_session()
        try:
            row = record_market_call(s, make_record(target=name))
            if hold:
                a_locked.set()
                assert release_a.wait(timeout=15)
            s.commit()
            results[name] = (row.seq, row.prev_hash, row.hash)
        except BaseException as exc:  # pragma: no cover - surfaces in the assertion below
            errors.append(exc)
            s.rollback()
        finally:
            s.close()

    a = threading.Thread(target=writer, args=("A", True))
    a.start()
    assert a_locked.wait(timeout=15)
    b = threading.Thread(target=writer, args=("B", False))
    b.start()
    b.join(timeout=1.0)
    assert b.is_alive() and "B" not in results, "B should be blocked on the ledger lock"
    release_a.set()
    a.join(timeout=15)
    b.join(timeout=15)
    assert not errors, errors
    assert results["A"][0] == 1 and results["B"][0] == 2
    assert results["A"][1] == GENESIS
    assert results["B"][1] == results["A"][2]
    assert verify_ledger(db).ok


# ------------------------------------------------------------------ reads


def test_corrections_supersede_and_history_lists_everything(db: Session) -> None:
    original = record_market_call(db, make_record())
    other = record_market_call(db, make_record(target="NIFTY BANK"))
    assert ledger.latest_call(db, "NIFTY 50", SESSION) is original

    fix = record_market_call(
        db,
        make_record(
            as_of=AS_OF + timedelta(minutes=20),
            probs={"UP": 0.4, "FLAT": 0.4, "DOWN": 0.2},
            predicted_class="FLAT",
            corrects=original.id,
            note="feature bug fixed",
        ),
    )
    db.commit()
    assert fix.corrects_prediction_id == original.id
    assert ledger.latest_call(db, "NIFTY 50", SESSION) is fix
    assert ledger.latest_call(db, "NIFTY BANK", SESSION) is other
    assert ledger.latest_call(db, "NIFTY 50", SESSION + timedelta(days=1)) is None
    assert [p.seq for p in ledger.calls_for_session(db, SESSION)] == [1, 2, 3]
    assert ledger.calls_for_session(db, SESSION + timedelta(days=1)) == []

    hist = ledger.history(db)
    assert [p.seq for p, _ in hist] == [3, 2, 1]  # as_of desc, then seq desc
    assert all(g is None for _, g in hist)
    assert [p.seq for p, _ in ledger.history(db, target="NIFTY BANK")] == [2]
    assert len(ledger.history(db, limit=1)) == 1
    assert verify_ledger(db).ok


def test_ungraded_filters_by_session_date(db: Session) -> None:
    record_market_call(db, make_record())
    record_market_call(db, make_record(session_date=SESSION + timedelta(days=1)))
    db.commit()
    assert [p.seq for p in ledger.ungraded(db)] == [1, 2]
    assert [p.seq for p in ledger.ungraded(db, through=SESSION)] == [1]
    assert ledger.ungraded(db, through=SESSION - timedelta(days=1)) == []


def test_late_flag_follows_the_lock_time(db: Session) -> None:
    on_time = record_market_call(db, make_record())
    late = record_market_call(
        db, make_record(target="NIFTY BANK", as_of=AS_OF.replace(hour=9, minute=16))
    )
    assert on_time.is_late is False and late.is_late is True


def test_record_refuses_ambiguous_or_missing_numbers(db: Session) -> None:
    with pytest.raises(ValueError, match="timezone"):
        record_market_call(db, make_record(as_of=AS_OF.replace(tzinfo=None)))
    with pytest.raises(ValueError, match="reference_close"):
        record_market_call(db, make_record(reference_close=math.nan))
    with pytest.raises(ValueError, match="sum to 1"):
        record_market_call(db, make_record(probs={"UP": 0.6, "FLAT": 0.3, "DOWN": 0.3}))
    with pytest.raises(ValueError, match="probability for DOWN"):
        record_market_call(db, make_record(probs={"UP": 0.6, "FLAT": 0.4}))
    db.rollback()
    assert ledger.ungraded(db) == []

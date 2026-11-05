"""The idea chain of the hash-chained ledger: chain, conflicts, corrections, both chains
verified, round trip through Postgres, tamper detection, immutability, reads."""

from __future__ import annotations

import math
import threading
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.config import IST
from app.db import get_sessionmaker
from app.ml.core import CostModel, Driver, IdeaOutput, IdeaRecordIn, horizons
from app.models import Idea, Instrument
from app.services import idea_grading, ledger
from app.services.ledger import (
    GENESIS,
    IDEA_HASHED_FIELDS,
    LedgerConflict,
    hashed_fields,
    record_hash,
    record_idea,
    verify_ledger,
)
from tests.test_ledger import make_record

SESSION = date(2026, 11, 2)  # a Monday
AS_OF = datetime(2026, 11, 2, 7, 35, 12, 345678, tzinfo=IST)
NOW = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)  # 09:20 IST
SNAPSHOT: dict[str, float | None] = {
    "ret_20d": 0.1 + 0.2,  # stored and hashed as 0.3
    "rsi14": 61.23456789012345,
    "atr14_pct": 2.0,
    "nan": math.nan,  # must become None, never a number
    "missing": None,
}
# A stock at 1000 with ATR(14) = 2 % of the close: zone 990..1010, swing stop 980 (1 ATR),
# target_1 1040 (2 ATR), target_2 1060 (1.5 x the target_1 distance).
CLOSE = 1000.0
ATR_PCT = 2.0


def make_instrument(db: Session, symbol: str = "RELIANCE", exchange: str = "NSE") -> Instrument:
    inst = db.scalars(
        select(Instrument).where(Instrument.symbol == symbol, Instrument.exchange == exchange)
    ).first()
    if inst is None:
        inst = Instrument(symbol=symbol, exchange=exchange, name=symbol, in_universe=True)
        db.add(inst)
        db.flush()
    return inst


def make_idea(
    instrument_id: int,
    symbol: str = "RELIANCE",
    horizon: str = "swing",
    session_date: date = SESSION,
    as_of: datetime = AS_OF,
    *,
    close: float = CLOSE,
    atr_pct: float = ATR_PCT,
    entry_low: float | None = None,
    entry_high: float | None = None,
    stop: float | None = None,
    target_1: float | None = None,
    target_2: float | None = None,
    p_target: float = 0.6,
    qty: int = 50,
    rank: int = 1,
    setup: str = "breakout",
    strategy: str = "breakout_20d",
    corrects: int | None = None,
    note: str | None = None,
    snapshot: dict[str, float | None] | None = None,
    surveillance_unknown: bool = True,
    model_version_id: int | None = None,
) -> IdeaRecordIn:
    spec = horizons()[horizon] if horizon in horizons() else horizons()["swing"]
    atr = close * atr_pct / 100.0
    lo = close - 0.5 * atr if entry_low is None else entry_low
    hi = close + 0.5 * atr if entry_high is None else entry_high
    st = close * (1 - spec.stop_atr * atr_pct / 100.0) if stop is None else stop
    t1 = close * (1 + spec.target_atr * atr_pct / 100.0) if target_1 is None else target_1
    t2 = close + 1.5 * (t1 - close) if target_2 is None else target_2
    mid = (lo + hi) / 2.0
    t_pct = (t1 / mid - 1.0) * 100.0
    s_pct = (1.0 - st / mid) * 100.0
    cost = CostModel.from_settings().round_trip_pct()
    out = IdeaOutput(
        symbol=symbol,
        exchange="NSE",
        horizon=horizon,
        horizon_days=spec.days,
        setup=setup,
        strategy=strategy,
        entry_low=lo,
        entry_high=hi,
        stop=st,
        target_1=t1,
        target_2=t2,
        p_target=p_target,
        expected_return_low_pct=t_pct,
        expected_return_high_pct=(t2 / mid - 1.0) * 100.0,
        risk_reward=t_pct / s_pct,
        ev_pct_after_costs=p_target * t_pct - (1 - p_target) * s_pct - cost,
        reasons=[
            f"Close {close:.2f} above the prior 20-session high",
            "Volume 1.4 standard deviations above its 20-session mean",
            "Above the 200-DMA",
        ],
        invalidation=f"A close below {st:.2f} (the stop) before the target.",
        drivers=[
            Driver("brk_20d_high_pct", 1.2, 0.08, "Breakout above the 20-day high by 1.2 %"),
            Driver("vol_z20", 1.4, 0.05, "Volume 1.4 sd above normal"),
        ],
    )
    return IdeaRecordIn(
        session_date=session_date,
        as_of=as_of,
        instrument_id=instrument_id,
        output=out,
        position_qty=qty,
        position_value=qty * mid,
        risk_amount=qty * (mid - st),
        rank=rank,
        feature_snapshot=dict(SNAPSHOT) if snapshot is None else snapshot,
        model_version_id=model_version_id,
        surveillance_unknown=surveillance_unknown,
        corrects_idea_id=corrects,
        note=note,
    )


# ------------------------------------------------------------------ the chain


def test_idea_chain_over_three_records_and_the_prediction_chain_stays_empty(db: Session) -> None:
    inst = make_instrument(db)
    other = make_instrument(db, "TCS")
    rows = [
        record_idea(db, make_idea(inst.id)),
        record_idea(db, make_idea(other.id, "TCS")),
        record_idea(db, make_idea(inst.id, horizon="positional")),
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
        assert r.direction == "long" and r.surveillance_unknown is True
    v = verify_ledger(db, chain="idea")
    assert v.ok and v.checked == 3 and v.first_bad_seq is None and "3 records" in v.message
    # The prediction chain is a separate ledger: still empty, still fine.
    p = verify_ledger(db)
    assert p.ok and p.checked == 0 and p.message == "ledger is empty"
    assert ledger.head(db) is None and ledger.head_idea(db) is rows[2]
    assert rows[2].horizon_days == horizons()["positional"].days


def test_chains_number_and_link_independently(db: Session) -> None:
    inst = make_instrument(db)
    call = ledger.record_market_call(db, make_record())
    idea = record_idea(db, make_idea(inst.id))
    call2 = ledger.record_market_call(db, make_record(target="NIFTY BANK"))
    idea2 = record_idea(db, make_idea(inst.id, horizon="positional"))
    db.commit()
    assert (call.seq, call2.seq) == (1, 2) and (idea.seq, idea2.seq) == (1, 2)
    assert call.prev_hash == GENESIS and idea.prev_hash == GENESIS
    assert idea2.prev_hash == idea.hash and call2.prev_hash == call.hash
    assert ledger.head(db) is call2 and ledger.head_idea(db) is idea2
    both = ledger.verify_all(db)
    assert set(both) == {"prediction", "idea"}
    assert both["prediction"].ok and both["prediction"].checked == 2
    assert both["idea"].ok and both["idea"].checked == 2
    # The prediction chain's hash still covers exactly the Phase 2 fields.
    assert tuple(hashed_fields(call)) == ledger.HASHED_FIELDS
    assert tuple(hashed_fields(idea)) == IDEA_HASHED_FIELDS
    with pytest.raises(ValueError, match="unknown ledger chain"):
        verify_ledger(db, chain="journal")


def test_hashed_fields_cover_every_idea_column_and_are_idempotent(db: Session) -> None:
    inst = make_instrument(db)
    row = record_idea(db, make_idea(inst.id))
    fields = hashed_fields(row)
    assert tuple(fields) == IDEA_HASHED_FIELDS
    assert set(IDEA_HASHED_FIELDS) == {
        c.name for c in Idea.__table__.columns if c.name not in ("id", "hash", "created_at")
    }
    assert fields["as_of"] == AS_OF.astimezone(UTC).isoformat()
    assert fields["session_date"] == "2026-11-02"
    assert fields["feature_snapshot"]["ret_20d"] == 0.3
    assert fields["feature_snapshot"]["nan"] is None
    assert fields["drivers"][0]["contribution"] == 0.08
    assert fields["prev_hash"] == GENESIS
    assert hashed_fields(fields) == fields  # inferred as the idea chain from its keys
    assert hashed_fields(fields, "idea") == fields
    assert "hash" not in fields and "created_at" not in fields and "id" not in fields


def test_round_trip_through_postgres_hashes_identically(db: Session) -> None:
    inst = make_instrument(db)
    written = record_idea(db, make_idea(inst.id))
    written_hash, written_fields = written.hash, hashed_fields(written)
    db.commit()
    db.close()

    fresh = get_sessionmaker()()
    try:
        row = fresh.get(Idea, written.id)
        assert row is not None
        assert row.hash == written_hash
        assert hashed_fields(row) == written_fields
        assert record_hash(hashed_fields(row)) == row.hash
        assert row.feature_snapshot["ret_20d"] == 0.3
        assert row.feature_snapshot["rsi14"] == 61.2345678901
        assert row.feature_snapshot["nan"] is None and row.feature_snapshot["missing"] is None
        assert row.reasons[0].startswith("Close 1000.00 above")
        assert row.drivers[1]["value"] == 1.4
        assert row.as_of == AS_OF
        assert row.entry_low == 990.0 and row.entry_high == 1010.0
        assert row.stop == 980.0 and row.target_1 == 1040.0 and row.target_2 == 1060.0
        assert verify_ledger(fresh, chain="idea").ok
    finally:
        fresh.close()


def test_tamper_detected_by_recomputing_a_modified_copy(db: Session) -> None:
    inst = make_instrument(db)
    row = record_idea(db, make_idea(inst.id))
    fields = hashed_fields(row)
    for key, value in (
        ("p_target", fields["p_target"] + 0.01),
        ("stop", 970.0),
        ("symbol", "TCS"),
        ("reasons", [*fields["reasons"], "made up later"]),
        ("feature_snapshot", {**fields["feature_snapshot"], "rsi14": 62.0}),
        ("surveillance_unknown", False),
        ("prev_hash", "f" * 64),
        ("note", "edited"),
    ):
        assert record_hash({**fields, key: value}) != row.hash, key
    assert record_hash(dict(fields)) == row.hash


# ------------------------------------------------------------------ immutability


def test_db_trigger_rejects_update_and_delete_on_idea(db: Session) -> None:
    inst = make_instrument(db)
    row = record_idea(db, make_idea(inst.id))
    db.commit()

    with pytest.raises(DBAPIError, match="append-only"):
        db.execute(text("UPDATE idea SET note = 'edited' WHERE seq = 1"))
    db.rollback()
    with pytest.raises(DBAPIError, match="append-only"):
        db.execute(text("DELETE FROM idea WHERE seq = 1"))
    db.rollback()

    loaded = db.get(Idea, row.id)
    assert loaded is not None
    loaded.p_target = 0.99
    with pytest.raises(DBAPIError):
        db.flush()
    db.rollback()
    reloaded = db.get(Idea, row.id)
    assert reloaded is not None
    db.delete(reloaded)
    with pytest.raises(DBAPIError):
        db.flush()
    db.rollback()

    db.expire_all()
    kept = db.get(Idea, row.id)
    assert kept is not None and kept.p_target == 0.6 and kept.note is None
    assert verify_ledger(db, chain="idea").ok


def test_verify_detects_an_out_of_band_edit_on_the_idea_chain(db: Session) -> None:
    inst = make_instrument(db)
    for i in range(3):
        record_idea(db, make_idea(inst.id, session_date=SESSION + timedelta(days=i)))
    ledger.record_market_call(db, make_record())
    db.commit()

    db.execute(text("ALTER TABLE idea DISABLE TRIGGER idea_immutable"))
    db.execute(text("UPDATE idea SET p_target = p_target + 0.1 WHERE seq = 2"))
    db.execute(text("ALTER TABLE idea ENABLE TRIGGER idea_immutable"))
    db.commit()

    v = verify_ledger(db, chain="idea")
    assert not v.ok and v.first_bad_seq == 2 and v.checked == 1
    assert "seq 2" in v.message and "hash" in v.message
    assert verify_ledger(db).ok, "the prediction chain is untouched"


def test_verify_detects_a_broken_link_and_a_seq_gap(db: Session) -> None:
    inst = make_instrument(db)
    first = record_idea(db, make_idea(inst.id))
    db.commit()

    def clone(seq: int, prev_hash: str) -> Idea:
        row = Idea(
            **{k: getattr(first, k) for k in IDEA_HASHED_FIELDS if k not in ("seq", "prev_hash")},
            seq=seq,
            prev_hash=prev_hash,
        )
        row.hash = record_hash(hashed_fields(row))
        return row

    db.add(clone(2, "a" * 64))
    db.commit()
    v = verify_ledger(db, chain="idea")
    assert not v.ok and v.first_bad_seq == 2 and "prev_hash" in v.message

    db.execute(text("ALTER TABLE idea DISABLE TRIGGER idea_immutable"))
    db.execute(text("DELETE FROM idea WHERE seq = 2"))
    db.execute(text("ALTER TABLE idea ENABLE TRIGGER idea_immutable"))
    db.commit()
    db.add(clone(4, first.hash))
    db.commit()
    v = verify_ledger(db, chain="idea")
    assert not v.ok and v.first_bad_seq == 4 and "gap" in v.message


# ------------------------------------------------------------------ one current idea per key


def test_duplicate_idea_is_refused_under_the_lock(db: Session) -> None:
    inst = make_instrument(db)
    first = record_idea(db, make_idea(inst.id))
    db.commit()
    with pytest.raises(LedgerConflict) as info:
        record_idea(db, make_idea(inst.id, as_of=AS_OF + timedelta(seconds=5), p_target=0.7))
    assert info.value.existing is first
    assert str(info.value) == "idea for RELIANCE (swing) on 2026-11-02 already recorded as seq 1"
    db.rollback()
    assert [i.seq for i in ledger.ideas_for_session(db, SESSION)] == [1]

    # Another horizon, another session or another symbol are all fine.
    record_idea(db, make_idea(inst.id, horizon="positional"))
    record_idea(db, make_idea(inst.id, session_date=SESSION + timedelta(days=1)))
    record_idea(db, make_idea(make_instrument(db, "TCS").id, "TCS"))
    db.commit()
    assert verify_ledger(db, chain="idea").checked == 4

    # A correction supersedes the first; a further plain record collides with the
    # correction (the current row), not with the superseded original.
    fix = record_idea(db, make_idea(inst.id, corrects=first.id, note="levels fixed"))
    db.commit()
    with pytest.raises(LedgerConflict) as info2:
        record_idea(db, make_idea(inst.id))
    assert info2.value.existing is fix
    db.rollback()


def test_corrections_must_name_a_current_idea_of_the_same_key(db: Session) -> None:
    inst = make_instrument(db)
    tcs = make_instrument(db, "TCS")
    original = record_idea(db, make_idea(inst.id))
    other = record_idea(db, make_idea(tcs.id, "TCS"))
    db.commit()
    with pytest.raises(ValueError, match="does not exist"):
        record_idea(db, make_idea(inst.id, corrects=original.id + 999))
    with pytest.raises(ValueError, match="correction targets RELIANCE swing"):
        record_idea(db, make_idea(inst.id, corrects=other.id))
    with pytest.raises(ValueError, match="correction targets"):
        record_idea(db, make_idea(inst.id, horizon="positional", corrects=original.id))
    with pytest.raises(ValueError, match="correction targets"):
        record_idea(
            db, make_idea(inst.id, session_date=SESSION + timedelta(days=1), corrects=original.id)
        )
    db.rollback()
    fix = record_idea(db, make_idea(inst.id, corrects=original.id, note="first fix"))
    db.commit()
    with pytest.raises(
        ValueError, match=f"seq {original.seq} is already corrected by seq {fix.seq}"
    ):
        record_idea(db, make_idea(inst.id, corrects=original.id, note="second fix"))
    db.rollback()
    again = record_idea(db, make_idea(inst.id, corrects=fix.id, note="second fix"))
    db.commit()
    assert ledger.latest_idea(db, "RELIANCE", "swing", SESSION) is again
    assert ledger.latest_idea(db, "TCS", "swing", SESSION) is other
    assert ledger.latest_idea(db, "RELIANCE", "positional", SESSION) is None
    assert [i.seq for i in ledger.ideas_for_session(db, SESSION)] == [1, 2, 3, 4]
    assert [i.seq for i in ledger.ideas_for_session(db, SESSION, current_only=True)] == [2, 4]
    assert verify_ledger(db, chain="idea").ok


# ------------------------------------------------------------------ validation


def test_record_refuses_ambiguous_or_missing_numbers(db: Session) -> None:
    inst = make_instrument(db)
    with pytest.raises(ValueError, match="timezone"):
        record_idea(db, make_idea(inst.id, as_of=AS_OF.replace(tzinfo=None)))
    with pytest.raises(ValueError, match="stop .* below the entry zone"):
        record_idea(db, make_idea(inst.id, stop=995.0))
    with pytest.raises(ValueError, match="target_1 .* above the entry zone"):
        record_idea(db, make_idea(inst.id, target_1=1005.0))
    with pytest.raises(ValueError, match="target_2"):
        record_idea(db, make_idea(inst.id, target_2=1030.0))
    with pytest.raises(ValueError, match="entry zone is inverted"):
        record_idea(db, make_idea(inst.id, entry_low=1011.0, entry_high=1009.0))
    with pytest.raises(ValueError, match="p_target"):
        record_idea(db, make_idea(inst.id, p_target=math.nan))
    with pytest.raises(ValueError, match="p_target"):
        record_idea(db, make_idea(inst.id, p_target=1.2))
    with pytest.raises(ValueError, match="position_qty"):
        record_idea(db, make_idea(inst.id, qty=-1))
    with pytest.raises(ValueError, match="rank"):
        record_idea(db, make_idea(inst.id, rank=0))
    with pytest.raises(ValueError, match="unknown horizon"):
        record_idea(db, make_idea(inst.id, horizon="intraday"))
    with pytest.raises(ValueError, match="entry_low"):
        record_idea(db, make_idea(inst.id, entry_low=math.inf))
    db.rollback()
    assert ledger.head_idea(db) is None and verify_ledger(db, chain="idea").checked == 0


def test_late_flag_follows_the_lock_time(db: Session) -> None:
    inst = make_instrument(db)
    on_time = record_idea(db, make_idea(inst.id))
    late = record_idea(
        db, make_idea(inst.id, horizon="positional", as_of=AS_OF.replace(hour=9, minute=16))
    )
    assert on_time.is_late is False and late.is_late is True


def test_surveillance_is_recorded_as_unknown(db: Session) -> None:
    inst = make_instrument(db)
    row = record_idea(db, make_idea(inst.id))
    assert row.surveillance_unknown is True
    assert hashed_fields(row)["surveillance_unknown"] is True


# ------------------------------------------------------------------ concurrency


def test_concurrent_idea_writers_get_consecutive_seq_and_a_linked_chain(db: Session) -> None:
    """Writer A holds the idea lock until told to commit; B must wait, then follow A."""
    inst_id = make_instrument(db).id
    tcs_id = make_instrument(db, "TCS").id
    db.commit()
    make_session = get_sessionmaker()
    a_locked = threading.Event()
    release_a = threading.Event()
    results: dict[str, tuple[int, str, str]] = {}
    errors: list[BaseException] = []

    def writer(name: str, instrument_id: int, hold: bool) -> None:
        s = make_session()
        try:
            row = record_idea(s, make_idea(instrument_id, name))
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

    a = threading.Thread(target=writer, args=("RELIANCE", inst_id, True))
    a.start()
    assert a_locked.wait(timeout=15)
    b = threading.Thread(target=writer, args=("TCS", tcs_id, False))
    b.start()
    b.join(timeout=1.0)
    assert b.is_alive() and "TCS" not in results, "B should be blocked on the idea lock"
    release_a.set()
    a.join(timeout=15)
    b.join(timeout=15)
    assert not errors, errors
    assert results["RELIANCE"][0] == 1 and results["TCS"][0] == 2
    assert results["RELIANCE"][1] == GENESIS
    assert results["TCS"][1] == results["RELIANCE"][2]
    assert verify_ledger(db, chain="idea").ok


# ------------------------------------------------------------------ reads


def test_open_ideas_history_and_head(db: Session) -> None:
    inst = make_instrument(db)
    tcs = make_instrument(db, "TCS")
    pending = record_idea(db, make_idea(inst.id))
    filled = record_idea(db, make_idea(tcs.id, "TCS", as_of=AS_OF + timedelta(minutes=1)))
    gapped = record_idea(
        db, make_idea(inst.id, horizon="positional", as_of=AS_OF + timedelta(minutes=2))
    )
    tomorrow = record_idea(
        db,
        make_idea(
            tcs.id,
            "TCS",
            session_date=SESSION + timedelta(days=1),
            as_of=AS_OF + timedelta(days=1),
        ),
    )
    superseded = record_idea(
        db,
        make_idea(
            inst.id, session_date=SESSION + timedelta(days=2), as_of=AS_OF + timedelta(days=2)
        ),
    )
    fix = record_idea(
        db,
        make_idea(
            inst.id,
            session_date=SESSION + timedelta(days=2),
            as_of=AS_OF + timedelta(days=2, minutes=5),
            corrects=superseded.id,
        ),
    )
    db.commit()
    idea_grading.open_check_fill(db, filled, 1000.0, now=NOW, source="test")
    idea_grading.open_check_fill(db, gapped, 1020.0, now=NOW, source="test")  # invalidated
    db.commit()

    assert [i.seq for i in ledger.open_ideas(db)] == [
        pending.seq,
        filled.seq,
        tomorrow.seq,
        fix.seq,
    ]
    assert [i.seq for i in ledger.open_ideas(db, through=SESSION)] == [pending.seq, filled.seq]
    assert ledger.open_ideas(db, through=SESSION - timedelta(days=1)) == []

    hist = ledger.idea_history(db)
    assert [i.seq for i, _ in hist] == [
        fix.seq,
        superseded.seq,
        tomorrow.seq,
        gapped.seq,
        filled.seq,
        pending.seq,
    ]
    by_seq = {i.seq: o for i, o in hist}
    assert by_seq[pending.seq] is None
    assert by_seq[filled.seq] is not None and by_seq[filled.seq].status == "open"
    assert by_seq[gapped.seq] is not None and by_seq[gapped.seq].status == "invalidated"
    assert [i.seq for i, _ in ledger.idea_history(db, horizon="positional")] == [gapped.seq]
    assert len(ledger.idea_history(db, limit=2)) == 2
    assert ledger.head_idea(db) is fix
    assert ledger.verify_all(db)["idea"].ok

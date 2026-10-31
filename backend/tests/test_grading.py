"""Grading math, the class threshold boundary, gap grading and the grade upsert."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.core import BASELINE_NAMES
from app.models import PredictionGrade
from app.services import grading, ledger
from app.services.grading import grade_close, grade_gap, grade_session
from app.services.ledger import record_market_call
from tests.test_ledger import SESSION, make_record

NOW = datetime(2026, 11, 2, 10, 16, tzinfo=UTC)  # 15:46 IST
OPEN_NOW = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)  # 09:20 IST
UNIFORM = {"UP": 1 / 3, "FLAT": 1 / 3, "DOWN": 1 / 3}


def _grade_rows(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(PredictionGrade)) or 0)


# ------------------------------------------------------------------ scoring math


def test_perfect_call_scores_zero(db: Session) -> None:
    p = record_market_call(
        db, make_record(probs={"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}, predicted_class="UP")
    )
    g = grade_close(db, p, None, 1010.0, now=NOW, source="test")
    assert g.actual_class == "UP" and g.correct is True
    assert g.actual_pct == pytest.approx(1.0)
    assert g.brier == 0.0
    assert g.log_loss == pytest.approx(0.0, abs=1e-9)


def test_uniform_forecast_scores_two_thirds_and_ln3(db: Session) -> None:
    p = record_market_call(db, make_record(probs=UNIFORM, predicted_class="FLAT"))
    g = grade_close(db, p, None, 1010.0, now=NOW, source="test")
    # Brier = (1/3)^2 + (1/3)^2 + (1 - 1/3)^2 = 2/3; log loss = -ln(1/3) = ln 3.
    assert g.brier == pytest.approx(2.0 / 3.0)
    assert g.log_loss == pytest.approx(math.log(3))
    assert g.correct is False and g.actual_class == "UP"


def test_confident_miss_scores_two(db: Session) -> None:
    p = record_market_call(
        db, make_record(probs={"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}, predicted_class="UP")
    )
    g = grade_close(db, p, None, 990.0, now=NOW, source="test")
    assert g.actual_class == "DOWN" and g.correct is False
    assert g.brier == pytest.approx(2.0)
    assert g.log_loss == pytest.approx(-math.log(1e-6))  # eps-clipped, never infinite


def test_brier_formula_matches_the_explicit_sum(db: Session) -> None:
    probs = {"UP": 0.52, "FLAT": 0.3, "DOWN": 0.18}
    p = record_market_call(db, make_record(probs=probs, predicted_class="UP"))
    g = grade_close(db, p, None, 1000.0, now=NOW, source="test")  # actual FLAT
    expected = (0.52 - 0) ** 2 + (0.3 - 1) ** 2 + (0.18 - 0) ** 2
    assert g.brier == pytest.approx(expected)
    assert g.log_loss == pytest.approx(-math.log(0.3))
    assert g.correct is False


# ------------------------------------------------------------------ thresholds


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        (1003.0, "FLAT"),  # exactly +0.30 % is FLAT
        (1003.1, "UP"),
        (997.0, "FLAT"),  # exactly -0.30 % is FLAT
        (996.9, "DOWN"),
        (1000.0, "FLAT"),
    ],
)
def test_threshold_boundary(db: Session, close: float, expected: str) -> None:
    p = record_market_call(db, make_record(reference_close=1000.0))
    g = grade_close(db, p, None, close, now=NOW, source="test")
    assert g.actual_class == expected
    assert g.actual_pct == pytest.approx((close / 1000.0 - 1) * 100, abs=1e-9)


def test_threshold_comes_from_settings(db: Session) -> None:
    p = record_market_call(db, make_record(reference_close=1000.0))
    wide = Settings(market_flat_threshold_pct=1.0)
    g = grade_close(db, p, None, 1008.0, now=NOW, source="test", settings=wide)
    assert g.actual_class == "FLAT"
    g = grade_close(db, p, None, 1008.0, now=NOW, source="test")
    assert g.actual_class == "UP"


# ------------------------------------------------------------------ baselines


def test_baseline_correct_marks_each_baseline_or_none(db: Session) -> None:
    p = record_market_call(
        db,
        make_record(
            baselines={
                "always_up": "UP",
                "same_as_yesterday": "DOWN",
                "overnight_gap": None,
                "random": "UP",
            }
        ),
    )
    g = grade_close(db, p, None, 1010.0, now=NOW, source="test")  # UP
    assert g.baseline_correct == {
        "always_up": True,
        "same_as_yesterday": False,
        "overnight_gap": None,
        "random": True,
    }
    assert set(g.baseline_correct or {}) == set(BASELINE_NAMES)


def test_baseline_correct_handles_a_missing_baseline_key(db: Session) -> None:
    p = record_market_call(db, make_record(baselines={"always_up": "UP"}))
    g = grade_close(db, p, None, 990.0, now=NOW, source="test")  # DOWN
    assert g.baseline_correct == {
        "always_up": False,
        "same_as_yesterday": None,
        "overnight_gap": None,
        "random": None,
    }


# ------------------------------------------------------------------ the gap call


def test_gap_grade_then_close_grade_keep_one_row(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_up"))  # threshold 0.25 %
    g1 = grade_gap(db, p, 1004.0, now=OPEN_NOW, source="kite")
    assert g1.gap_actual_pct == pytest.approx(0.4)
    assert g1.gap_correct is True and g1.gap_graded_at == OPEN_NOW
    assert g1.actual_close is None and g1.actual_class is None
    assert [q.id for q in ledger.ungraded(db)] == [p.id]  # gap alone is not a close grade

    g2 = grade_close(db, p, 1009.0, 1010.0, now=NOW, source="yfinance")
    assert g2.id == g1.id and _grade_rows(db) == 1
    # The 09:20 gap grade stands; the later open is not used to re-grade it.
    assert g2.actual_open == 1004.0 and g2.gap_actual_pct == pytest.approx(0.4)
    assert g2.gap_graded_at == OPEN_NOW and g2.graded_at == NOW
    assert g2.actual_class == "UP" and g2.correct is True and g2.source == "yfinance"
    assert ledger.ungraded(db) == []


def test_close_grade_grades_the_gap_when_nobody_did(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_down"))
    g = grade_close(db, p, 1001.0, 990.0, now=NOW, source="yfinance")
    assert g.gap_actual_pct == pytest.approx(0.1)
    assert g.gap_correct is False and g.gap_graded_at == NOW and g.actual_open == 1001.0


def test_gap_without_a_call_records_the_number_but_no_verdict(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call=None, gap_expected_pct=None))
    g = grade_gap(db, p, 1004.0, now=OPEN_NOW, source="kite")
    assert g.gap_actual_pct == pytest.approx(0.4) and g.gap_correct is None


def test_close_grade_without_open_leaves_gap_fields_empty(db: Session) -> None:
    p = record_market_call(db, make_record())
    g = grade_close(db, p, None, 1010.0, now=NOW, source="yfinance")
    assert g.actual_open is None and g.gap_actual_pct is None and g.gap_correct is None
    assert g.gap_graded_at is None


def test_regrading_overwrites_in_place(db: Session) -> None:
    p = record_market_call(db, make_record())
    first = grade_close(db, p, None, 1010.0, now=NOW, source="yfinance")
    assert first.actual_class == "UP"
    later = grade_close(db, p, None, 1009.5, now=NOW + timedelta(hours=1), source="nse_bhavcopy")
    assert later.id == first.id and _grade_rows(db) == 1
    assert later.actual_close == 1009.5 and later.source == "nse_bhavcopy"
    assert later.graded_at == NOW + timedelta(hours=1)


def test_grading_refuses_non_finite_market_numbers(db: Session) -> None:
    p = record_market_call(db, make_record())
    with pytest.raises(ValueError):
        grade_close(db, p, None, math.nan, now=NOW, source="test")
    with pytest.raises(ValueError):
        grade_gap(db, p, math.inf, now=OPEN_NOW, source="test")
    assert _grade_rows(db) == 0


# ------------------------------------------------------------------ a whole session


def test_grade_session_grades_what_it_can_and_names_the_rest(db: Session) -> None:
    nifty = record_market_call(db, make_record(target="NIFTY 50"))
    bank = record_market_call(db, make_record(target="NIFTY BANK", reference_close=50000.0))
    record_market_call(db, make_record(target="NIFTY 50", session_date=SESSION + timedelta(days=1)))
    db.commit()

    report = grade_session(
        db,
        SESSION,
        {"NIFTY 50": (1002.0, 1010.0), "NIFTY BANK": (None, math.nan)},
        now=NOW,
        source="yfinance",
    )
    assert report.graded == 1
    assert report.skipped == [f"NIFTY BANK seq {bank.seq}: no close for {SESSION.isoformat()}"]
    assert [q.id for q in ledger.ungraded(db, through=SESSION)] == [bank.id]
    hist = {p.id: g for p, g in ledger.history(db)}
    nifty_grade = hist[nifty.id]
    assert nifty_grade is not None and nifty_grade.actual_class == "UP"
    assert hist[bank.id] is None

    report2 = grade_session(db, SESSION, {}, now=NOW, source="yfinance")
    assert report2.graded == 0 and len(report2.skipped) == 2
    assert isinstance(report2, grading.GradingReport)


# ------------------------------------------------------------------ provenance


def _ist(at: datetime) -> str:
    from app.config import IST

    return at.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")


def test_close_grade_keeps_the_gap_grades_source_in_notes(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_up"))
    g1 = grade_gap(db, p, 1004.0, now=OPEN_NOW, source="kite")
    assert g1.source == "kite" and g1.notes is None

    g2 = grade_close(db, p, 1009.0, 1010.0, now=NOW, source="nse_bhavcopy")
    assert g2.source == "nse_bhavcopy", "the close's source"
    assert g2.notes == f"gap graded from kite at {_ist(OPEN_NOW)}; close from nse_bhavcopy"
    assert g2.gap_actual_pct == pytest.approx(0.4) and g2.gap_graded_at == OPEN_NOW

    # Re-grading the close later keeps the gap's provenance and updates the close's.
    g3 = grade_close(db, p, None, 1010.5, now=NOW + timedelta(hours=1), source="yfinance")
    assert g3.notes == f"gap graded from kite at {_ist(OPEN_NOW)}; close from yfinance"
    assert g3.source == "yfinance" and g3.gap_actual_pct == pytest.approx(0.4)


def test_gap_grade_after_a_close_grade_does_not_steal_its_source(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_up"))
    grade_close(db, p, None, 1010.0, now=NOW, source="nse_bhavcopy")  # no open: gap ungraded
    g = grade_gap(db, p, 1004.0, now=NOW + timedelta(days=1), source="kite")
    assert g.source == "nse_bhavcopy" and g.actual_close == 1010.0
    assert (
        g.notes
        == f"gap graded from kite at {_ist(NOW + timedelta(days=1))}; close from nse_bhavcopy"
    )
    assert g.gap_correct is True


def test_close_grade_that_grades_the_gap_itself_says_so(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_up"))
    g = grade_close(db, p, 1004.0, 1010.0, now=NOW, source="yfinance")
    assert g.notes == f"gap graded from yfinance at {_ist(NOW)}; close from yfinance"
    only_close = grade_close(
        db,
        record_market_call(db, make_record(target="NIFTY BANK")),
        None,
        1010.0,
        now=NOW,
        source="yfinance",
    )
    assert only_close.notes is None and only_close.gap_graded_at is None


# ------------------------------------------------------------------ voiding


def test_void_grade_marks_a_call_ungradeable_without_inventing_a_close(db: Session) -> None:
    p = record_market_call(db, make_record())
    g = grading.void_grade(db, p, now=NOW, reason="session never traded (ad-hoc holiday)")
    assert g.actual_close is None and g.actual_class is None and g.correct is None
    assert g.error_category == "no_session" and g.source == "manual"
    assert g.notes == "voided (no_session): session never traded (ad-hoc holiday)"
    assert g.graded_at == NOW
    assert ledger.ungraded(db) == []
    assert _grade_rows(db) == 1

    with pytest.raises(ValueError, match="category"):
        grading.void_grade(db, p, now=NOW, reason="x", category="typo")
    with pytest.raises(ValueError, match="reason"):
        grading.void_grade(db, p, now=NOW, reason="   ")

    # A real close later supersedes the void (and clears its category).
    g2 = grade_close(db, p, None, 1010.0, now=NOW + timedelta(days=1), source="nse_bhavcopy")
    assert g2.id == g.id and g2.error_category is None and g2.actual_class == "UP"
    with pytest.raises(ValueError, match="already has a close grade"):
        grading.void_grade(db, p, now=NOW, reason="too late")


def test_void_keeps_an_existing_gap_grade(db: Session) -> None:
    p = record_market_call(db, make_record(gap_call="gap_up"))
    grade_gap(db, p, 1004.0, now=OPEN_NOW, source="kite")
    g = grading.void_grade(
        db, p, now=NOW, reason="vendor bar missing for good", category="data_error"
    )
    assert g.gap_actual_pct == pytest.approx(0.4) and g.gap_graded_at == OPEN_NOW
    assert g.source == "kite", "the gap's source is kept; the void adds no close source"
    assert g.error_category == "data_error" and g.actual_close is None
    assert ledger.ungraded(db) == []

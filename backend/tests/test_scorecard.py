"""The market-call scorecard: aggregates, calibration, sample-size warning, verdict text."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import BASELINE_NAMES, CONFIDENCE_LEVELS, MODEL_NAME
from app.models import ModelVersion, Prediction
from app.services.grading import grade_close, grade_gap
from app.services.ledger import record_market_call
from app.services.scorecard import Scorecard, scorecard, verdict_text
from tests.test_ledger import AS_OF, SESSION, make_record

NOW = datetime(2026, 11, 2, 10, 16, tzinfo=UTC)
P_UP = {"UP": 0.6, "FLAT": 0.25, "DOWN": 0.15}
P_DOWN = {"UP": 0.2, "FLAT": 0.35, "DOWN": 0.45}
BASELINES = {
    "always_up": "UP",
    "same_as_yesterday": "UP",
    "overnight_gap": None,
    "random": "DOWN",
}


def _call(
    db: Session,
    i: int,
    probs: dict[str, float],
    close: float | None,
    *,
    target: str = "NIFTY 50",
    confidence: str = "medium",
    actual_open: float | None = None,
    corrects: int | None = None,
) -> Prediction:
    session = SESSION + timedelta(days=i)
    p = record_market_call(
        db,
        make_record(
            target=target,
            session_date=session,
            as_of=AS_OF + timedelta(days=i),
            probs=probs,
            confidence=confidence,
            baselines=dict(BASELINES),
            corrects=corrects,
            snapshot={"ret_1d": 0.1},
        ),
    )
    if close is not None:
        grade_close(db, p, actual_open, close, now=NOW + timedelta(days=i), source="test")
    return p


def test_empty_scorecard_is_honest_about_having_nothing(db: Session) -> None:
    sc = scorecard(db, "NIFTY 50")
    assert isinstance(sc, Scorecard)
    assert sc.target == "NIFTY 50" and sc.n_graded == 0
    assert sc.accuracy is None and sc.brier is None and sc.log_loss is None
    assert sc.gap_n == 0 and sc.gap_accuracy is None
    assert sc.by_confidence == [] and sc.calibration == [] and sc.recent == []
    assert [b["name"] for b in sc.baselines] == list(BASELINE_NAMES)
    assert all(b["n"] == 0 and b["accuracy"] is None for b in sc.baselines)
    assert sc.sample_warning is True and sc.champion is None
    assert "unproven" in sc.verdict and "No graded" in sc.verdict


def test_aggregates_over_graded_calls(db: Session) -> None:
    # 5 graded calls: UP,UP,UP correct; UP wrong (actual FLAT); DOWN correct. One ungraded.
    _call(db, 0, P_UP, 1010.0, confidence="high", actual_open=1004.0)  # gap_up correct
    _call(db, 1, P_UP, 1010.0, confidence="high", actual_open=1001.0)  # gap_up wrong
    _call(db, 2, P_UP, 1010.0, confidence="medium")
    _call(db, 3, P_UP, 1000.0, confidence="low")  # FLAT: wrong
    _call(db, 4, P_DOWN, 990.0, confidence="low")  # DOWN: correct
    _call(db, 5, P_UP, None)  # today: not graded yet
    _call(db, 0, P_UP, 1010.0, target="NIFTY BANK")  # another target: not counted
    db.commit()

    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 5
    assert sc.accuracy == pytest.approx(4 / 5)
    brier_up_right = (0.6 - 1) ** 2 + 0.25**2 + 0.15**2
    brier_up_flat = 0.6**2 + (0.25 - 1) ** 2 + 0.15**2
    brier_down_right = 0.2**2 + 0.35**2 + (0.45 - 1) ** 2
    assert sc.brier == pytest.approx((3 * brier_up_right + brier_up_flat + brier_down_right) / 5)
    assert sc.log_loss is not None and sc.log_loss > 0
    assert sc.gap_n == 2 and sc.gap_accuracy == pytest.approx(0.5)

    by_conf = {r["confidence"]: r for r in sc.by_confidence}
    assert list(by_conf) == list(CONFIDENCE_LEVELS)
    assert by_conf["high"]["n"] == 2 and by_conf["high"]["accuracy"] == 1.0
    assert by_conf["medium"]["n"] == 1 and by_conf["medium"]["accuracy"] == 1.0
    assert by_conf["low"]["n"] == 2 and by_conf["low"]["accuracy"] == 0.5
    assert by_conf["none"]["n"] == 0 and "accuracy" not in by_conf["none"]

    assert sum(b["n"] for b in sc.calibration) == 5
    bins = {(b["bin_lo"], b["bin_hi"]) for b in sc.calibration}
    assert len(bins) == 2  # top-prob 0.60 and 0.45 fall in two different bins
    for b in sc.calibration:
        assert set(b) >= {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}

    base = {b["name"]: b for b in sc.baselines}
    assert base["always_up"] == {
        "name": "always_up",
        "label": "always UP",
        "n": 5,
        "accuracy": pytest.approx(3 / 5),
    }
    assert base["overnight_gap"]["n"] == 0 and base["overnight_gap"]["accuracy"] is None
    assert base["random"]["n"] == 5 and base["random"]["accuracy"] == pytest.approx(1 / 5)

    assert sc.sample_warning is True
    assert "After 5 graded calls" in sc.verdict and "80%" in sc.verdict
    assert "always UP" in sc.verdict and "60%" in sc.verdict
    assert "ahead of the baselines" in sc.verdict
    assert f"Fewer than {get_settings().min_sample_size} graded calls" in sc.verdict

    assert len(sc.recent) == 6  # the ungraded call is listed too, newest first
    assert sc.recent[0]["session_date"] == (SESSION + timedelta(days=5)).isoformat()
    assert sc.recent[0]["actual_class"] is None and sc.recent[0]["correct"] is None
    assert sc.recent[1]["actual_class"] == "DOWN" and sc.recent[1]["correct"] is True
    assert sc.recent[-1]["probs"] == P_UP and sc.recent[-1]["gap_correct"] is True
    assert all(r["target"] == "NIFTY 50" for r in sc.recent)
    assert sc.as_dict()["n_graded"] == 5


def test_superseded_calls_count_once(db: Session) -> None:
    original = _call(db, 0, P_UP, 1010.0)
    fix = _call(db, 0, P_DOWN, 1010.0, corrects=original.id)
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 1
    assert sc.accuracy == 0.0  # the correction (DOWN) is what counts, not the original
    assert [r["id"] for r in sc.recent] == [fix.id]


def test_sample_warning_flips_at_the_configured_sample_size(db: Session) -> None:
    small = Settings(min_sample_size=3)
    for i in range(2):
        _call(db, i, P_UP, 1010.0)
    db.commit()
    sc = scorecard(db, "NIFTY 50", settings=small)
    assert sc.n_graded == 2 and sc.sample_warning is True
    assert "Fewer than 3 graded calls" in sc.verdict
    _call(db, 2, P_UP, 1010.0)
    db.commit()
    sc = scorecard(db, "NIFTY 50", settings=small)
    assert sc.n_graded == 3 and sc.sample_warning is False
    assert "Fewer than" not in sc.verdict and "unproven" not in sc.verdict
    assert get_settings().min_sample_size == 100  # the default the brief asks for


def test_sample_warning_default_flips_at_100(db: Session) -> None:
    for i in range(99):
        _call(db, i, P_UP, 1010.0 if i % 2 else 990.0)
    db.commit()
    assert scorecard(db, "NIFTY 50").sample_warning is True
    _call(db, 99, P_UP, 1010.0)
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 100 and sc.sample_warning is False
    assert sc.accuracy == pytest.approx(0.5)
    assert len(sc.recent) == 20


def test_champion_box_reports_the_champion_only(db: Session) -> None:
    def mv(version: str, champion: bool) -> ModelVersion:
        return ModelVersion(
            name=MODEL_NAME,
            target="NIFTY 50",
            version=version,
            kind="lightgbm",
            params={},
            feature_names=["ret_1d"],
            metrics={"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03, "folds": []},
            artifact="{}",
            trained_through=date(2026, 10, 30),
            train_rows=1200,
            is_champion=champion,
            promoted_at=NOW if champion else None,
            promotion_note="first model" if champion else None,
        )

    db.add_all([mv("lgbm-old", False), mv("lgbm-new", True)])
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.champion is not None
    assert sc.champion["version"] == "lgbm-new"
    assert sc.champion["trained_through"] == "2026-10-30"
    assert sc.champion["oos"] == {"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03}
    assert sc.champion["created_at"] is not None
    assert scorecard(db, "NIFTY BANK").champion is None


def test_gap_only_grades_count_toward_gap_accuracy_not_n_graded(db: Session) -> None:
    p = _call(db, 0, P_UP, None)
    grade_gap(db, p, 1004.0, now=NOW, source="test")
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 0 and sc.accuracy is None
    assert sc.gap_n == 1 and sc.gap_accuracy == 1.0


def test_verdict_text_names_the_baseline_and_the_sample_size() -> None:
    v = verdict_text(37, 0.54, "always_up", 0.49, True)
    assert v.startswith("After 37 graded calls the model is right 54% of the time")
    assert "best naive baseline (always UP) scores 49%" in v
    assert "ahead of the baselines" in v
    assert v.endswith("Fewer than 100 graded calls: treat as unproven.")

    v = verdict_text(150, 0.44, "same_as_yesterday", 0.51, False)
    assert "NOT beating the naive baselines" in v and "Fewer than" not in v

    v = verdict_text(150, 0.50, "random", 0.50, False)
    assert "level with the naive baselines" in v

    v = verdict_text(1, 1.0, None, None, True, min_sample_size=20)
    assert "After 1 graded call " in v and "no baseline has a call" in v
    assert "Fewer than 20 graded calls" in v

    assert "No graded market calls yet" in verdict_text(0, None, None, None, True)

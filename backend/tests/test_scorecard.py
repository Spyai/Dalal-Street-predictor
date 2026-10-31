"""The market-call scorecard: aggregates, paired baselines, calibration, verdict, champion box."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import BASELINE_NAMES, CONFIDENCE_LEVELS, MODEL_NAME
from app.models import ModelVersion, Prediction
from app.services.grading import grade_close, grade_gap
from app.services.ledger import record_market_call
from app.services.scorecard import (
    MIN_PAIRED_N,
    Scorecard,
    best_baseline,
    champion_summary,
    comparison_state,
    oos_verdict_text,
    scorecard,
    verdict_text,
)
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
OOS_METRICS: dict[str, Any] = {
    "n_oos": 400,
    "accuracy": 0.47,
    "brier": 0.701,
    "log_loss": 1.09,
    "class_frequencies": {"DOWN": 0.31, "FLAT": 0.30, "UP": 0.39},
    "baselines": {
        "always_up": {"n": 400, "accuracy": 0.39, "brier": 1.22},
        "same_as_yesterday": {"n": 400, "accuracy": 0.35, "brier": 1.30},
        "overnight_gap": {"n": 310, "accuracy": 0.41, "brier": 1.18},  # not every session
        "random": {"n": 400, "accuracy": 0.34, "brier": 0.668},  # the climatology forecast
    },
    "range_coverage": 0.78,
    "folds": [],
}


@contextmanager
def captured_sql(db: Session) -> Iterator[list[str]]:
    """Every SQL statement the session's engine runs inside the block (any session)."""
    statements: list[str] = []

    def record(
        conn: object, cursor: object, statement: str, params: object, ctx: object, many: bool
    ) -> None:
        statements.append(statement)

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


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
    baselines: dict[str, str | None] | None = None,
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
            baselines=dict(BASELINES if baselines is None else baselines),
            corrects=corrects,
            snapshot={"ret_1d": 0.1},
        ),
    )
    if close is not None:
        grade_close(db, p, actual_open, close, now=NOW + timedelta(days=i), source="test")
    return p


def _champion(target: str, version: str, metrics: dict[str, Any], champion: bool) -> ModelVersion:
    return ModelVersion(
        name=MODEL_NAME,
        target=target,
        version=version,
        kind="lightgbm",
        params={},
        feature_names=["ret_1d"],
        metrics=metrics,
        artifact="{" + '"classifier": "' + "x" * 2000 + '"}',
        trained_through=date(2026, 10, 30),
        train_rows=1200,
        is_champion=champion,
        promoted_at=NOW if champion else None,
        promotion_note="first model" if champion else None,
    )


def test_empty_scorecard_is_honest_about_having_nothing(db: Session) -> None:
    sc = scorecard(db, "NIFTY 50")
    assert isinstance(sc, Scorecard)
    assert sc.target == "NIFTY 50" and sc.n_graded == 0
    assert sc.accuracy is None and sc.brier is None and sc.log_loss is None
    assert sc.gap_n == 0 and sc.gap_accuracy is None
    assert sc.by_confidence == [] and sc.calibration == [] and sc.recent == []
    assert [b["name"] for b in sc.baselines] == list(BASELINE_NAMES)
    assert all(b["n"] == 0 and b["accuracy"] is None for b in sc.baselines)
    assert all(b["model_accuracy"] is None for b in sc.baselines)
    assert sc.best_baseline is None and sc.comparison == "no_calls"
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
        "model_accuracy": pytest.approx(4 / 5),
    }
    assert base["overnight_gap"]["n"] == 0 and base["overnight_gap"]["accuracy"] is None
    assert base["overnight_gap"]["model_accuracy"] is None
    assert base["random"]["n"] == 5 and base["random"]["accuracy"] == pytest.approx(1 / 5)
    assert sc.best_baseline is not None and sc.best_baseline["name"] == "always_up"

    assert sc.sample_warning is True
    assert sc.comparison == "too_early"  # 5 paired calls are far below MIN_PAIRED_N
    assert "After 5 graded calls" in sc.verdict and "80%" in sc.verdict
    assert "toughest naive baseline (always UP) scores 60% on the same 5 calls" in sc.verdict
    assert "Too early to tell" in sc.verdict and f"{MIN_PAIRED_N} paired calls" in sc.verdict
    assert "ahead" not in sc.verdict
    assert f"Fewer than {get_settings().min_sample_size} graded calls" in sc.verdict

    assert len(sc.recent) == 6  # the ungraded call is listed too, newest first
    assert sc.recent[0]["session_date"] == (SESSION + timedelta(days=5)).isoformat()
    assert sc.recent[0]["actual_class"] is None and sc.recent[0]["correct"] is None
    assert sc.recent[1]["actual_class"] == "DOWN" and sc.recent[1]["correct"] is True
    assert sc.recent[-1]["probs"] == P_UP and sc.recent[-1]["gap_correct"] is True
    assert all(r["target"] == "NIFTY 50" for r in sc.recent)
    assert sc.as_dict()["n_graded"] == 5


def test_baseline_comparison_is_paired_on_the_sessions_the_baseline_called(db: Session) -> None:
    """A baseline with calls on a subset is compared with the model on that subset only."""
    gap_none = dict(BASELINES)
    gap_down = {**BASELINES, "overnight_gap": "DOWN"}
    _call(db, 0, P_UP, 1010.0, baselines=gap_none)  # UP: model right, no gap call
    _call(db, 1, P_UP, 1010.0, baselines=gap_none)  # UP: model right, no gap call
    _call(db, 2, P_UP, 990.0, baselines=gap_down)  # DOWN: model wrong, gap call right
    _call(db, 3, P_UP, 990.0, baselines=gap_down)  # DOWN: model wrong, gap call right
    db.commit()

    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 4 and sc.accuracy == pytest.approx(0.5)
    base = {b["name"]: b for b in sc.baselines}
    # On the two sessions the gap rule called, it was 2/2 and the model 0/2: a paired n of 2.
    assert base["overnight_gap"] == {
        "name": "overnight_gap",
        "label": "overnight gap direction",
        "n": 2,
        "accuracy": 1.0,
        "model_accuracy": 0.0,
    }
    # Baselines that called every session are paired with the model's full record.
    assert base["always_up"]["n"] == 4 and base["always_up"]["model_accuracy"] == 0.5
    assert base["always_up"]["accuracy"] == 0.5
    # The toughest comparison is the gap rule (model trails it by 100 points), not the
    # baseline with the highest raw accuracy on a different set of sessions.
    assert sc.best_baseline is not None and sc.best_baseline["name"] == "overnight_gap"
    assert sc.best_baseline["n"] == 2 and sc.best_baseline["model_accuracy"] == 0.0
    assert (
        "On the 2 of them where the toughest naive baseline (overnight gap direction) had a "
        "call, the model scores 0% and the baseline 100%." in sc.verdict
    )
    assert "the model scores 50%" not in sc.verdict  # never the unpaired number
    assert sc.comparison == "too_early"


def test_verdict_claims_an_edge_only_with_enough_paired_calls(db: Session) -> None:
    # Model right (DOWN) and always-UP wrong on every session; no random/gap calls.
    base = {"always_up": "UP", "same_as_yesterday": "UP", "overnight_gap": None, "random": None}
    for i in range(MIN_PAIRED_N - 1):
        _call(db, i, P_DOWN, 990.0, baselines=base)
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == MIN_PAIRED_N - 1 and sc.accuracy == 1.0
    assert sc.best_baseline is not None and sc.best_baseline["n"] == MIN_PAIRED_N - 1
    assert sc.comparison == "too_early"
    assert "Too early to tell whether the model beats the baselines" in sc.verdict
    assert "ahead" not in sc.verdict and "NOT beating" not in sc.verdict
    assert "Fewer than 100 graded calls: treat as unproven." in sc.verdict  # still shown

    _call(db, MIN_PAIRED_N - 1, P_DOWN, 990.0, baselines=base)
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == MIN_PAIRED_N and sc.comparison == "ahead"
    assert "The model is ahead of the baselines." in sc.verdict
    assert "Too early" not in sc.verdict
    assert sc.sample_warning is True and "Fewer than 100 graded calls" in sc.verdict


def test_best_baseline_prefers_a_meaningful_paired_sample() -> None:
    def row(name: str, n: int, acc: float, model: float) -> dict[str, Any]:
        return {"name": name, "label": name, "n": n, "accuracy": acc, "model_accuracy": model}

    rows = [
        row("always_up", 40, 0.50, 0.55),  # enough calls, model +5
        row("same_as_yesterday", 40, 0.40, 0.55),  # enough calls, model +15
        row("overnight_gap", 3, 0.67, 0.33),  # tiny sample, model -34: not yet meaningful
        row("random", 0, None, None),  # type: ignore[arg-type]
    ]
    best = best_baseline(rows)
    assert best is not None and best["name"] == "always_up"
    assert comparison_state(best, 40) == "ahead"
    # With no baseline at MIN_PAIRED_N the toughest of the small samples is named ...
    small = [row("always_up", 5, 0.6, 0.6), row("random", 5, 0.2, 0.6)]
    best = best_baseline(small)
    assert best is not None and best["name"] == "always_up"
    # ... and the state is "too early", never ahead/behind.
    assert comparison_state(best, 5) == "too_early"
    assert comparison_state(None, 5) == "no_baseline"
    assert comparison_state(None, 0) == "no_calls"
    assert best_baseline([row("random", 0, None, None)]) is None  # type: ignore[arg-type]
    # Level within the tolerance, behind beyond it.
    assert comparison_state(row("always_up", 50, 0.50, 0.503), 50) == "level"
    assert comparison_state(row("always_up", 50, 0.50, 0.46), 50) == "behind"


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
    # 100 paired calls: the verdict takes a position (level: 50% vs always UP 50%).
    assert sc.comparison == "level" and "level with the naive baselines" in sc.verdict


def test_champion_box_reports_the_champion_only(db: Session) -> None:
    bare = {"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03, "folds": []}
    db.add_all([_champion("NIFTY 50", "lgbm-old", bare, False)])
    db.add_all([_champion("NIFTY 50", "lgbm-new", bare, True)])
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.champion is not None
    assert sc.champion["version"] == "lgbm-new"
    assert sc.champion["trained_through"] == "2026-10-30"
    assert sc.champion["oos"] == {"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03}
    assert sc.champion["created_at"] is not None
    # Stored without walk-forward baselines: nothing is invented to fill the box.
    assert sc.champion["baselines"] == [] and sc.champion["class_frequencies"] is None
    assert sc.champion["range_coverage"] is None and sc.champion["climatology_brier"] is None
    assert sc.champion["oos_verdict"] == (
        "Out of sample the model's Brier score was 0.610; no class-frequency Brier was "
        "stored with it, so there is nothing to compare that with."
    )
    assert scorecard(db, "NIFTY BANK").champion is None


def test_champion_box_shows_the_oos_baselines_beside_the_model(db: Session) -> None:
    db.add(_champion("NIFTY 50", "lgbm-full", OOS_METRICS, True))
    db.commit()
    champ = scorecard(db, "NIFTY 50").champion
    assert champ is not None
    assert [b["name"] for b in champ["baselines"]] == list(BASELINE_NAMES)
    assert champ["baselines"][0] == {
        "name": "always_up",
        "label": "always UP",
        "n": 400,
        "accuracy": 0.39,
        "brier": 1.22,
    }
    assert champ["baselines"][2]["n"] == 310  # the gap rule had no call on 90 sessions
    assert champ["class_frequencies"] == {"UP": 0.39, "FLAT": 0.30, "DOWN": 0.31}
    assert champ["range_coverage"] == 0.78
    assert champ["climatology_brier"] == 0.668
    # Plain words: worse than climatology on Brier; accuracy only against baselines that
    # called every OOS session (the gap rule's 41% on 310 sessions is not comparable).
    assert champ["oos_verdict"] == (
        "Out of sample the model's probabilities were worse than class frequencies "
        "(Brier 0.701 vs 0.668). Accuracy 47% vs the best naive baseline (always UP) 39% "
        "on the same 400 sessions."
    )
    assert champion_summary(db, "NIFTY 50") == champ


def test_oos_verdict_wording() -> None:
    bases = [
        {"name": "random", "label": "random", "n": 400, "accuracy": 0.34, "brier": 0.668},
        {"name": "always_up", "label": "always UP", "n": 400, "accuracy": 0.50, "brier": 1.0},
    ]
    better = oos_verdict_text({"brier": 0.60, "accuracy": 0.47, "n_oos": 400}, bases, 0.005)
    assert better.startswith("Out of sample the model's probabilities were better than class ")
    assert (
        "(Brier 0.600 vs 0.668)" in better and "Accuracy 47% vs the best naive baseline" in better
    )
    same = oos_verdict_text({"brier": 0.666, "accuracy": 0.47, "n_oos": 400}, bases, 0.005)
    assert "no better than class frequencies (Brier 0.666 vs 0.668)" in same
    assert (
        oos_verdict_text({}, bases, 0.005)
        == "No out-of-sample metrics were stored with this model."
    )
    # Brier only: no accuracy clause when there is no accuracy or no fully paired baseline.
    brier_only = oos_verdict_text({"brier": 0.7}, bases, 0.005)
    assert brier_only.endswith("(Brier 0.700 vs 0.668).")


def test_gap_only_grades_count_toward_gap_accuracy_not_n_graded(db: Session) -> None:
    p = _call(db, 0, P_UP, None)
    grade_gap(db, p, 1004.0, now=NOW, source="test")
    db.commit()
    sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 0 and sc.accuracy is None
    assert sc.gap_n == 1 and sc.gap_accuracy == 1.0


def test_verdict_text_names_the_baseline_and_the_sample_size() -> None:
    v = verdict_text(37, 0.54, "always_up", 0.49, True)
    assert v.startswith("After 37 graded calls the model is right 54% of the time.")
    assert "toughest naive baseline (always UP) scores 49% on the same 37 calls" in v
    assert "The model is ahead of the baselines." in v
    assert v.endswith("Fewer than 100 graded calls: treat as unproven.")

    v = verdict_text(150, 0.44, "same_as_yesterday", 0.51, False)
    assert "NOT beating the naive baselines" in v and "Fewer than" not in v

    v = verdict_text(150, 0.50, "random", 0.50, False)
    assert "level with the naive baselines" in v

    # Paired numbers: the baseline's subset and the model's score on that same subset.
    v = verdict_text(150, 0.52, "overnight_gap", 0.55, False, paired_n=80, paired_model_acc=0.50)
    assert "After 150 graded calls the model is right 52% of the time." in v
    assert (
        "On the 80 of them where the toughest naive baseline (overnight gap direction) had a "
        "call, the model scores 50% and the baseline 55%." in v
    )
    assert "NOT beating the naive baselines" in v

    # Below MIN_PAIRED_N no edge is claimed either way.
    v = verdict_text(150, 0.52, "overnight_gap", 0.30, False, paired_n=29, paired_model_acc=0.9)
    assert "Too early to tell whether the model beats the baselines: that needs 30 paired" in v
    assert "ahead" not in v
    v = verdict_text(150, 0.52, "overnight_gap", 0.30, False, paired_n=30, paired_model_acc=0.9)
    assert "The model is ahead of the baselines." in v

    v = verdict_text(1, 1.0, None, None, True, min_sample_size=20)
    assert "After 1 graded call " in v and "No baseline has a call" in v
    assert "Fewer than 20 graded calls" in v

    assert "No graded market calls yet" in verdict_text(0, None, None, None, True)


def test_scorecard_never_loads_the_feature_snapshot_drivers_or_artifact(db: Session) -> None:
    _call(db, 0, P_UP, 1010.0)
    _call(db, 1, P_UP, None)
    db.add(_champion("NIFTY 50", "lgbm-full", OOS_METRICS, True))
    db.commit()
    db.expunge_all()  # force the aggregate to read the database, not the identity map
    with captured_sql(db) as sql:
        sc = scorecard(db, "NIFTY 50")
    assert sc.n_graded == 1 and sc.champion is not None and len(sc.recent) == 2
    selects = [s for s in sql if s.lstrip().upper().startswith("SELECT")]
    assert selects, "the scorecard reads the ledger"
    for stmt in selects:
        assert "feature_snapshot" not in stmt, stmt
        assert "drivers" not in stmt, stmt
        assert "artifact" not in stmt, stmt

"""The ideas scorecard: aggregates per setup and horizon, calibration of p_target, the
sample-size warning, the verdict, the champion box, the per-setup track record."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml.core import IDEA_STATUSES
from app.ml.ideas_models import model_name, model_target
from app.models import ModelVersion
from app.services import ideas_scorecard as isc
from app.services.idea_grading import grade_idea, open_check_fill, void_idea
from app.services.ledger import record_idea
from tests.test_idea_grading import bars
from tests.test_idea_ledger import AS_OF, SESSION, make_idea, make_instrument
from tests.test_scorecard import captured_sql

NOW = datetime(2026, 11, 2, 10, 16, tzinfo=UTC)
OPEN_NOW = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)
OOS_METRICS: dict[str, Any] = {
    "n_oos": 3200,
    "accuracy": 0.61,
    "brier": 0.231,
    "log_loss": 0.66,
    "hit_rate": 0.58,
    "n_selected": 410,
    "hit_rate_ranked": 0.6,
    "n_ranked": 150,
    "expectancy_after_costs_pct": 0.82,
    "expectancy_ranked_pct": 1.01,
    "expectancy_all_candidates_pct": -0.12,
    "cost_round_trip_pct": 0.494,
    "baselines": {
        "all_candidates": {"n": 3200, "hit_rate": 0.41, "brier": 0.242},
        "random_universe": {"n": 3200, "hit_rate": 0.37},
    },
    "per_strategy": {"breakout_20d": {"n": 900, "hit_rate": 0.44}},
    "class_frequencies": {"target": 0.41, "stop": 0.47, "expired": 0.12},
    "calibration": [
        {"bin_lo": 0.5, "bin_hi": 0.6, "n": 300, "predicted_mean": 0.55, "observed": 0.5}
    ],
    "notes": ["no atr14_pct column: EV bar and expectancy not computed (NaN)"],
    "folds": [],
}


def _idea(db: Session, symbol: str, horizon: str, i: int, **kw: Any):  # type: ignore[no-untyped-def]
    inst = make_instrument(db, symbol)
    return record_idea(
        db,
        make_idea(
            inst.id,
            symbol,
            horizon,
            session_date=SESSION + timedelta(days=i),
            as_of=AS_OF + timedelta(days=i),
            **kw,
        ),
    )


def settle(db: Session, symbol: str, horizon: str, result: str, i: int = 0, **kw: Any):  # type: ignore[no-untyped-def]
    """Record an idea and grade it to `result`: target, stop, expired_win or expired_loss."""
    idea = _idea(db, symbol, horizon, i, **kw)
    n = idea.horizon_days
    if result == "target":
        rows = [(i, 1000.0, 1045.0 if horizon == "swing" else 1085.0, 998.0, 1030.0)]
    elif result == "stop":
        rows = [(i, 1000.0, 1005.0, 975.0 if horizon == "swing" else 955.0, 990.0)]
    elif result == "expired_win":  # drifts up 0.5 / session, never reaches the target
        rows = [
            (i + d, 1000 + 0.5 * d, 1006 + 0.5 * d, 996 + 0.5 * d, 1002 + 0.5 * d) for d in range(n)
        ]
    elif result == "expired_loss":  # drifts down 0.3 / session, never reaches the stop
        rows = [
            (i + d, 1000 - 0.3 * d, 1004 - 0.3 * d, 994 - 0.3 * d, 998 - 0.3 * d) for d in range(n)
        ]
    else:
        raise AssertionError(result)
    outcome = grade_idea(db, idea, bars(*rows), now=NOW + timedelta(days=i), source="test")
    assert outcome.status == result.split("_")[0], (result, outcome.status, outcome.notes)
    return idea, outcome


def _champion(horizon: str, version: str, metrics: dict[str, Any], champion: bool) -> ModelVersion:
    return ModelVersion(
        name=model_name(horizon),
        target=model_target(),
        version=version,
        kind="lightgbm",
        params={},
        feature_names=["ret_20d"],
        metrics=metrics,
        artifact="{" + '"booster": "' + "x" * 2000 + '"}',
        trained_through=date(2026, 10, 30),
        train_rows=5000,
        is_champion=champion,
        promoted_at=NOW if champion else None,
        promotion_note="first ideas model" if champion else None,
    )


def test_empty_scorecard_is_honest_about_having_nothing(db: Session) -> None:
    sc = isc.ideas_scorecard(db)
    assert isinstance(sc, isc.IdeasScorecard)
    assert sc.horizon is None and sc.n_ideas == 0 and sc.n_graded == 0
    assert sc.by_status == dict.fromkeys(IDEA_STATUSES, 0)
    for name in (
        "hit_rate",
        "win_rate",
        "avg_win_pct",
        "avg_loss_pct",
        "expectancy_pct",
        "profit_factor",
        "avg_r_multiple",
        "avg_days_held",
    ):
        assert getattr(sc, name) is None, name
    assert sc.by_setup == [] and sc.calibration == [] and sc.recent == []
    assert [r["horizon"] for r in sc.by_horizon] == ["swing", "positional"]
    assert all(r["n"] == 0 and r["hit_rate"] is None for r in sc.by_horizon)
    assert sc.champions == {"swing": None, "positional": None}
    assert sc.sample_warning is True
    assert sc.verdict.startswith("No graded ideas yet: the ideas engine is unproven.")
    assert f"until it has {get_settings().min_sample_size} graded ideas" in sc.verdict
    assert sc.as_dict()["n_graded"] == 0
    with pytest.raises(ValueError, match="unknown horizon"):
        isc.ideas_scorecard(db, "intraday")


def test_aggregates_over_graded_ideas(db: Session) -> None:
    _, t1 = settle(db, "RELIANCE", "swing", "target", p_target=0.65)
    _, s1 = settle(db, "TCS", "swing", "stop", p_target=0.6)
    _, w1 = settle(db, "INFY", "swing", "expired_win", p_target=0.58, setup="pullback")
    _, l1 = settle(db, "HDFCBANK", "swing", "expired_loss", p_target=0.56, setup="pullback")
    _, t2 = settle(db, "RELIANCE", "positional", "target", p_target=0.7, setup="momentum")
    _idea(db, "WIPRO", "swing", 5)  # pending
    open_check_fill(db, _idea(db, "ITC", "swing", 6), 1000.0, now=OPEN_NOW, source="t")  # open
    open_check_fill(db, _idea(db, "SBIN", "swing", 7), 1020.0, now=OPEN_NOW, source="t")  # gapped
    void_idea(db, _idea(db, "LT", "positional", 8), now=NOW, reason="holiday")
    db.commit()

    sc = isc.ideas_scorecard(db)
    assert sc.n_ideas == 9 and sc.n_graded == 5
    assert sc.by_status == {
        "pending": 1,
        "open": 1,
        "target": 2,
        "stop": 1,
        "expired": 2,
        "invalidated": 1,
        "void": 1,
    }
    nets = [o.net_return_pct for o in (t1, s1, w1, l1, t2)]
    assert all(n is not None for n in nets)
    wins = [n for n in nets if n is not None and n > 0]
    losses = [n for n in nets if n is not None and n < 0]
    assert len(wins) == 3 and len(losses) == 2, nets  # expiries: one won, one lost after costs
    assert sc.hit_rate == pytest.approx(2 / 5)  # target before stop: the two target hits
    assert sc.win_rate == pytest.approx(3 / 5)
    assert sc.avg_win_pct == pytest.approx(sum(wins) / 3)
    assert sc.avg_loss_pct == pytest.approx(sum(losses) / 2)
    assert sc.expectancy_pct == pytest.approx(sum(wins + losses) / 5)
    assert sc.profit_factor == pytest.approx(sum(wins) / -sum(losses))
    assert sc.avg_r_multiple == pytest.approx(
        sum(float(o.r_multiple) for o in (t1, s1, w1, l1, t2) if o.r_multiple is not None) / 5
    )
    assert sc.avg_days_held == pytest.approx((1 + 1 + 10 + 10 + 1) / 5)

    by_setup = {r["setup"]: r for r in sc.by_setup}
    assert list(by_setup) == ["breakout", "momentum", "pullback"]
    assert by_setup["breakout"]["n"] == 2 and by_setup["breakout"]["hit_rate"] == 0.5
    assert by_setup["momentum"]["n"] == 1 and by_setup["momentum"]["hit_rate"] == 1.0
    assert by_setup["pullback"]["n"] == 2 and by_setup["pullback"]["hit_rate"] == 0.0
    assert (
        by_setup["pullback"]["win_rate"] == 0.5
        and by_setup["pullback"]["profit_factor"] is not None
    )
    by_h = {r["horizon"]: r for r in sc.by_horizon}
    assert list(by_h) == ["swing", "positional"]
    assert by_h["swing"]["n"] == 4 and by_h["swing"]["hit_rate"] == 0.25
    assert by_h["positional"]["n"] == 1 and by_h["positional"]["hit_rate"] == 1.0

    assert sum(b["n"] for b in sc.calibration) == 5
    cal = {(b["bin_lo"], b["bin_hi"]): b for b in sc.calibration}
    assert cal[(0.5, 0.6)]["n"] == 2 and cal[(0.5, 0.6)]["observed"] == 0.0
    assert cal[(0.6, 0.7)]["n"] == 2 and cal[(0.6, 0.7)]["observed"] == 0.5
    assert cal[(0.6, 0.7)]["predicted_mean"] == pytest.approx(0.625)
    assert cal[(0.7, 0.8)]["n"] == 1 and cal[(0.7, 0.8)]["observed"] == 1.0

    assert sc.sample_warning is True
    assert sc.verdict.startswith(
        "After 5 graded ideas the target came before the stop 40% of the time."
    )
    assert f"{sc.expectancy_pct:+.2f}% after costs" in sc.verdict
    assert f"profit factor {sc.profit_factor:.2f}" in sc.verdict
    assert "making money after costs so far" in sc.verdict
    assert f"Fewer than {get_settings().min_sample_size} graded ideas" in sc.verdict

    assert len(sc.recent) == 9 and sc.recent[0]["symbol"] == "LT"  # newest as_of first
    assert sc.recent[0]["status"] == "void" and sc.recent[0]["error_category"] == "no_session"
    assert sc.recent[1]["status"] == "invalidated" and sc.recent[2]["status"] == "open"
    assert sc.recent[3]["status"] == "pending" and sc.recent[3]["entry_price"] is None
    assert sc.recent[-1]["status"] in {"target", "stop", "expired"}
    assert all(r["surveillance_unknown"] is True for r in sc.recent)
    assert sc.champions == {"swing": None, "positional": None}

    swing = isc.ideas_scorecard(db, "swing")
    assert swing.horizon == "swing" and swing.n_graded == 4 and swing.n_ideas == 7
    assert swing.hit_rate == 0.25 and [r["horizon"] for r in swing.by_horizon] == ["swing"]
    assert list(swing.champions) == ["swing"] and "(swing)" in swing.verdict
    assert all(r["horizon"] == "swing" for r in swing.recent)
    positional = isc.ideas_scorecard(db, "positional")
    assert positional.n_graded == 1 and positional.hit_rate == 1.0
    assert positional.profit_factor is None  # no losing idea yet
    assert "no losing idea yet, so no profit factor" in positional.verdict


def test_superseded_ideas_count_once(db: Session) -> None:
    original, _ = settle(db, "RELIANCE", "swing", "target")
    fix = _idea(db, "RELIANCE", "swing", 0, corrects=original.id, note="levels fixed")
    grade_idea(db, fix, bars((0, 1000.0, 1005.0, 975.0, 990.0)), now=NOW, source="t")
    db.commit()
    sc = isc.ideas_scorecard(db)
    assert sc.n_ideas == 1 and sc.n_graded == 1
    assert sc.hit_rate == 0.0  # the correction (a stop) is what counts
    assert [r["id"] for r in sc.recent] == [fix.id]


def test_sample_warning_flips_at_the_configured_size(db: Session) -> None:
    settle(db, "RELIANCE", "swing", "target")
    settle(db, "TCS", "swing", "stop")
    db.commit()
    small = Settings(min_sample_size=2)
    sc = isc.ideas_scorecard(db, settings=small)
    assert sc.n_graded == 2 and sc.sample_warning is False
    assert "Fewer than" not in sc.verdict and "unproven" not in sc.verdict
    sc3 = isc.ideas_scorecard(db, settings=Settings(min_sample_size=3))
    assert sc3.sample_warning is True and "Fewer than 3 graded ideas" in sc3.verdict
    assert isc.ideas_scorecard(db).sample_warning is True  # the default 100


def test_verdict_text_wording() -> None:
    v = isc.verdict_text(37, 0.54, 0.83, 1.9, True)
    assert v.startswith("After 37 graded ideas the target came before the stop 54% of the time.")
    assert "The average idea made +0.83% after costs (profit factor 1.90)." in v
    assert "The ideas are making money after costs so far." in v
    assert v.endswith("Fewer than 100 graded ideas: treat as unproven.")

    v = isc.verdict_text(150, 0.44, -0.31, 0.8, False, horizon="swing")
    assert "(swing)" in v and "NOT making money" in v and "Fewer than" not in v
    v = isc.verdict_text(150, 0.5, 0.0, 1.0, False)
    assert "breaking even" in v
    v = isc.verdict_text(1, 1.0, 2.5, None, True, min_sample_size=20)
    assert "After 1 graded idea " in v and "no losing idea yet" in v
    assert "Fewer than 20 graded ideas" in v
    assert isc.verdict_text(0, None, None, None, True).startswith("No graded ideas yet")
    assert "(positional)" in isc.verdict_text(0, None, None, None, True, horizon="positional")


def test_champion_box_reports_the_champion_per_horizon(db: Session) -> None:
    db.add(_champion("swing", "lgbm-old", {"n_oos": 100}, False))
    db.add(_champion("swing", "lgbm-new", OOS_METRICS, True))
    db.commit()
    sc = isc.ideas_scorecard(db)
    champ = sc.champions["swing"]
    assert champ is not None and sc.champions["positional"] is None
    assert champ["version"] == "lgbm-new" and champ["name"] == "ideas_swing"
    assert champ["horizon"] == "swing" and champ["trained_through"] == "2026-10-30"
    assert champ["train_rows"] == 5000 and champ["kind"] == "lightgbm"
    assert champ["oos"]["hit_rate"] == 0.58 and champ["oos"]["n_selected"] == 410
    assert champ["oos"]["expectancy_after_costs_pct"] == 0.82 and champ["oos"]["brier"] == 0.231
    assert champ["baselines"] == [
        {
            "name": "all_candidates",
            "label": "every candidate that session",
            "n": 3200,
            "hit_rate": 0.41,
            "brier": 0.242,
        },
        {
            "name": "random_universe",
            "label": "random universe stocks, same barriers",
            "n": 3200,
            "hit_rate": 0.37,
            "brier": None,
        },
    ]
    assert champ["per_strategy"] == {"breakout_20d": {"n": 900, "hit_rate": 0.44}}
    assert champ["class_frequencies"] == {"target": 0.41, "stop": 0.47, "expired": 0.12}
    assert champ["calibration"][0]["n"] == 300 and len(champ["notes"]) == 1
    assert champ["oos_verdict"] == (
        "Out of sample 410 ideas cleared the bar and hit the target 58% of the time, "
        "against 41% for every candidate (n=3200). Expectancy after costs +0.82% per idea."
    )
    assert champ["promoted_at"] is not None and champ["created_at"] is not None
    assert isc.champion_summary(db, "swing") == champ

    bare = isc.oos_verdict_text({"n_oos": 100}, [])
    assert bare == "No out-of-sample hit rate was stored with this model."
    no_base = isc.oos_verdict_text({"hit_rate": 0.5, "n_selected": 20}, [])
    assert "no all-candidates baseline was stored" in no_base
    assert no_base.endswith("No expectancy after costs was stored.")


def test_idea_track_per_setup_and_horizon(db: Session) -> None:
    settle(db, "RELIANCE", "swing", "target")
    settle(db, "TCS", "swing", "stop")
    settle(db, "INFY", "swing", "expired_win", setup="pullback")
    db.commit()
    t = isc.idea_track(db, "swing", "breakout")
    assert isinstance(t, isc.IdeaTrack)
    assert t.horizon == "swing" and t.setup == "breakout"
    assert t.n_graded == 2 and t.hit_rate == 0.5 and t.win_rate == 0.5
    assert t.expectancy_pct is not None and t.sample_warning is True
    whole = isc.idea_track(db, "swing")
    assert whole.setup is None and whole.n_graded == 3 and whole.hit_rate == pytest.approx(1 / 3)
    none = isc.idea_track(db, "positional", "breakout", settings=Settings(min_sample_size=0))
    assert none.n_graded == 0 and none.hit_rate is None and none.sample_warning is False
    assert whole.as_dict()["n_graded"] == 3


def test_calibration_bins_binary() -> None:
    rows = isc.calibration_bins([0.55, 0.58, 0.65, 1.0], [False, True, True, True], bins=10)
    assert [(r["bin_lo"], r["bin_hi"], r["n"]) for r in rows] == [
        (0.5, 0.6, 2),
        (0.6, 0.7, 1),
        (0.9, 1.0, 1),
    ]
    assert rows[0]["observed"] == 0.5 and rows[0]["predicted_mean"] == pytest.approx(0.565)
    assert isc.calibration_bins([], []) == []


def test_scorecard_never_loads_the_feature_snapshot_drivers_or_artifact(db: Session) -> None:
    settle(db, "RELIANCE", "swing", "target")
    _idea(db, "TCS", "swing", 1)
    db.add(_champion("swing", "lgbm-full", OOS_METRICS, True))
    db.commit()
    db.expunge_all()
    with captured_sql(db) as sql:
        sc = isc.ideas_scorecard(db)
    assert sc.n_graded == 1 and sc.champions["swing"] is not None and len(sc.recent) == 2
    selects = [s for s in sql if s.lstrip().upper().startswith("SELECT")]
    assert selects, "the scorecard reads the ledger"
    for stmt in selects:
        assert "feature_snapshot" not in stmt, stmt
        assert "drivers" not in stmt, stmt
        assert "artifact" not in stmt, stmt

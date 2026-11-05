"""Phase 3 review fixes.

Labels are the graded event (barriers anchored at close(D), a fill only inside the entry
zone, otherwise `invalidated`); the walk-forward purges a row whose label resolved inside
a test window; the grading walk counts sessions only and never re-anchors a filled idea on
a later bar; the paper book never borrows; the ranking report lists the nearest misses;
the metrics carry the decile lift and the "best ideas every session" replay.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ml import ideas_models as im
from app.ml.core import ENTRY_ZONE_ATR, HorizonSpec, TrainResult
from app.ml.ideas_features import LABEL_COLUMNS, triple_barrier_labels
from app.models import PaperTrade
from app.services import brief, paper
from app.services.idea_grading import grade_idea, open_check_fill, session_bars
from app.services.ledger import record_idea
from tests.test_idea_grading import bars, quiet
from tests.test_idea_ledger import SESSION, make_idea, make_instrument
from tests.test_ideas_features import _flat_bars
from tests.test_ideas_models import settings as model_settings
from tests.test_ideas_models import synthetic_candidates
from tests.test_paper import NOW, OPEN_NOW

# ------------------------------------------------------------------------ labels


def test_labels_are_the_graded_event_anchored_at_the_signal_close() -> None:
    h = HorizonSpec("t", 5, 2.0, 1.0)  # flat bars: ATR 2 -> zone +/-1, target 104, stop 98
    d = 15

    def run(**mods: dict[int, float]):  # type: ignore[no-untyped-def]
        b = _flat_bars(30)
        for col, changes in mods.items():
            for pos, v in changes.items():
                b.iloc[pos, b.columns.get_loc(col)] = v
        out = triple_barrier_labels(b, [b.index[d].date()], h)
        assert list(out.columns) == list(LABEL_COLUMNS)
        return out.iloc[0]

    assert ENTRY_ZONE_ATR == 0.5 and "invalidated" in LABEL_COLUMNS
    # Open inside the zone: filled at it; the barriers stay at close(D) x (1 +/- k ATR%),
    # exactly the levels the ledger records and the grading walks.
    row = run(open={16: 100.9}, high={18: 104.0})
    assert not row["invalidated"] and row["entry_price"] == pytest.approx(100.9)
    assert row["target_price"] == pytest.approx(104.0) and row["stop_price"] == pytest.approx(98.0)
    assert row["label"] == 1.0 and row["exit_price"] == pytest.approx(104.0)
    assert row["realised_return_pct"] == pytest.approx((104.0 / 100.9 - 1.0) * 100.0)
    # Open outside the zone (1.5 > 0.5 ATR away): the 09:20 rule invalidates, so no label,
    # no exit, no path read, and the row is reported as such.
    row = run(open={16: 101.5}, high={18: 104.0})
    assert row["invalidated"] and math.isnan(row["label"]) and not row["expired"]
    assert row["exit_session"] is None and math.isnan(row["exit_price"])
    assert row["entry_price"] == pytest.approx(101.5) and row["window_closed"]
    row = run(open={16: 98.9})
    assert row["invalidated"] and math.isnan(row["label"])
    # Just inside the edge is a fill.
    row = run(open={16: 100.99})
    assert not row["invalidated"] and not math.isnan(row["label"])
    # The ATR of D sets the zone too: a wide bar on D widens it.
    row = run(open={16: 101.5}, high={15: 110.0}, low={15: 90.0})
    assert not row["invalidated"]


# --------------------------------------------------------------- walk-forward purge


@pytest.fixture(scope="module")
def trained() -> TrainResult:
    return im.train_walk_forward(synthetic_candidates(), "swing", settings=model_settings())


def test_walk_forward_purges_rows_whose_label_resolved_inside_the_test_window() -> None:
    frame = synthetic_candidates()
    s = model_settings()
    sessions = sorted(set(frame["session_date"]))
    pos = {d: i for i, d in enumerate(sessions)}
    # The exit the label pipeline writes: at most `days` sessions after D -> nothing to purge.
    exits = [sessions[min(pos[d] + 10, len(sessions) - 1)] for d in frame["session_date"]]
    clean = im.train_walk_forward(frame.assign(exit_session=exits), "swing", settings=s)
    assert not any("purged beyond the embargo" in n for n in clean.metrics.notes)
    # Stretch the first row's exit to the last session (a stock that skipped sessions): it
    # leaks into every test window, so every fold purges it.
    late = frame.assign(exit_session=exits)
    late.loc[late.index[0], "exit_session"] = sessions[-1]
    result = im.train_walk_forward(late, "swing", settings=s)
    notes = [n for n in result.metrics.notes if "purged beyond the embargo" in n]
    assert len(notes) == 1 and int(notes[0].split()[0]) == len(result.metrics.folds)
    # Without an exit column there is nothing to purge by, and no note is invented.
    assert not any(
        "purged" in n for n in im.train_walk_forward(frame, "swing", settings=s).metrics.notes
    )


# ------------------------------------------------------------- lift and the replay


def test_lift_table_math() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    net = np.where(y == 1, 4.0, -2.0) - 0.5
    rows = im.lift_table(p, y, net, bins=5)
    assert [r["decile"] for r in rows] == [1, 2, 3, 4, 5] and all(r["n"] == 2 for r in rows)
    assert [r["hit_rate"] for r in rows] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert rows[0]["p_mean"] == pytest.approx(0.15)
    assert rows[-1]["expectancy_pct"] == pytest.approx(3.5)
    assert rows[2]["expectancy_pct"] == pytest.approx(0.5)
    assert im.lift_table(np.array([]), np.array([])) == []
    assert all(r["expectancy_pct"] is None for r in im.lift_table(p, y, bins=2))
    nan_net = np.full(10, np.nan)
    assert all(r["expectancy_pct"] is None for r in im.lift_table(p, y, nan_net, bins=2))


def test_lift_within_groups_measures_ranking_not_base_rate_drift() -> None:
    # Two sessions. Session A: base rate 100 %, session B: 0 %. Globally the high-p rows
    # are all B's (a drift of the base rate between periods), so a global split shows an
    # inverted lift; within each session the model ranks nothing, so the grouped table is flat.
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    global_rows = im.lift_table(p, y, bins=2)
    assert [r["hit_rate"] for r in global_rows] == [1.0, 0.0]
    grouped = im.lift_table(p, y, bins=2, groups=groups)
    assert [r["hit_rate"] for r in grouped] == [0.5, 0.5] and all(r["n"] == 4 for r in grouped)
    # A real within-session ranking shows up as lift in the grouped table.
    y2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert [r["hit_rate"] for r in im.lift_table(p, y2, bins=2, groups=groups)] == [0.0, 1.0]
    # Small groups: 3 rows into 10 deciles land in deciles 2, 5 and 9 (percentile mid-points).
    small = im.lift_table(
        np.array([0.3, 0.1, 0.2]), np.array([1, 0, 1]), groups=np.array([1, 1, 1])
    )
    assert [r["decile"] for r in small] == [2, 6, 9] and [r["hit_rate"] for r in small] == [
        0.0,
        1.0,
        1.0,
    ]
    with pytest.raises(ValueError, match="one key per row"):
        im.lift_table(p, y, groups=np.array([1, 2]))


def test_label_version_travels_with_the_model_and_forces_promotion(
    trained: TrainResult,
) -> None:
    from app.ml.ideas_features import LABEL_VERSION
    from app.models import ModelVersion

    payload = json.loads(trained.artifact)
    assert payload["label_version"] == LABEL_VERSION
    current = ModelVersion(
        name="ideas_swing", target="X", version="v1", kind="lgbm", artifact=trained.artifact
    )
    assert im.label_version_of(current) == LABEL_VERSION
    assert im.label_change_reason(current) is None and im.label_change_reason(None) is None
    legacy_payload = {k: v for k, v in payload.items() if k != "label_version"}
    legacy = ModelVersion(
        name="ideas_swing",
        target="X",
        version="v0",
        kind="lgbm",
        artifact=json.dumps(legacy_payload),
    )
    assert im.label_version_of(legacy) == im.LEGACY_LABEL_VERSION
    reason = im.label_change_reason(legacy)
    assert reason is not None and "not comparable" in reason and LABEL_VERSION in reason
    broken = ModelVersion(name="ideas_swing", target="X", version="v?", kind="lgbm", artifact="{")
    assert im.label_version_of(broken) == im.LEGACY_LABEL_VERSION


def test_metrics_carry_the_top_k_replay_and_the_lift(trained: TrainResult) -> None:
    m = trained.metrics
    assert isinstance(m, im.IdeaModelMetrics)
    assert m.n_top > 0 and 0.0 <= m.hit_rate_top <= 1.0 and m.n_top <= m.n_oos
    assert m.n_top <= 3 * m.n_sessions  # at most ideas_max_per_horizon a session
    assert len(m.lift) == 10 and sum(int(r["n"]) for r in m.lift) == m.n_oos
    assert [int(r["decile"]) for r in m.lift] == list(range(1, 11))  # type: ignore[arg-type]
    assert math.isfinite(m.expectancy_top_pct)
    d = m.as_dict()
    assert d["n_top"] == m.n_top and d["lift"] == m.lift
    assert d["hit_rate_top"] == pytest.approx(m.hit_rate_top)
    assert d["expectancy_top_pct"] == pytest.approx(m.expectancy_top_pct)


# -------------------------------------------------------------------- grading


def test_grading_counts_sessions_only_and_never_reanchors_a_filled_idea(db: Session) -> None:
    inst = make_instrument(db)
    idea = record_idea(db, make_idea(inst.id))  # swing, zone 990..1010, stop 980, target 1040
    fill = open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    assert fill.status == "open"
    holiday = SESSION + timedelta(days=3)
    frame = bars(*quiet(3), (3, 1000.0, 1041.0, 999.0, 1030.0))  # a print on a non-session
    walked = session_bars(frame, SESSION, is_session=lambda d: d != holiday)
    assert [b.trade_date for b in walked] == [SESSION + timedelta(days=i) for i in range(3)]
    # With the calendar the holiday print is not a session: the target it shows never counts.
    out = grade_idea(db, idea, frame, now=NOW, source="test", is_session=lambda d: d != holiday)
    assert out.status == "open" and "3 of 10" in (out.notes or "")
    # Without it, every stored bar is a session (the old behaviour).
    assert grade_idea(db, idea, frame, now=NOW, source="test").status == "target"

    inst2 = make_instrument(db, "TCS")
    idea2 = record_idea(db, make_idea(inst2.id, "TCS"))
    assert open_check_fill(db, idea2, 1000.0, now=OPEN_NOW, source="kite").status == "open"
    # The entry session's bar is missing: the window is not re-anchored on the next bar.
    with pytest.raises(ValueError, match="no bar for that session"):
        grade_idea(db, idea2, bars(*quiet(3, start_day=1)), now=NOW, source="test")


# ----------------------------------------------------------------- paper book cash


def test_paper_book_never_borrows(db: Session) -> None:
    inst = make_instrument(db)
    big = record_idea(db, make_idea(inst.id, qty=1_500))  # Rs 15 lakh at 1000 > 10 lakh capital
    fill = open_check_fill(db, big, 1000.0, now=OPEN_NOW, source="kite")
    t = paper.open_trade_for_idea(db, big, fill, now=NOW)
    assert t is not None and t.qty == 1_000
    assert "sized down from 1500 to 1000 shares" in (t.note or "")
    assert paper.cash_as_of(db, SESSION) == pytest.approx(0.0)
    assert paper.open_trade_for_idea(db, big, fill, now=NOW) is t  # idempotent
    # The next fill the same day finds no cash: no trade; the idea is still graded.
    inst2 = make_instrument(db, "TCS")
    second = record_idea(db, make_idea(inst2.id, "TCS"))
    fill2 = open_check_fill(db, second, 1000.0, now=OPEN_NOW, source="kite")
    assert paper.open_trade_for_idea(db, second, fill2, now=NOW) is None
    assert paper.sync_trade_with_outcome(db, second, fill2, now=NOW) is None
    assert db.scalar(select(func.count()).select_from(PaperTrade)) == 1
    # Cash comes back when a position closes.
    paper.close_trade(db, t, SESSION + timedelta(days=1), 1040.0, now=NOW)
    assert paper.cash_as_of(db, SESSION + timedelta(days=1)) > 1_000_000.0


# ------------------------------------------------------------ what the 09:20 note says


def test_fill_summary_reports_the_paper_position_actually_taken(db: Session) -> None:
    inst = make_instrument(db)
    idea = record_idea(db, make_idea(inst.id, qty=1_500))
    fill = open_check_fill(db, idea, 1000.0, now=OPEN_NOW, source="kite")
    trade = paper.open_trade_for_idea(db, idea, fill, now=NOW)
    assert trade is not None and trade.qty == 1_000
    msg = brief.fill_summary([(idea, fill, trade)], SESSION)
    assert "1000 shares paper (sized down from 1500: cash)" in msg and "1500 shares" not in msg
    unfunded = brief.fill_summary([(idea, fill, None)], SESSION)
    assert "no paper position: no cash left in the paper book" in unfunded
    assert "shares paper" not in unfunded
    # The two-tuple form (older callers) still states the idea's own size.
    assert "1500 shares paper" in brief.fill_summary([(idea, fill)], SESSION)

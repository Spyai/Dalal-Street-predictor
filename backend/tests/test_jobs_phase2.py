"""Phase 2 jobs end to end: 07:30 prediction, 09:20 gap grade, 15:45 close grade, Saturday
retrain, the Telegram texts, the CLI and the GitHub Actions schedule.

No network: the market-data chain and the NSE provider are faked and Telegram is captured.
The jobs run through `run_job` against the real database on a synthetic year of bars, with
the production feature pipeline and model code; a small walk-forward window (Settings
override) keeps each training run to a few seconds.
"""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app import cli
from app.config import IST, Settings
from app.jobs import runner, tasks
from app.jobs.definitions import get_spec
from app.jobs.runner import run_job
from app.ml import features, models
from app.ml import registry as model_registry
from app.ml.core import (
    BASELINE_NAMES,
    CLASSES,
    CONFIDENCE_LEVELS,
    Driver,
    MarketCallOutput,
    MarketCallRecordIn,
)
from app.models import DailyBar, JobRun, ModelVersion, Prediction, PredictionGrade
from app.providers.base import Bar, FlowRecord, NotSupported, SourceUnavailable
from app.services import brief, instruments, ledger
from app.services.grading import grade_close
from app.services.ledger import record_market_call
from app.services.market_data import upsert_daily_bars
from app.services.scorecard import scorecard
from tests.test_ledger import AS_OF, SESSION, make_record
from tests.test_market_data import FakeProvider

REPO = Path(__file__).resolve().parents[2]
TARGET = "NIFTY 50"
FIRST = date(2025, 10, 1)
LAST = date(2026, 10, 29)  # Thursday: the last bar before the Friday session
FRIDAY = date(2026, 10, 30)
FRIDAY_0730 = datetime(2026, 10, 30, 7, 30, tzinfo=IST)
FRIDAY_0920 = datetime(2026, 10, 30, 9, 20, tzinfo=IST)
FRIDAY_1545 = datetime(2026, 10, 30, 15, 45, tzinfo=IST)
SATURDAY_1000 = datetime(2026, 10, 31, 10, 0, tzinfo=IST)
# A one-year history is enough for three walk-forward folds with this window.
SETTINGS = Settings(prediction_targets=TARGET, wf_min_train_days=150, wf_step_days=60)
ALLOWLIST = (
    "^(migrate|sync-instruments|backfill|test-telegram|train|predict( --force)?|"
    "verify-ledger|run-job [a-z_]+( --force)?)$"
)


# ------------------------------------------------------------------ synthetic market


def _weekdays(first: date, last: date) -> list[date]:
    return [
        first + timedelta(i)
        for i in range((last - first).days + 1)
        if (first + timedelta(i)).weekday() < 5
    ]


def _bars(rng: np.random.Generator, days: list[date]) -> list[Bar]:
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(days))))
    out: list[Bar] = []
    for i, (d, c) in enumerate(zip(days, close, strict=True)):
        prev = close[i - 1] if i else c
        o = prev * (1 + rng.normal(0, 0.004))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
        out.append(Bar(d, float(o), float(hi), float(lo), float(c), float(rng.integers(1e5, 4e5))))
    return out


@pytest.fixture()
def world(db: Session) -> dict[str, Bar]:
    """Targets, India VIX and every global cue with bars on all weekdays through LAST.
    Returns the target's bars by date."""
    rng = np.random.default_rng(7)
    instruments.ensure_reference_instruments(db)
    db.flush()
    days = _weekdays(FIRST, LAST)
    target_bars: dict[date, Bar] = {}
    for ref in instruments.ALL_REF:
        inst = instruments.get_instrument(db, ref.symbol, ref.exchange)
        assert inst is not None
        bars = _bars(rng, days)
        upsert_daily_bars(db, inst.id, bars, "yfinance")
        if ref.symbol == TARGET:
            target_bars = {b.trade_date: b for b in bars}
    db.commit()
    return {d.isoformat(): b for d, b in target_bars.items()}


@pytest.fixture()
def champion(db: Session, world: dict[str, Bar]) -> ModelVersion:
    frame = features.build_frame(db, TARGET, settings=SETTINGS)
    result = models.train_walk_forward(frame, settings=SETTINGS, target=TARGET)
    mv = model_registry.save_model(db, TARGET, result, promote=True, note="test")
    db.commit()
    return mv


# ------------------------------------------------------------------ fakes


class ExchangeStub:
    """NSE stand-in: the FII/DII pull returns nothing (or fails when asked to)."""

    name = "nse"

    def __init__(self) -> None:
        self.fail = False
        self.calls = 0

    def fii_dii(self) -> list[FlowRecord]:
        self.calls += 1
        if self.fail:
            raise SourceUnavailable(self.name, "blocked")
        return []

    def holidays(self) -> list[Any]:
        raise NotSupported(self.name, "not in this test")

    def index_constituents(self, index: str) -> list[Any]:
        raise NotSupported(self.name, "not in this test")

    def bhavcopy(self, trade_date: date) -> list[Any]:
        raise NotSupported(self.name, "not in this test")


class BarProvider(FakeProvider):
    """Market-data stand-in whose only data is the bars it is given (by symbol)."""

    def __init__(self) -> None:
        super().__init__("yfinance", set())
        self.bars: dict[str, list[Bar]] = {}

    def daily_bars(self, symbol: str, exchange: str, start: date, end: date) -> list[Bar]:
        self.requests.append((symbol, start, end))
        if self.fail:
            raise SourceUnavailable(self.name, "down")
        return [b for b in self.bars.get(symbol, []) if start <= b.trade_date <= end]


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sent: list[str] = []
    provider = BarProvider()
    nse = ExchangeStub()
    monkeypatch.setattr(tasks.registry, "market_data_chain", lambda db, **kw: [provider])
    monkeypatch.setattr(tasks.registry, "nse_provider", lambda: nse)
    monkeypatch.setattr(runner.telegram, "send", lambda text, **kw: sent.append(text) or True)
    return {"sent": sent, "provider": provider, "nse": nse}


def _run(name: str, now: datetime, **kw: Any) -> str:
    return run_job(get_spec(name), now=now, sleep=lambda s: None, settings=SETTINGS, **kw)


def _last_run(db: Session, name: str) -> JobRun:
    db.expire_all()
    run = db.scalars(
        select(JobRun).where(JobRun.job_name == name).order_by(JobRun.id.desc()).limit(1)
    ).first()
    assert run is not None
    return run


def _predictions(db: Session) -> list[Prediction]:
    db.expire_all()
    return list(db.scalars(select(Prediction).order_by(Prediction.seq)).all())


def _grade(db: Session, p: Prediction) -> PredictionGrade | None:
    db.expire_all()
    return db.scalars(select(PredictionGrade).where(PredictionGrade.prediction_id == p.id)).first()


def _friday_bar(world: dict[str, Bar], *, open_pct: float, close_pct: float) -> Bar:
    ref = world[LAST.isoformat()].close
    o, c = ref * (1 + open_pct / 100), ref * (1 + close_pct / 100)
    return Bar(FRIDAY, o, max(o, c) * 1.002, min(o, c) * 0.998, c, 250_000.0)


# ------------------------------------------------------------------ 07:30 prediction


def test_premarket_records_a_locked_call_and_rerun_is_idempotent(
    db: Session, champion: ModelVersion, fakes: dict[str, Any]
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    (p,) = _predictions(db)
    assert (p.target, p.session_date, p.reference_date) == (TARGET, FRIDAY, LAST)
    assert p.model_version_id == champion.id and p.kind == "market_call"
    assert not p.is_late
    assert p.predicted_class in CLASSES and p.confidence in CONFIDENCE_LEVELS
    assert p.prob_up + p.prob_flat + p.prob_down == pytest.approx(1.0)
    assert set(p.baselines) == set(BASELINE_NAMES) and p.baselines["always_up"] == "UP"
    assert list(p.feature_snapshot) == features.FEATURE_NAMES
    assert p.gap_call in {"gap_up", "gap_down", "flat"}  # ES futures bars exist
    assert p.gap_expected_pct == pytest.approx(0.5 * p.feature_snapshot["es_fut_ret_1d"])
    assert p.timing_guidance and p.timing_guidance.count(".") <= 2
    assert 1 <= len(p.drivers) <= 5
    assert ledger.verify_ledger(db).ok
    # Bars were fresh through the reference session: nothing was fetched.
    assert fakes["provider"].requests == [] and fakes["nse"].calls == 1

    run = _last_run(db, "premarket_prediction")
    d = run.details[TARGET]
    assert d["status"] == "recorded" and d["model_version"] == champion.version
    assert (d["class"], d["confidence"], d["is_late"]) == (p.predicted_class, p.confidence, False)
    assert run.details["refresh"] == "ok" and run.details["brief"] == "sent"
    assert TARGET in run.message and "gaps" not in run.message
    (msg,) = fakes["sent"]  # the brief, and no partial alert
    assert TARGET in msg and "Fri 30 Oct 2026" in msg
    assert "Track record: no graded calls yet" in msg
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")

    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    assert len(_predictions(db)) == 1
    again = _last_run(db, "premarket_prediction")
    assert again.details[TARGET]["status"] == "already recorded"
    assert again.details["brief"].startswith("not sent")
    assert len(fakes["sent"]) == 1


def test_premarket_without_a_model_is_partial_with_the_training_hint(
    db: Session, world: dict[str, Bar], fakes: dict[str, Any]
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "partial"
    assert _predictions(db) == []
    run = _last_run(db, "premarket_prediction")
    assert "run `python -m app.cli train`" in (run.message or "")
    assert run.details[TARGET]["status"] == tasks.TRAIN_HINT
    assert run.details["brief"].startswith("not sent")
    (alert,) = fakes["sent"]
    assert "finished with gaps" in alert and "app.cli train" in alert


def test_premarket_reports_a_missing_cue_bar_but_still_records(
    db: Session, champion: ModelVersion, fakes: dict[str, Any]
) -> None:
    """Best effort: a cue that cannot be refreshed is missing in the snapshot, named in the
    details, and makes the run partial; the call is still made and locked."""
    dxy = instruments.get_instrument(db, "DX-Y.NYB", "GLOBAL")
    assert dxy is not None
    db.execute(
        text("DELETE FROM daily_bar WHERE instrument_id = :i AND trade_date > :d"),
        {"i": dxy.id, "d": date(2026, 10, 15)},
    )
    db.commit()
    fakes["provider"].fail = True
    fakes["nse"].fail = True

    assert _run("premarket_prediction", FRIDAY_0730) == "partial"
    (p,) = _predictions(db)
    assert p.feature_snapshot["dxy_ret_1d"] is None
    assert p.feature_snapshot["sp500_ret_1d"] is not None
    run = _last_run(db, "premarket_prediction")
    assert run.details[TARGET]["status"] == "recorded"
    (problem,) = run.details["refresh"]
    assert problem.startswith("GLOBAL:DX-Y.NYB has no bar for 2026-10-29")
    assert run.details["fii_dii"] == "unavailable: blocked"
    assert "DX-Y.NYB" in (run.message or "")
    assert [m for m in fakes["sent"] if "Not investment advice" in m], "brief still sent"


def test_premarket_marks_a_late_record(
    db: Session, champion: ModelVersion, fakes: dict[str, Any]
) -> None:
    late = datetime(2026, 10, 30, 9, 40, tzinfo=IST)  # GitHub started the run late
    assert _run("premarket_prediction", late) == "success"
    (p,) = _predictions(db)
    assert p.is_late and _last_run(db, "premarket_prediction").details[TARGET]["is_late"]
    assert "Recorded late" in fakes["sent"][0]


# ------------------------------------------------------------------ 09:20 open check


def test_open_check_grades_the_gap_once_the_open_exists(
    db: Session, world: dict[str, Bar], champion: ModelVersion, fakes: dict[str, Any]
) -> None:
    assert _run("open_check", FRIDAY_0920) == "skipped"
    assert "no market call recorded" in (_last_run(db, "open_check").message or "")

    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    (p,) = _predictions(db)
    # No bar for the session yet: partial, nothing invented.
    assert _run("open_check", FRIDAY_0920) == "partial"
    assert "not available yet" in (_last_run(db, "open_check").message or "")
    assert _grade(db, p) is None

    fakes["provider"].bars[TARGET] = [_friday_bar(world, open_pct=0.4, close_pct=0.1)]
    assert _run("open_check", FRIDAY_0920) == "success"
    g = _grade(db, p)
    assert g is not None and g.gap_graded_at is not None and g.actual_close is None
    assert g.gap_actual_pct == pytest.approx(0.4)
    assert g.gap_correct is (p.gap_call == "gap_up")
    assert g.source == "yfinance"
    d = _last_run(db, "open_check").details[f"{TARGET} seq {p.seq}"]
    assert d["status"] == "gap graded" and d["gap_actual_pct"] == pytest.approx(0.4)
    # The partial session bar is never stored as if it were the day's bar.
    inst = instruments.get_instrument(db, TARGET, "INDEX")
    assert inst is not None and db.get(DailyBar, (inst.id, FRIDAY)) is None

    assert _run("open_check", FRIDAY_0920) == "success"
    d = _last_run(db, "open_check").details[f"{TARGET} seq {p.seq}"]
    assert d["status"] == "already graded"
    assert len(fakes["provider"].requests) == 2, "no second fetch once graded"


# ------------------------------------------------------------------ 15:45 close grading


def test_close_grading_grades_the_call_and_sends_a_summary(
    db: Session, world: dict[str, Bar], champion: ModelVersion, fakes: dict[str, Any]
) -> None:
    assert _run("close_grading", FRIDAY_1545) == "skipped"
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    (p,) = _predictions(db)
    fakes["sent"].clear()

    assert _run("close_grading", FRIDAY_1545) == "partial"  # close not published yet
    run = _last_run(db, "close_grading")
    assert "no close yet" in (run.message or "") and run.details["graded"] == []
    assert _grade(db, p) is None

    fakes["provider"].bars[TARGET] = [_friday_bar(world, open_pct=-0.1, close_pct=1.0)]
    assert _run("close_grading", FRIDAY_1545) == "success"
    g = _grade(db, p)
    assert g is not None and g.actual_class == "UP"
    assert g.actual_pct == pytest.approx(1.0, abs=1e-6)
    assert g.correct is (p.predicted_class == "UP")
    assert g.brier is not None and g.log_loss is not None
    assert g.gap_graded_at is not None and g.gap_actual_pct == pytest.approx(-0.1)
    assert set(g.baseline_correct) == set(BASELINE_NAMES)
    inst = instruments.get_instrument(db, TARGET, "INDEX")
    assert inst is not None and db.get(DailyBar, (inst.id, FRIDAY)) is not None
    run = _last_run(db, "close_grading")
    (row,) = run.details["graded"]
    assert row["actual_class"] == "UP" and row["correct"] == g.correct
    assert run.details["summary"] == "sent"
    (msg,) = [m for m in fakes["sent"] if "close grading" in m]
    assert TARGET in msg and "+1.00%" in msg and ("correct" in msg or "wrong" in msg)
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")

    assert _run("close_grading", FRIDAY_1545) == "skipped"
    assert scorecard(db, TARGET, settings=SETTINGS).n_graded == 1


# ------------------------------------------------------------------ Saturday retrain


def test_weekly_retrain_promotes_the_first_model_then_rejects_an_equal_one(
    db: Session, world: dict[str, Bar], fakes: dict[str, Any]
) -> None:
    assert _run("weekly_retrain", SATURDAY_1000) == "success"  # not a trading day: fine
    db.expire_all()
    versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all())
    assert len(versions) == 1 and versions[0].is_champion
    d = _last_run(db, "weekly_retrain").details[TARGET]
    assert d["status"] == "promoted" and d["version"] == versions[0].version
    assert d["version"].startswith("lgbm-") and d["trained_through"] == LAST.isoformat()
    assert 0.0 <= d["oos_brier"] <= 2.0 and 0.0 <= d["oos_accuracy"] <= 1.0
    assert set(d["baselines"]) == set(BASELINE_NAMES)
    assert "no champion yet" in d["reason"]

    assert _run("weekly_retrain", SATURDAY_1000) == "success"
    db.expire_all()
    versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all())
    assert len(versions) == 2
    assert [v.is_champion for v in versions] == [True, False]
    d = _last_run(db, "weekly_retrain").details[TARGET]
    assert d["status"] == "rejected" and d["previous_champion"] == versions[0].version
    assert "Brier gain too small" in d["reason"]
    assert not fakes["sent"]


def test_weekly_retrain_without_data_is_partial(db: Session, fakes: dict[str, Any]) -> None:
    instruments.ensure_reference_instruments(db)
    db.commit()
    assert _run("weekly_retrain", SATURDAY_1000) == "partial"
    run = _last_run(db, "weekly_retrain")
    assert "no daily bars" in (run.message or "") and "app.cli backfill" in (run.message or "")
    assert db.scalar(select(ModelVersion)) is None


# ------------------------------------------------------------------ timing guidance


def test_timing_guidance_is_rule_based_and_short() -> None:
    assert tasks.timing_guidance(None, {"rvol_10d": 2.0, "rvol_20d": 1.0}) is None
    plain = tasks.timing_guidance("gap_up", {})
    assert plain == "Gap-up likely; avoid buying the first 15 minutes."
    trend = tasks.timing_guidance("gap_down", {"rvol_10d": 2.0, "rvol_20d": 1.0})
    assert trend is not None and trend.startswith("Gap-down likely") and "Trend day" in trend
    rng = tasks.timing_guidance("flat", {"rvol_10d": 0.5, "rvol_20d": 1.0})
    assert rng is not None and "Range day" in rng
    for s in (plain, trend, rng):
        assert s is not None and s.count(". ") <= 1  # at most two sentences
    assert tasks.timing_guidance("flat", {"rvol_10d": None, "rvol_20d": 1.0}) == (
        "Flat open likely; let the first 15 minutes set the range."
    )


# ------------------------------------------------------------------ Telegram texts


def _record_in(*, no_edge: bool, as_of: datetime = AS_OF) -> MarketCallRecordIn:
    probs = (
        {"UP": 0.36, "FLAT": 0.34, "DOWN": 0.30}
        if no_edge
        else {"UP": 0.6, "FLAT": 0.25, "DOWN": 0.15}
    )
    out = MarketCallOutput(
        probs=probs,
        predicted_class="UP",
        range_p10_pct=-0.9,
        range_p90_pct=1.1,
        drivers=[
            Driver("vix_level", 13.4, 0.2, "<b>VIX & co</b> at 13.4 pushes toward UP"),
            Driver("ret_1d", 0.5, 0.1, "1-day return at +0.50% pushes toward UP"),
        ],
        confidence="none" if no_edge else "high",
        no_edge=no_edge,
    )
    return MarketCallRecordIn(
        target="NIFTY <50>",
        session_date=SESSION,
        as_of=as_of,
        model_version_id=None,
        reference_date=SESSION - timedelta(days=3),
        reference_close=1000.0,
        output=out,
        gap_call=None,
        gap_expected_pct=None,
        timing_guidance=None,
        baselines={
            "always_up": "UP",
            "same_as_yesterday": None,
            "overnight_gap": None,
            "random": "DOWN",
        },
        feature_snapshot={"ret_1d": 0.5},
    )


def test_morning_brief_escapes_html_and_says_no_edge_today(db: Session) -> None:
    p = record_market_call(db, _record_in(no_edge=True))
    msg = brief.morning_brief([(p, None)], SESSION)
    assert "<b>No edge today</b>" in msg and "leans UP" in msg
    assert "<b>NIFTY &lt;50&gt;</b>" in msg and "NIFTY <50>" not in msg
    assert "&lt;b&gt;VIX &amp; co&lt;/b&gt; at 13.4" in msg and "<b>VIX" not in msg
    assert "UP 36% · FLAT 34% · DOWN 30%" in msg
    assert "Range (10–90%): -0.90% to +1.10%" in msg
    assert "Gap: no call (overnight futures missing)" in msg
    assert "Track record: unavailable" in msg
    assert "Recorded late" not in msg
    assert msg.count("<b>Dalal market call") == 1
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")


def test_morning_brief_shows_confidence_track_record_and_lateness(db: Session) -> None:
    late = datetime(2026, 11, 2, 9, 30, tzinfo=IST)
    p = record_market_call(db, _record_in(no_edge=False, as_of=late))
    grade_close(db, p, None, 1006.0, now=late, source="test")  # +0.60%: UP, correct
    card = scorecard(db, p.target)
    msg = brief.morning_brief([(p, card)], SESSION)
    assert "<b>UP</b> (high confidence)" in msg and "No edge today" not in msg
    assert "Track record: 1 graded, 100% right vs best baseline (always UP) 100%" in msg
    assert "unproven (&lt; 100 graded)" in msg
    assert "Recorded late (after 09:15 IST)." in msg
    assert "Drivers:" in msg and msg.count("•") == 2

    empty = brief.morning_brief([], SESSION)
    assert "No market call was recorded" in empty and "Not investment advice" in empty


def test_close_summary_states_the_verdict_per_call(db: Session) -> None:
    right = record_market_call(db, make_record(probs={"UP": 0.6, "FLAT": 0.25, "DOWN": 0.15}))
    wrong = record_market_call(
        db, make_record(target="NIFTY BANK", probs={"UP": 0.2, "FLAT": 0.3, "DOWN": 0.5})
    )
    now = datetime(2026, 11, 2, 10, 16, tzinfo=IST)
    g1 = grade_close(db, right, 1004.0, 1010.0, now=now, source="test")  # +1.0% UP
    g2 = grade_close(db, wrong, None, 1010.0, now=now, source="test")
    msg = brief.close_summary([(right, g1), (wrong, g2)])
    assert "<b>Dalal close grading — Mon 02 Nov 2026</b>" in msg
    assert "NIFTY 50: called <b>UP</b> (60%), actual +1.00% <b>UP</b> — correct. Brier" in msg
    assert "NIFTY BANK: called <b>DOWN</b> (50%), actual +1.00% <b>UP</b> — wrong." in msg
    assert "Gap call gap up: correct (+0.40% actual)" in msg
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")
    assert "Nothing was graded" in brief.close_summary([])


# ------------------------------------------------------------------ CLI


def test_cli_verify_ledger_exit_codes(db: Session, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["verify-ledger"]) == 0  # an empty ledger is intact
    assert "ledger is empty" in capsys.readouterr().out
    for i in range(2):
        record_market_call(db, make_record(session_date=SESSION + timedelta(days=i)))
    db.commit()
    assert cli.main(["verify-ledger"]) == 0
    assert "OK: 2 records verified" in capsys.readouterr().out

    db.execute(text("ALTER TABLE prediction DISABLE TRIGGER prediction_immutable"))
    db.execute(text("UPDATE prediction SET note = 'edited' WHERE seq = 1"))
    db.execute(text("ALTER TABLE prediction ENABLE TRIGGER prediction_immutable"))
    db.commit()
    assert cli.main(["verify-ledger"]) == 1
    out = capsys.readouterr().out
    assert "BROKEN" in out and "seq 1" in out and "first bad seq: 1" in out


def test_cli_job_commands_pass_force_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        runner, "run_job", lambda spec, **kw: calls.append((spec.name, kw["force"])) or "success"
    )
    assert cli.main(["predict", "--force"]) == 0
    assert cli.main(["predict"]) == 0
    assert cli.main(["grade"]) == 0
    assert cli.main(["run-job", "close_grading", "--force"]) == 0
    assert cli.main(["run-job", "evening_data_pull"]) == 0
    assert calls == [
        ("premarket_prediction", True),
        ("premarket_prediction", False),
        ("close_grading", False),
        ("close_grading", True),
        ("evening_data_pull", False),
    ]
    monkeypatch.setattr(runner, "run_job", lambda spec, **kw: "failed")
    assert cli.main(["predict"]) == 1


def test_cli_train_and_scorecard(
    db: Session,
    world: dict[str, Bar],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    assert cli.main(["train", "--target", TARGET]) == 0
    out = capsys.readouterr().out
    assert "PROMOTED" in out and "no champion yet" in out and "baseline always_up" in out
    first = model_registry.champion(db, TARGET)
    assert first is not None

    # The same data again: an equal challenger is saved but not promoted...
    assert cli.main(["train", "--target", TARGET]) == 0
    out = capsys.readouterr().out
    assert "rejected" in out and "Brier gain too small" in out
    db.expire_all()
    assert model_registry.champion(db, TARGET).id == first.id  # type: ignore[union-attr]
    # ...unless told to promote regardless.
    assert cli.main(["train", "--target", TARGET, "--promote-always"]) == 0
    assert "--promote-always" in capsys.readouterr().out
    db.expire_all()
    promoted = model_registry.champion(db, TARGET)
    assert promoted is not None and promoted.id != first.id
    assert promoted.promotion_note and promoted.promotion_note.startswith("--promote-always")

    assert cli.main(["scorecard", "--target", TARGET]) == 0
    out = capsys.readouterr().out
    assert "No graded market calls yet" in out and "sample-size warning" in out
    assert f"champion: {promoted.version}" in out and "baseline always UP" in out

    assert cli.main(["train", "--target", "NO SUCH INDEX"]) == 1  # the honest failure path
    assert "no daily bars" in capsys.readouterr().err


# ------------------------------------------------------------------ workflow + env keys


def _workflow() -> dict[str, Any]:
    doc = yaml.safe_load((REPO / ".github" / "workflows" / "dalal-jobs.yml").read_text())
    return {"on": doc[True] if True in doc else doc["on"], "jobs": doc["jobs"]}


def _bash_matches(cmd: str) -> bool:
    res = subprocess.run(
        ["bash", "-c", '[[ "$1" =~ $2 ]]', "_", cmd, ALLOWLIST], capture_output=True
    )
    return res.returncode == 0


def test_jobs_workflow_schedules_the_phase2_jobs_and_keeps_the_allowlist() -> None:
    wf = _workflow()
    crons = {c["cron"] for c in wf["on"]["schedule"]}
    expected = {
        "0 2 * * 1-5": "premarket_prediction",
        "50 3 * * 1-5": "open_check",
        "15 10 * * 1-5": "close_grading",
        "30 4 * * 6": "weekly_retrain",
    }
    assert set(expected) <= crons
    options = wf["on"]["workflow_dispatch"]["inputs"]["command"]["options"]
    assert {
        "run-job premarket_prediction",
        "run-job open_check",
        "run-job close_grading",
        "run-job weekly_retrain",
        "train",
        "predict --force",
        "verify-ledger",
    } <= set(options)
    steps = {s.get("id") or s.get("name"): s for s in wf["jobs"]["run"]["steps"]}
    pick, run = steps["pick"]["run"], steps["Run"]["run"]
    for cron, job in expected.items():
        assert f'"{cron}")' in pick and f'CMD="run-job {job}"' in pick, job
    assert f"ALLOW='{ALLOWLIST}'" in pick and '[[ "$CMD" =~ $ALLOW ]]' in pick
    assert 'read -ra ARGS <<< "$CMD"' in run and 'python -m app.cli "${ARGS[@]}"' in run
    # The allowlist, evaluated by bash exactly as the workflow does.
    for opt in options:
        assert _bash_matches(opt), opt
    for bad in ("run-job evening_data_pull; rm -rf /", "predict --force --now", "train x", ""):
        assert not _bash_matches(bad), bad


def test_vercel_env_exports_the_phase2_tunables(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "vercel_env", REPO / ".github" / "scripts" / "vercel_env.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for key in (
        "PREDICTION_TARGETS",
        "WF_MIN_TRAIN_DAYS",
        "MLFLOW_TRACKING_URI",
        "MIN_SAMPLE_SIZE",
    ):
        assert key in mod.APP_KEYS
    f = tmp_path / ".env"
    f.write_text(
        'DATABASE_URL="postgres://x"\nSECRET_KEY="s"\nPREDICTION_TARGETS="NIFTY 50"\n'
        'MLFLOW_TRACKING_URI="https://u:pw@mlflow.example"\n'
    )
    env = mod.parse(str(f))
    lines = [mod.env_line(k, env[k]) for k in mod.APP_KEYS if mod.readable(env, k)]
    assert "PREDICTION_TARGETS=NIFTY 50" in lines
    assert mod.SECRETISH.search("MLFLOW_TRACKING_URI"), "a tracking URI may carry credentials"
    assert not mod.SECRETISH.search("MIN_SAMPLE_SIZE")

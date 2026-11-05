"""API: auth gate, login, Today (incl. the market call and the ideas), Ideas, Paper,
Scorecard (market call + ideas), Stock page, Health.

No network: providers are faked. Ledger rows are seeded through
`app.services.ledger.record_market_call` with a minimal `MarketCallRecordIn` and
`app.services.ledger.record_idea` with `tests.test_idea_ledger.make_idea`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.health import market_call_freshness
from app.api.ideas import STATUS_FILTERS, SURVEILLANCE_NOTE
from app.cache import get_kv
from app.config import IST, get_settings
from app.main import app
from app.ml.core import (
    BASELINE_NAMES,
    CONFIDENCE_LEVELS,
    IDEA_STATUSES,
    MODEL_NAME,
    CostModel,
    Driver,
    MarketCallOutput,
    MarketCallRecordIn,
    session_for,
)
from app.ml.ideas_models import model_name, model_target
from app.models import Idea, ModelVersion, Prediction
from app.providers.base import Bar
from app.security.auth import create_user
from app.services import instruments, ledger, paper
from app.services.calendar import TradingCalendar
from app.services.flows import store as store_flows
from app.services.grading import grade_close
from app.services.idea_grading import grade_idea, open_check_fill
from app.services.market_data import upsert_daily_bars
from app.services.scorecard import MIN_PAIRED_N
from tests.test_idea_ledger import SNAPSHOT, make_idea, make_instrument
from tests.test_ideas_scorecard import OOS_METRICS as IDEA_OOS_METRICS
from tests.test_scorecard import OOS_METRICS, captured_sql

PW = "correct horse battery staple"
P_UP = {"UP": 0.52, "FLAT": 0.30, "DOWN": 0.18}
P_NO_EDGE = {"UP": 0.36, "FLAT": 0.33, "DOWN": 0.31}


def _current_session(db: Session) -> date:
    """The session /api/today reports on: the same rule the endpoint uses, evaluated now."""
    return session_for(TradingCalendar.load(db), datetime.now(UTC))


def _record_call(
    db: Session,
    target: str = "NIFTY 50",
    session: date | None = None,
    *,
    probs: dict[str, float] | None = None,
    confidence: str = "medium",
    no_edge: bool = False,
    corrects: int | None = None,
    gap_baseline: str | None = "UP",
) -> Prediction:
    """Append one minimal market call to the ledger (07:30 IST of its session) and commit.

    `gap_baseline` is the overnight-gap baseline's call (None = no futures that night).
    """
    cal = TradingCalendar.load(db)
    session = session or session_for(cal, datetime.now(UTC))
    p = probs or P_UP
    out = MarketCallOutput(
        probs=p,
        predicted_class=max(p, key=lambda c: p[c]),
        range_p10_pct=-0.9,
        range_p90_pct=1.1,
        drivers=[
            Driver("es_fut_ret_1d", 0.8, 0.12, "S&P 500 futures 1-day return at +0.80%"),
            Driver("vix_level", None, -0.02, "India VIX level missing"),
        ],
        confidence=confidence,
        no_edge=no_edge,
    )
    rec = MarketCallRecordIn(
        target=target,
        session_date=session,
        as_of=datetime.combine(session, time(7, 30), tzinfo=IST),
        model_version_id=None,
        reference_date=cal.previous_trading_day(session),
        reference_close=24500.0,
        output=out,
        gap_call="gap_up",
        gap_expected_pct=0.4,
        timing_guidance="Gap-up likely; avoid buying the first 15 minutes.",
        baselines={"always_up": "UP", "same_as_yesterday": "DOWN", "overnight_gap": gap_baseline},
        feature_snapshot={"ret_1d": 0.3, "vix_level": None},
        corrects_prediction_id=corrects,
    )
    row = ledger.record_market_call(db, rec)
    db.commit()
    return row


IDEA_NOW = datetime.now(UTC)


def _record_idea(
    db: Session,
    symbol: str = "RELIANCE",
    horizon: str = "swing",
    session: date | None = None,
    **kw: object,
) -> Idea:
    """Append one idea (07:35 IST of its session) to the idea ledger and commit."""
    session = session or _current_session(db)
    inst = make_instrument(db, symbol)
    row = ledger.record_idea(
        db,
        make_idea(
            inst.id,
            symbol,
            horizon,
            session_date=session,
            as_of=datetime.combine(session, time(7, 35), tzinfo=IST),
            **kw,  # type: ignore[arg-type]
        ),
    )
    db.commit()
    return row


def _bars(start: date, *rows: tuple[int, float, float, float, float]) -> pd.DataFrame:
    """Daily bars like `market_data.daily_frame`: (day offset from start, o, h, l, c)."""
    idx = pd.to_datetime([start + timedelta(days=d) for d, *_ in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0e6] * len(rows),
            "delivery_pct": [float("nan")] * len(rows),
            "source": ["test"] * len(rows),
        },
        index=idx,
    )


def _ideas_champion(horizon: str, metrics: dict | None = None) -> ModelVersion:
    return ModelVersion(
        name=model_name(horizon),
        target=model_target(),
        version=f"lgbm-ideas-{horizon}",
        kind="lightgbm",
        params={},
        feature_names=["ret_20d"],
        metrics=metrics or {"n_oos": 10},
        artifact="{" + '"classifier": "' + "x" * 2000 + '"}',
        trained_through=date(2026, 10, 30),
        train_rows=5000,
        is_champion=True,
        promoted_at=IDEA_NOW,
    )


@pytest.fixture()
def client(db: Session) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def logged_in(db: Session, client: TestClient) -> TestClient:
    new = create_user(db, "me", PW)
    db.commit()
    resp = client.post(
        "/api/auth/login",
        json={"username": "me", "password": PW, "totp": pyotp.TOTP(new.totp_secret).now()},
    )
    assert resp.status_code == 200, resp.text
    assert "dalal_session" in resp.cookies
    return client


def test_healthz_is_public(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"ok": True}


@pytest.mark.parametrize(
    "path",
    [
        "/api/today",
        "/api/health",
        "/api/stocks/NSE/RELIANCE",
        "/api/kite/status",
        "/api/auth/me",
        "/api/stocks/search?q=REL",
        "/api/scorecard",
        "/api/scorecard?target=NIFTY%20BANK",
        "/api/scorecard/ledger",
        "/api/scorecard/ideas",
        "/api/ideas",
        "/api/ideas?status=open&horizon=swing",
        "/api/ideas/1",
        "/api/paper",
        "/api/paper/trades",
        "/api/today/market-call/1/factors",
    ],
)
def test_everything_else_requires_login(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_verify_ledger_requires_login(client: TestClient) -> None:
    assert client.post("/api/scorecard/verify-ledger").status_code == 401


def test_bad_login_rejected(db: Session, client: TestClient) -> None:
    create_user(db, "me", PW)
    db.commit()
    resp = client.post("/api/auth/login", json={"username": "me", "password": PW, "totp": "000000"})
    assert resp.status_code == 401
    assert "dalal_session" not in resp.cookies


def test_logout(logged_in: TestClient) -> None:
    assert logged_in.get("/api/auth/me").json() == {"username": "me"}
    assert logged_in.post("/api/auth/logout").status_code == 204
    logged_in.cookies.clear()
    assert logged_in.get("/api/auth/me").status_code == 401


def test_today_shows_missing_data_honestly(logged_in: TestClient) -> None:
    body = logged_in.get("/api/today").json()
    assert body["market_call"] == []
    assert body["market_call_note"].startswith("No call recorded for ")
    assert "07:30" in body["market_call_note"] and "no trained model" in body["market_call_note"]
    assert body["flows"]["status"] == "unavailable"
    cues = {c["key"]: c for c in body["cues"]}
    assert cues["sp500"]["status"] == "unavailable" and cues["sp500"]["last"] is None
    assert cues["gift_nifty"]["status"] == "unavailable"
    assert body["kite"]["configured"] is False
    assert body["clock"]["calendar_verified"] is False


def test_today_flows(db: Session, logged_in: TestClient) -> None:
    from app.providers.base import FlowRecord

    store_flows(
        db,
        [
            FlowRecord(date(2026, 10, 29), "FII", 100.0, 150.0, -50.0),
            FlowRecord(date(2026, 10, 29), "DII", 200.0, 120.0, 80.0),
        ],
        "nse",
    )
    db.commit()
    flows = logged_in.get("/api/today").json()["flows"]
    assert flows["days"][0]["fii_net_cr"] == -50.0 and flows["days"][0]["dii_net_cr"] == 80.0
    assert flows["days"][0]["provisional"] is True


def test_today_shows_the_recorded_call_with_grade_and_track_record(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    row = _record_call(db, "NIFTY 50")

    body = logged_in.get("/api/today").json()
    assert body["market_call_session"] == session.isoformat()
    assert len(body["market_call"]) == 1
    call = body["market_call"][0]
    assert call["id"] == row.id and call["seq"] == 1 and call["is_previous"] is False
    assert call["target"] == "NIFTY 50" and call["session_date"] == session.isoformat()
    assert call["predicted_class"] == "UP" and call["probs"] == P_UP
    assert call["confidence"] == "medium" and call["no_edge"] is False
    assert call["is_late"] is False  # recorded at 07:30 IST, before the 09:15 lock
    assert call["range_p10_pct"] == -0.9 and call["range_p90_pct"] == 1.1
    assert call["gap_call"] == "gap_up" and call["gap_expected_pct"] == 0.4
    assert call["reference_close"] == 24500.0
    assert [d["feature"] for d in call["drivers"]] == ["es_fut_ret_1d", "vix_level"]
    assert call["drivers"][1]["value"] is None  # a missing number stays missing
    assert call["timing_guidance"].startswith("Gap-up likely")
    assert call["baselines"]["always_up"] == "UP" and call["model_version"] is None
    assert call["grade"] is None
    track = call["track"]
    assert track["n_graded"] == 0 and track["accuracy"] is None
    assert track["best_baseline"] is None and track["best_baseline_accuracy"] is None
    assert track["paired_n"] is None and track["model_paired_accuracy"] is None
    assert track["comparison"] == "no_calls" and track["min_paired_n"] == MIN_PAIRED_N
    assert track["sample_warning"] is True and track["min_sample_size"] == 100
    # The other configured target has no call at all, and the note says so.
    assert "NIFTY BANK" in body["market_call_note"]
    assert "NIFTY 50" not in body["market_call_note"]

    # A "No edge today" call for the second target, then both are shown and the note is empty.
    _record_call(db, "NIFTY BANK", probs=P_NO_EDGE, confidence="none", no_edge=True)
    body = logged_in.get("/api/today").json()
    assert [c["target"] for c in body["market_call"]] == ["NIFTY 50", "NIFTY BANK"]
    bank = body["market_call"][1]
    assert bank["no_edge"] is True and bank["confidence"] == "none"
    assert bank["predicted_class"] == "UP"  # the lean is still recorded in the ledger
    assert body["market_call_note"] == ""

    # After the close grade the card shows the outcome and the track record moves.
    grade_close(db, row, 24550.0, 24700.0, now=datetime.now(UTC), source="test")
    db.commit()
    call = logged_in.get("/api/today").json()["market_call"][0]
    assert call["grade"]["actual_close"] == 24700.0
    assert call["grade"]["actual_class"] == "UP" and call["grade"]["correct"] is True
    assert call["grade"]["actual_pct"] == pytest.approx(200 / 24500 * 100)
    assert call["grade"]["gap_correct"] is False  # +0.20 % open is below the 0.25 % gap threshold
    assert call["grade"]["brier"] == pytest.approx((1 - 0.52) ** 2 + 0.30**2 + 0.18**2)
    assert call["track"]["n_graded"] == 1 and call["track"]["accuracy"] == 1.0
    assert call["track"]["best_baseline"] == "always_up"
    assert call["track"]["best_baseline_label"] == "always UP"
    assert call["track"]["best_baseline_accuracy"] == 1.0
    assert call["track"]["paired_n"] == 1 and call["track"]["model_paired_accuracy"] == 1.0
    assert call["track"]["comparison"] == "too_early"  # one paired call proves nothing
    assert call["track"]["sample_warning"] is True


def test_today_track_line_uses_the_paired_baseline_numbers(
    db: Session, logged_in: TestClient
) -> None:
    """The gap rule only called on the sessions the model got wrong: paired, it is ahead."""
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    days: list[date] = []
    d = session
    while len(days) < 4:
        d = cal.previous_trading_day(d)
        days.append(d)
    now = datetime.now(UTC)
    for i, day in enumerate(days):
        p = _record_call(db, "NIFTY 50", session=day, gap_baseline="DOWN" if i < 2 else None)
        grade_close(db, p, None, 24400.0 if i < 2 else 24700.0, now=now, source="test")
    _record_call(db, "NIFTY 50", session=session)
    db.commit()

    track = logged_in.get("/api/today").json()["market_call"][0]["track"]
    assert track["n_graded"] == 4 and track["accuracy"] == 0.5  # UP call, 2 UP / 2 DOWN closes
    # Toughest comparison: on the 2 sessions the gap rule called, it was 2/2 and the model 0/2.
    assert track["best_baseline"] == "overnight_gap"
    assert track["paired_n"] == 2 and track["best_baseline_accuracy"] == 1.0
    assert track["model_paired_accuracy"] == 0.0
    assert track["comparison"] == "too_early"


def test_today_falls_back_to_the_latest_previous_call_with_its_grade(
    db: Session, logged_in: TestClient
) -> None:
    """After 15:30 IST the upcoming session has no call yet; the page keeps the graded one."""
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    previous = cal.previous_trading_day(session)
    _record_call(db, "NIFTY 50", session=previous - timedelta(days=14))  # older, superseded by date
    last = _record_call(db, "NIFTY 50", session=previous)
    grade_close(db, last, 24550.0, 24700.0, now=datetime.now(UTC), source="test")
    db.commit()

    body = logged_in.get("/api/today").json()
    assert body["market_call_session"] == session.isoformat()
    [call] = body["market_call"]
    assert call["id"] == last.id and call["is_previous"] is True
    assert call["session_date"] == previous.isoformat()
    assert call["grade"]["actual_class"] == "UP" and call["grade"]["correct"] is True
    assert call["grade"]["actual_pct"] == pytest.approx(200 / 24500 * 100)
    assert call["track"]["n_graded"] == 1
    # Only the target with no call at all is in the note; NIFTY 50 has one to show.
    assert "NIFTY BANK" in body["market_call_note"]
    assert "NIFTY 50" not in body["market_call_note"]
    assert not body["market_call_note"].startswith("No call recorded for ")

    # Once the upcoming session's call is recorded it takes over (and is not "previous").
    current = _record_call(db, "NIFTY 50", session=session)
    [call] = logged_in.get("/api/today").json()["market_call"]
    assert call["id"] == current.id and call["is_previous"] is False and call["grade"] is None


def test_today_does_not_show_a_future_call_as_the_previous_one(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    _record_call(db, "NIFTY 50", session=cal.next_trading_day(session))
    body = logged_in.get("/api/today").json()
    assert body["market_call"] == []
    assert body["market_call_note"].startswith(f"No call recorded for {session.isoformat()}")


def test_today_shows_the_correction_not_the_superseded_call(
    db: Session, logged_in: TestClient
) -> None:
    original = _record_call(db, "NIFTY 50")
    fix = _record_call(
        db, "NIFTY 50", probs={"UP": 0.2, "FLAT": 0.3, "DOWN": 0.5}, corrects=original.id
    )
    calls = logged_in.get("/api/today").json()["market_call"]
    assert [c["id"] for c in calls] == [fix.id]
    assert calls[0]["predicted_class"] == "DOWN" and calls[0]["seq"] == 2


def test_today_does_not_read_the_feature_snapshot(db: Session, logged_in: TestClient) -> None:
    row = _record_call(db, "NIFTY 50")
    grade_close(db, row, None, 24700.0, now=datetime.now(UTC), source="test")
    db.commit()
    with captured_sql(db) as sql:
        body = logged_in.get("/api/today").json()
    assert body["market_call"][0]["drivers"], "the card still shows the drivers"
    selects = [s for s in sql if s.lstrip().upper().startswith("SELECT")]
    assert selects and all("feature_snapshot" not in s for s in selects)


# ----------------------------------------------------------------- scorecard


def test_scorecard_endpoint_shape_when_empty(logged_in: TestClient) -> None:
    body = logged_in.get("/api/scorecard").json()
    assert body["target"] == "NIFTY 50"  # the first configured target is the default
    assert body["targets"] == ["NIFTY 50", "NIFTY BANK"]
    assert body["n_graded"] == 0 and body["gap_n"] == 0
    assert body["accuracy"] is None and body["brier"] is None and body["log_loss"] is None
    assert body["gap_accuracy"] is None
    assert body["by_confidence"] == [] and body["calibration"] == [] and body["recent"] == []
    assert [b["name"] for b in body["baselines"]] == list(BASELINE_NAMES)
    assert all(b["n"] == 0 and b["accuracy"] is None for b in body["baselines"])
    assert all(b["model_accuracy"] is None for b in body["baselines"])
    assert body["best_baseline"] is None and body["comparison"] == "no_calls"
    assert body["min_paired_n"] == MIN_PAIRED_N
    assert body["sample_warning"] is True and body["min_sample_size"] == 100
    assert body["champion"] is None
    assert "unproven" in body["verdict"]
    other = logged_in.get("/api/scorecard?target=nifty%20bank").json()
    assert other["target"] == "NIFTY BANK" and other["n_graded"] == 0


def test_scorecard_unknown_target_is_404(db: Session, logged_in: TestClient) -> None:
    resp = logged_in.get("/api/scorecard?target=SENSEX")
    assert resp.status_code == 404
    assert "Unknown target 'SENSEX'" in resp.json()["detail"]
    assert "NIFTY 50, NIFTY BANK" in resp.json()["detail"]
    assert logged_in.get("/api/scorecard/ledger?target=SENSEX").status_code == 404
    # A target that is not configured but has ledger records is known.
    _record_call(db, "SENSEX")
    body = logged_in.get("/api/scorecard?target=sensex").json()
    assert body["target"] == "SENSEX" and body["targets"] == ["NIFTY 50", "NIFTY BANK", "SENSEX"]
    assert len(logged_in.get("/api/scorecard/ledger?target=SENSEX").json()) == 1


def test_scorecard_endpoint_aggregates_graded_calls(db: Session, logged_in: TestClient) -> None:
    session = _current_session(db)
    first = _record_call(db, "NIFTY 50", session=session - timedelta(days=3), confidence="high")
    second = _record_call(db, "NIFTY 50", session=session - timedelta(days=2))
    _record_call(db, "NIFTY 50", session=session)  # today: ungraded
    _record_call(db, "NIFTY BANK", session=session)  # another target: not counted
    now = datetime.now(UTC)
    grade_close(db, first, 24550.0, 24700.0, now=now, source="test")  # UP: correct
    grade_close(db, second, None, 24400.0, now=now, source="test")  # DOWN: wrong
    db.commit()

    body = logged_in.get("/api/scorecard?target=NIFTY%2050").json()
    assert body["n_graded"] == 2 and body["accuracy"] == 0.5
    assert body["brier"] is not None and body["log_loss"] is not None
    assert body["gap_n"] == 1 and body["gap_accuracy"] == 0.0
    assert [r["confidence"] for r in body["by_confidence"]] == list(CONFIDENCE_LEVELS)
    by_conf = {r["confidence"]: r for r in body["by_confidence"]}
    assert by_conf["high"] == {"confidence": "high", "n": 1, "accuracy": 1.0}
    assert by_conf["medium"] == {"confidence": "medium", "n": 1, "accuracy": 0.0}
    assert by_conf["none"] == {"confidence": "none", "n": 0, "accuracy": None}
    assert sum(b["n"] for b in body["calibration"]) == 2
    for b in body["calibration"]:
        assert set(b) == {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}
    base = {b["name"]: b for b in body["baselines"]}
    assert base["always_up"] == {
        "name": "always_up",
        "label": "always UP",
        "n": 2,
        "accuracy": 0.5,
        "model_accuracy": 0.5,
    }
    assert base["random"]["n"] == 0 and base["random"]["accuracy"] is None  # no random call
    assert body["best_baseline"]["name"] == "always_up" and body["best_baseline"]["n"] == 2
    assert body["comparison"] == "too_early"
    assert "After 2 graded calls" in body["verdict"] and "50%" in body["verdict"]
    assert "Too early to tell" in body["verdict"]
    assert body["sample_warning"] is True

    recent = body["recent"]
    assert [r["session_date"] for r in recent] == [
        session.isoformat(),
        (session - timedelta(days=2)).isoformat(),
        (session - timedelta(days=3)).isoformat(),
    ]
    assert recent[0]["actual_class"] is None and recent[0]["correct"] is None
    assert recent[1]["actual_class"] == "DOWN" and recent[1]["correct"] is False
    assert recent[2]["correct"] is True and recent[2]["gap_correct"] is False
    assert recent[2]["probs"] == P_UP and recent[2]["is_late"] is False
    assert all(r["target"] == "NIFTY 50" for r in recent)


def test_scorecard_champion_box(db: Session, logged_in: TestClient) -> None:
    db.add(
        ModelVersion(
            name=MODEL_NAME,
            target="NIFTY 50",
            version="lgbm-20261031T0430Z-abc123",
            kind="lightgbm",
            params={},
            feature_names=["ret_1d"],
            metrics={"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03},
            artifact="{}",
            trained_through=date(2026, 10, 30),
            train_rows=1200,
            is_champion=True,
            promoted_at=datetime.now(UTC),
        )
    )
    db.commit()
    champ = logged_in.get("/api/scorecard").json()["champion"]
    assert champ["version"] == "lgbm-20261031T0430Z-abc123" and champ["kind"] == "lightgbm"
    assert champ["trained_through"] == "2026-10-30" and champ["train_rows"] == 1200
    assert champ["oos"] == {"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03}
    assert champ["promoted_at"] is not None
    # Stored without baselines: the box says so instead of showing invented numbers.
    assert champ["baselines"] == [] and champ["class_frequencies"] is None
    assert champ["range_coverage"] is None and champ["climatology_brier"] is None
    assert "no class-frequency Brier was stored" in champ["oos_verdict"]
    assert logged_in.get("/api/scorecard?target=NIFTY%20BANK").json()["champion"] is None


def test_scorecard_champion_box_with_oos_baselines(db: Session, logged_in: TestClient) -> None:
    db.add(
        ModelVersion(
            name=MODEL_NAME,
            target="NIFTY 50",
            version="lgbm-full",
            kind="lightgbm",
            params={},
            feature_names=["ret_1d"],
            metrics=OOS_METRICS,
            artifact="{" + '"classifier": "' + "x" * 2000 + '"}',
            trained_through=date(2026, 10, 30),
            train_rows=1200,
            is_champion=True,
            promoted_at=datetime.now(UTC),
        )
    )
    db.commit()
    with captured_sql(db) as sql:
        body = logged_in.get("/api/scorecard").json()
    champ = body["champion"]
    assert [b["name"] for b in champ["baselines"]] == list(BASELINE_NAMES)
    assert champ["baselines"][3] == {
        "name": "random",
        "label": "random with historical class frequencies",
        "n": 400,
        "accuracy": 0.34,
        "brier": 0.668,
    }
    assert champ["class_frequencies"] == {"UP": 0.39, "FLAT": 0.3, "DOWN": 0.31}
    assert champ["range_coverage"] == 0.78 and champ["climatology_brier"] == 0.668
    assert champ["oos_verdict"].startswith(
        "Out of sample the model's probabilities were worse than class frequencies "
        "(Brier 0.701 vs 0.668)."
    )
    selects = [s for s in sql if s.lstrip().upper().startswith("SELECT")]
    assert selects and all("artifact" not in s for s in selects)
    assert all("feature_snapshot" not in s and "drivers" not in s for s in selects)


def test_scorecard_ledger_rows(db: Session, logged_in: TestClient) -> None:
    session = _current_session(db)
    _record_call(db, "NIFTY 50", session=session - timedelta(days=1))
    _record_call(db, "NIFTY BANK", session=session)
    _record_call(db, "NIFTY 50", session=session)

    rows = logged_in.get("/api/scorecard/ledger").json()
    assert [r["seq"] for r in rows] == [3, 2, 1]  # newest first, every target
    assert rows[0]["target"] == "NIFTY 50" and rows[0]["actual_close"] is None
    assert set(rows[0]) >= {"id", "seq", "probs", "confidence", "no_edge", "brier", "graded_at"}
    nifty = logged_in.get("/api/scorecard/ledger?target=NIFTY%2050").json()
    assert [r["seq"] for r in nifty] == [3, 1]
    assert len(logged_in.get("/api/scorecard/ledger?limit=1").json()) == 1
    assert logged_in.get("/api/scorecard/ledger?limit=0").status_code == 422


def test_verify_ledger_endpoint(db: Session, logged_in: TestClient) -> None:
    body = logged_in.post("/api/scorecard/verify-ledger").json()
    assert body == {"ok": True, "checked": 0, "first_bad_seq": None, "message": "ledger is empty"}
    _record_call(db, "NIFTY 50")
    _record_call(db, "NIFTY BANK")
    body = logged_in.post("/api/scorecard/verify-ledger").json()
    assert body["ok"] is True and body["checked"] == 2 and body["first_bad_seq"] is None
    assert body["message"] == "2 records verified"


def _seed_bars(db: Session, symbol: str, n: int = 400) -> date:
    inst = instruments.get_or_create_equity(db, symbol, "NSE")
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days: list[date] = []
    d = date.today() - timedelta(days=int(n * 1.45))
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    bars = [Bar(day, c, c * 1.01, c * 0.99, c, 1e5) for day, c in zip(days, close, strict=True)]
    upsert_daily_bars(db, inst.id, bars, "yfinance")
    db.commit()
    return days[-1]


def test_stock_page_returns_candles_indicators_levels(db: Session, logged_in: TestClient) -> None:
    last_day = _seed_bars(db, "TESTCO", n=600)
    # Pretend we fetched recently so the endpoint doesn't call any provider.
    get_kv().set("stockfetch:NSE:TESTCO", "1", ex=60)
    body = logged_in.get("/api/stocks/NSE/TESTCO?range=1y").json()
    assert body["instrument"]["symbol"] == "TESTCO"
    assert body["candles"] and body["candles"][-1]["time"] == last_day.isoformat()
    assert body["sources"] == ["yfinance"]
    for key in ("sma20", "sma50", "sma200", "bb_upper", "rsi14", "macd"):
        assert body["indicators"][key], key
    # SMA200 is defined from the first visible bar thanks to the warm-up window.
    assert body["indicators"]["sma200"][0]["time"] == body["candles"][0]["time"]
    assert body["data_status"] in {"ok", "stale"}


def test_unknown_symbol_404_and_not_persisted(
    db: Session, logged_in: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api import stocks as stocks_api

    monkeypatch.setattr(stocks_api.registry, "market_data_chain", lambda db, **kw: [])
    resp = logged_in.get("/api/stocks/NSE/NOSUCHCO")
    assert resp.status_code == 404
    assert instruments.get_instrument(db, "NOSUCHCO", "NSE") is None


def test_health_lists_all_jobs_and_sources(logged_in: TestClient) -> None:
    body = logged_in.get("/api/health").json()
    names = {j["name"]: j for j in body["jobs"]}
    assert names["premarket_prediction"]["phase"] == 2
    assert names["evening_data_pull"]["built"] is True
    assert {s["source"] for s in body["sources"]} >= {"kite", "yfinance", "nse", "telegram"}
    assert body["scheduler_alive"] is False
    assert body["models"] == []
    assert "No champion model" in body["models_note"]
    datasets = {f["dataset"]: f for f in body["freshness"]}
    assert datasets["NSE holiday calendar"]["status"] == "missing"
    assert datasets["Last market call"]["status"] == "missing"
    assert datasets["Last market call"]["latest"] is None


def test_health_reports_champions_and_the_last_market_call(
    db: Session, logged_in: TestClient
) -> None:
    def mv(target: str, version: str, champion: bool) -> ModelVersion:
        return ModelVersion(
            name=MODEL_NAME,
            target=target,
            version=version,
            kind="lightgbm",
            params={},
            feature_names=["ret_1d"],
            metrics={"n_oos": 400, "accuracy": 0.47, "brier": 0.61, "log_loss": 1.03},
            artifact="{}",
            trained_through=date(2026, 10, 30),
            train_rows=1200,
            is_champion=champion,
            promoted_at=datetime.now(UTC) if champion else None,
        )

    db.add_all([mv("NIFTY 50", "lgbm-old", False), mv("NIFTY 50", "lgbm-new", True)])
    db.commit()
    session = _record_call(db, "NIFTY 50").session_date

    with captured_sql(db) as sql:
        body = logged_in.get("/api/health").json()
    assert body["models"] == [
        {
            "name": MODEL_NAME,
            "target": "NIFTY 50",
            "version": "lgbm-new",
            "trained_through": "2026-10-30",
            "oos_brier": 0.61,
            "climatology_brier": None,  # stored without baselines: nothing is made up
            "oos_accuracy": 0.47,
            "n_oos": 400,
            "promoted_at": body["models"][0]["promoted_at"],
        }
    ]
    assert body["models"][0]["promoted_at"] is not None
    assert body["models_note"].startswith("1 champion model in production")
    selects = [s for s in sql if s.lstrip().upper().startswith("SELECT")]
    assert selects and all("artifact" not in s for s in selects)
    last = next(f for f in body["freshness"] if f["dataset"] == "Last market call")
    assert last["status"] == "ok" and last["latest"] == session.isoformat()
    assert last["expected"] == session.isoformat()
    assert "1 ledger record;" in last["detail"]

    bank = ModelVersion(
        name=MODEL_NAME,
        target="NIFTY BANK",
        version="lgbm-bank",
        kind="lightgbm",
        params={},
        feature_names=["ret_1d"],
        metrics=OOS_METRICS,
        artifact="{}",
        trained_through=date(2026, 10, 30),
        train_rows=1200,
        is_champion=True,
        promoted_at=datetime.now(UTC),
    )
    db.add(bank)
    db.commit()
    models = logged_in.get("/api/health").json()["models"]
    assert [m["target"] for m in models] == ["NIFTY 50", "NIFTY BANK"]
    assert models[1]["oos_brier"] == 0.701 and models[1]["climatology_brier"] == 0.668


def test_last_market_call_freshness_is_session_aware() -> None:
    """A call is due by the 09:15 IST lock of its session; before that the previous one counts."""
    cal = TradingCalendar({}, {2026})
    thu, fri, sat = date(2026, 10, 29), date(2026, 10, 30), date(2026, 10, 31)
    mon, tue = date(2026, 11, 2), date(2026, 11, 3)
    lock = time(9, 15)

    def at(d: date, hh: int, mm: int) -> datetime:
        return datetime.combine(d, time(hh, mm), tzinfo=IST)

    # Monday 08:00 with Friday's call: fine, Monday's call is not due before the lock.
    f = market_call_freshness(cal, at(mon, 8, 0), fri, 1, lock)
    assert f.status == "ok" and f.latest == fri and f.expected == mon
    assert f.detail == (
        "1 ledger record; latest call is for 2026-10-30; the next is due for 2026-11-02 "
        "by 09:15 IST"
    )
    # Monday 09:30 with only Friday's call: stale.
    f = market_call_freshness(cal, at(mon, 9, 30), fri, 1, lock)
    assert f.status == "stale" and f.expected == mon
    assert "the call for 2026-11-02 was due by 09:15 IST and is missing" in (f.detail or "")
    # Monday 10:00 with Monday's call: ok.
    f = market_call_freshness(cal, at(mon, 10, 0), mon, 2, lock)
    assert f.status == "ok" and f.detail == "2 ledger records; the call for 2026-11-02 is recorded"
    # Monday 16:00 with Monday's call: ok; the page is heading into Tuesday.
    f = market_call_freshness(cal, at(mon, 16, 0), mon, 2, lock)
    assert f.status == "ok" and f.expected == tue
    # Saturday with Friday's call: ok (Monday's call is not due yet) ...
    assert market_call_freshness(cal, at(sat, 10, 0), fri, 1, lock).status == "ok"
    # ... but with only Thursday's call, Friday's was due: stale.
    assert market_call_freshness(cal, at(sat, 10, 0), thu, 1, lock).status == "stale"
    # Empty ledger.
    f = market_call_freshness(cal, at(mon, 8, 0), None, 0, lock)
    assert f.status == "missing" and f.latest is None and f.expected == mon


# ----------------------------------------------------------------- ideas on Today


def test_today_ideas_say_there_is_no_model_before_training(logged_in: TestClient) -> None:
    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["n_ideas"] == 0 and ideas["is_previous"] is False
    assert [g["horizon"] for g in ideas["by_horizon"]] == ["swing", "positional"]
    for g in ideas["by_horizon"]:
        assert g["ideas"] == [] and g["no_edge"] is False and g["champion_version"] is None
        assert "No ideas model" in g["note"] and "train-ideas" in g["note"]
    assert (
        ideas["by_horizon"][0]["horizon_days"] == 10
        and ideas["by_horizon"][1]["horizon_days"] == 60
    )
    assert "train-ideas" in ideas["note"]
    assert ideas["surveillance_note"] == SURVEILLANCE_NOTE and "not checked" in SURVEILLANCE_NOTE
    assert ideas["min_p_target"] == 0.35 and ideas["min_ev_pct"] == 0.5
    assert ideas["max_per_horizon"] == 3


def test_today_ideas_say_no_edge_only_when_the_engine_ran(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    db.add_all([_ideas_champion("swing"), _ideas_champion("positional")])
    db.commit()
    # A model, but the 07:30 job has not run for the session: not "No edge".
    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["session"] == session.isoformat()
    for g in ideas["by_horizon"]:
        assert g["no_edge"] is False and "hasn't run" in g["note"]
        assert g["champion_version"] == f"lgbm-ideas-{g['horizon']}"
    assert "hasn't run" in ideas["note"] and "No edge" not in ideas["note"]
    # The job recorded the session's market call (ideas come right after): no idea = no edge.
    _record_call(db, "NIFTY 50", session=session)
    ideas = logged_in.get("/api/today").json()["ideas"]
    for g in ideas["by_horizon"]:
        assert g["no_edge"] is True
        assert g["note"].startswith(f"No edge today ({g['horizon']})")
        assert "P(target) >= 35%" in g["note"] and "EV >= 0.5%" in g["note"]
    assert ideas["note"].startswith("No edge today")


def test_today_prefers_no_edge_over_a_previous_sessions_ideas(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    previous = session - timedelta(days=7)
    db.add_all([_ideas_champion("swing"), _ideas_champion("positional")])
    db.commit()
    _record_call(db, "NIFTY 50", session=previous)
    _record_idea(db, "RELIANCE", "swing", previous, rank=1)
    # The 07:30 run has not happened for this session: the last session's ideas, marked.
    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["is_previous"] is True and ideas["session"] == previous.isoformat()
    assert ideas["n_ideas"] == 1
    # It has run (the call is recorded) and found nothing: "No edge today", never a stale list.
    _record_call(db, "NIFTY 50", session=session)
    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["is_previous"] is False and ideas["session"] == session.isoformat()
    assert ideas["n_ideas"] == 0 and all(g["no_edge"] is True for g in ideas["by_horizon"])
    assert ideas["note"].startswith("No edge today")


def test_today_shows_the_sessions_ideas_with_track_record_and_surveillance(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    db.add(_ideas_champion("swing"))
    db.commit()
    _record_call(db, "NIFTY 50", session=session)
    swing = _record_idea(db, "RELIANCE", "swing", session, rank=1)
    _record_idea(
        db,
        "TCS",
        "swing",
        session,
        rank=2,
        p_target=0.58,
        setup="pullback",
        strategy="pullback_20dma",
    )

    with captured_sql(db) as sql:
        body = logged_in.get("/api/today").json()
    ideas = body["ideas"]
    assert ideas["session"] == session.isoformat() and ideas["is_previous"] is False
    assert ideas["n_ideas"] == 2 and ideas["note"] == ""
    sw, po = ideas["by_horizon"]
    assert [i["symbol"] for i in sw["ideas"]] == ["RELIANCE", "TCS"]  # by rank
    assert sw["no_edge"] is False and sw["note"] is None
    first = sw["ideas"][0]
    assert first["id"] == swing.id and first["seq"] == 1 and first["is_previous"] is False
    assert first["horizon"] == "swing" and first["horizon_days"] == 10
    assert first["setup"] == "breakout" and first["setup_label"] == "Breakout"
    assert first["strategy_label"] == "20-day breakout"
    assert first["entry_low"] == 990.0 and first["entry_high"] == 1010.0
    assert first["stop"] == 980.0 and first["target_1"] == 1040.0 and first["target_2"] == 1060.0
    assert first["p_target"] == 0.6 and first["position_qty"] == 50
    assert first["expected_return_low_pct"] == pytest.approx(4.0)
    assert first["expected_return_high_pct"] == pytest.approx(6.0)
    assert first["risk_reward"] == pytest.approx(2.0)
    assert first["ev_pct_after_costs"] == pytest.approx(
        0.6 * 4.0 - 0.4 * 2.0 - CostModel.from_settings().round_trip_pct()
    )
    assert first["is_late"] is False and first["direction"] == "long"
    assert first["surveillance_unknown"] is True
    assert first["surveillance_note"] == SURVEILLANCE_NOTE
    assert first["model_version"] is None and first["corrected_by_seq"] is None
    assert first["outcome"]["status"] == "pending" and first["outcome"]["entry_price"] is None
    track = first["track"]
    assert track == {
        "horizon": "swing",
        "setup": "breakout",
        "n_graded": 0,
        "hit_rate": None,
        "win_rate": None,
        "expectancy_pct": None,
        "sample_warning": True,
        "min_sample_size": 100,
    }
    assert sw["ideas"][1]["track"]["setup"] == "pullback"
    # The positional horizon has no idea and no model: the note says so, not "No edge".
    assert po["ideas"] == [] and po["no_edge"] is False and "No ideas model" in po["note"]
    # The list never reads the ideas' feature snapshot.
    selects = [q for q in sql if q.lstrip().upper().startswith("SELECT")]
    assert selects and all("feature_snapshot" not in q for q in selects)


def test_today_ideas_fall_back_to_the_previous_session_with_outcomes(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    previous = cal.previous_trading_day(session)
    idea = _record_idea(db, "RELIANCE", "swing", previous)
    outcome = grade_idea(
        db, idea, _bars(previous, (0, 1000.0, 1045.0, 998.0, 1030.0)), now=IDEA_NOW, source="test"
    )
    db.commit()
    assert outcome.status == "target"

    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["session"] == previous.isoformat() and ideas["is_previous"] is True
    [shown] = ideas["by_horizon"][0]["ideas"]
    assert shown["seq"] == idea.seq and shown["is_previous"] is True
    out = shown["outcome"]
    assert (
        out["status"] == "target" and out["entry_price"] == 1000.0 and out["exit_price"] == 1040.0
    )
    assert out["gross_return_pct"] == pytest.approx(4.0)
    assert out["net_return_pct"] == pytest.approx(4.0 - CostModel.from_settings().round_trip_pct())
    assert out["days_held"] == 1 and out["exit_date"] == previous.isoformat()
    # The track record beside it has moved: one graded breakout at the swing horizon.
    assert shown["track"]["n_graded"] == 1 and shown["track"]["hit_rate"] == 1.0
    assert shown["track"]["sample_warning"] is True
    # Once the upcoming session has an idea, that one is shown instead.
    _record_idea(db, "TCS", "swing", session)
    ideas = logged_in.get("/api/today").json()["ideas"]
    assert ideas["session"] == session.isoformat() and ideas["is_previous"] is False
    assert [i["symbol"] for i in ideas["by_horizon"][0]["ideas"]] == ["TCS"]


# ----------------------------------------------------------------- /api/ideas


def test_ideas_list_filters(db: Session, logged_in: TestClient) -> None:
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    previous = cal.previous_trading_day(session)
    pending = _record_idea(db, "RELIANCE", "swing", session)
    hit = _record_idea(db, "TCS", "swing", previous, setup="pullback", strategy="pullback_20dma")
    grade_idea(
        db, hit, _bars(previous, (0, 1000.0, 1045.0, 998.0, 1030.0)), now=IDEA_NOW, source="t"
    )
    gapped = _record_idea(db, "INFY", "positional", previous)
    open_check_fill(db, gapped, 1020.0, now=IDEA_NOW, source="t")  # above the zone: invalidated
    original = _record_idea(db, "HDFCBANK", "positional", session, p_target=0.57)
    fix = _record_idea(db, "HDFCBANK", "positional", session, p_target=0.61, corrects=original.id)
    db.commit()

    with captured_sql(db) as sql:
        body = logged_in.get("/api/ideas").json()
    assert body["n"] == 4 and body["status"] is None and body["horizon"] is None
    assert body["limit"] == 50 and body["min_sample_size"] == 100
    assert body["surveillance_note"] == SURVEILLANCE_NOTE
    assert body["statuses"] == list(STATUS_FILTERS) and body["horizons"] == ["swing", "positional"]
    assert body["setups"] == ["breakout", "momentum", "mean_reversion", "pullback"]
    # Newest session first, the correction instead of the superseded original.
    assert [i["seq"] for i in body["ideas"]] == [fix.seq, pending.seq, gapped.seq, hit.seq]
    by_seq = {i["seq"]: i for i in body["ideas"]}
    assert (
        by_seq[fix.seq]["corrects_idea_id"] == original.id and by_seq[fix.seq]["p_target"] == 0.61
    )
    assert by_seq[hit.seq]["outcome"]["status"] == "target"
    assert by_seq[gapped.seq]["outcome"]["status"] == "invalidated"
    assert "gapped above the entry zone" in by_seq[gapped.seq]["outcome"]["notes"]
    assert by_seq[pending.seq]["outcome"]["status"] == "pending"
    selects = [q for q in sql if q.lstrip().upper().startswith("SELECT")]
    assert selects and all("feature_snapshot" not in q and "drivers" not in q for q in selects)

    def seqs(query: str) -> list[int]:
        resp = logged_in.get(f"/api/ideas?{query}")
        assert resp.status_code == 200, resp.text
        return [i["seq"] for i in resp.json()["ideas"]]

    assert seqs("status=pending") == [fix.seq, pending.seq]
    assert seqs("status=active") == [fix.seq, pending.seq]
    assert seqs("status=target") == [hit.seq]
    assert seqs("status=graded") == [hit.seq]
    assert seqs("status=invalidated") == [gapped.seq]
    assert seqs("horizon=positional") == [fix.seq, gapped.seq]
    assert seqs("horizon=SWING&status=pending") == [pending.seq]
    assert seqs("setup=pullback") == [hit.seq]
    assert seqs("setup=results_play") == []
    assert seqs("limit=1") == [fix.seq]
    assert logged_in.get("/api/ideas?limit=0").status_code == 422
    assert logged_in.get("/api/ideas?status=bogus").status_code == 422
    resp = logged_in.get("/api/ideas?horizon=weekly")
    assert resp.status_code == 404
    assert "Unknown horizon 'weekly'" in resp.json()["detail"]
    assert "swing, positional" in resp.json()["detail"]
    assert logged_in.get("/api/scorecard/ideas?horizon=weekly").status_code == 404


def test_idea_detail_has_the_full_reasoning_factors_outcome_and_paper_trade(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    mv = _ideas_champion("swing")
    db.add(mv)
    db.commit()
    champ = mv.id
    idea = _record_idea(db, "RELIANCE", "swing", session, model_version_id=champ)
    outcome = open_check_fill(db, idea, 1000.0, now=IDEA_NOW, source="open_check")
    trade = paper.open_trade_for_idea(db, idea, outcome, now=IDEA_NOW)
    db.commit()

    body = logged_in.get(f"/api/ideas/{idea.seq}").json()
    assert body["seq"] == idea.seq and body["symbol"] == "RELIANCE"
    assert body["model_version"] == "lgbm-ideas-swing" and body["model_version_id"] == champ
    assert body["hash"] == idea.hash and body["prev_hash"] == "0" * 64
    assert len(body["reasons"]) == 3 and body["reasons"][0].startswith("Close 1000.00 above")
    assert body["invalidation"].startswith("A close below 980.00")
    assert body["surveillance_unknown"] is True and body["surveillance_note"] == SURVEILLANCE_NOTE
    # Outcome and paper trade.
    assert body["outcome"]["status"] == "open" and body["outcome"]["entry_price"] == 1000.0
    assert body["outcome"]["entry_date"] == session.isoformat()
    assert body["outcome"]["source"] == "open_check" and body["outcome"]["exit_price"] is None
    pt = body["paper_trade"]
    assert pt["id"] == trade.id and pt["idea_id"] == idea.id and pt["status"] == "open"
    assert pt["qty"] == 50 and pt["entry_price"] == 1000.0 and pt["entry_value"] == 50_000.0
    assert pt["stop"] == 980.0 and pt["target"] == 1040.0 and pt["net_pnl"] is None
    # Factors: the recorded drivers first (largest |contribution| first), then every other
    # feature of the snapshot with no contribution (none is invented).
    f = body["factors"]
    assert f["drivers_top_n"] == 8
    assert [d["feature"] for d in f["drivers"]] == ["brk_20d_high_pct", "vol_z20"]
    assert f["drivers"][0] == {
        "feature": "brk_20d_high_pct",
        "value": 1.2,
        "contribution": 0.08,
        "text": "Breakout above the 20-day high by 1.2 %",
    }
    assert [x["feature"] for x in f["all_factors"]] == [
        "brk_20d_high_pct",
        "vol_z20",
        "ret_20d",
        "rsi14",
        "atr14_pct",
        "nan",
        "missing",
    ]
    assert f["n_features"] == 7 and f["n_with_contribution"] == 2
    by_name = {x["feature"]: x for x in f["all_factors"]}
    assert by_name["vol_z20"]["contribution"] == 0.05 and by_name["vol_z20"]["value"] == 1.4
    assert by_name["vol_z20"]["label"] == "Volume z-score (20 days)"
    assert by_name["ret_20d"] == {
        "feature": "ret_20d",
        "label": "20-day return (%)",
        "value": 0.3,
        "contribution": None,
        "text": None,
    }
    assert by_name["rsi14"]["value"] == pytest.approx(61.2345678901)  # finite(): 10 decimals
    assert by_name["nan"]["value"] is None and by_name["missing"]["value"] is None
    assert by_name["brk_20d_high_pct"]["label"]  # an unknown name still gets a label
    snap = f["feature_snapshot"]
    assert set(snap) == set(SNAPSHOT) and snap["nan"] is None and snap["ret_20d"] == 0.3
    assert logged_in.get("/api/ideas/999").status_code == 404


def test_idea_detail_of_a_superseded_idea_points_at_its_correction(
    db: Session, logged_in: TestClient
) -> None:
    original = _record_idea(db, "RELIANCE", "swing", p_target=0.57)
    fix = _record_idea(db, "RELIANCE", "swing", p_target=0.62, corrects=original.id)
    old = logged_in.get(f"/api/ideas/{original.seq}").json()
    assert old["corrected_by_seq"] == fix.seq and old["corrects_idea_id"] is None
    new = logged_in.get(f"/api/ideas/{fix.seq}").json()
    assert new["corrected_by_seq"] is None and new["corrects_idea_id"] == original.id
    assert new["paper_trade"] is None and new["factors"]["n_with_contribution"] == 2
    assert [i["seq"] for i in logged_in.get("/api/ideas").json()["ideas"]] == [fix.seq]


# ----------------------------------------------------------------- /api/paper


def test_paper_endpoint_is_honest_when_empty(logged_in: TestClient) -> None:
    body = logged_in.get("/api/paper").json()
    assert body["capital"] == 1_000_000.0 and body["n_trades"] == 0 and body["n_open"] == 0
    assert body["n_days"] == 0 and body["n_returns"] == 0
    for key in (
        "hit_rate",
        "avg_win",
        "avg_loss",
        "avg_win_pct",
        "avg_loss_pct",
        "expectancy",
        "expectancy_pct",
        "profit_factor",
        "total_net_pnl",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "total_return_pct",
        "nifty_return_pct",
    ):
        assert body[key] is None, key
    assert body["equity_curve"] == [] and body["open_trades"] == [] and body["recent_closed"] == []
    assert body["sample_warning"] is True and body["min_sample_size"] == 100
    assert "price index" in body["note"] and "Paper" in body["note"]
    assert body["risk_per_trade_pct"] == 1.0 and body["max_position_pct"] == 10.0
    assert body["cost_round_trip_pct"] == pytest.approx(0.494142)
    assert logged_in.get("/api/paper/trades").json() == []
    assert logged_in.get("/api/paper/trades?status=bogus").status_code == 422


def test_paper_endpoint_with_a_closed_trade_and_equity_curve(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    idea = _record_idea(db, "RELIANCE", "swing", session)
    filled = open_check_fill(db, idea, 1000.0, now=IDEA_NOW, source="t")
    paper.sync_trade_with_outcome(db, idea, filled, now=IDEA_NOW)
    paper.snapshot_equity(db, session, {"RELIANCE": 1010.0}, 25_000.0)
    db.commit()
    body = logged_in.get("/api/paper").json()
    assert body["n_open"] == 1 and body["n_trades"] == 0 and body["n_days"] == 1
    assert body["open_trades"][0]["symbol"] == "RELIANCE" and body["open_trades"][0]["qty"] == 50
    assert body["equity_curve"][0] == {
        "trade_date": session.isoformat(),
        "equity": 1_000_500.0,  # 50 shares marked from 1000 to 1010
        "cash": 950_000.0,
        "invested": 50_500.0,
        "open_positions": 1,
        "nifty_close": 25_000.0,
        "nifty_rebased": 1_000_000.0,
    }
    assert body["total_return_pct"] == pytest.approx(0.05)
    assert body["sharpe"] is None and body["max_drawdown_pct"] == 0.0  # one snapshot: no returns

    graded = grade_idea(
        db, idea, _bars(session, (0, 1000.0, 1045.0, 998.0, 1030.0)), now=IDEA_NOW, source="t"
    )
    paper.sync_trade_with_outcome(db, idea, graded, now=IDEA_NOW)
    next_day = session + timedelta(days=1)
    paper.snapshot_equity(db, next_day, {}, 25_250.0)
    db.commit()
    body = logged_in.get("/api/paper").json()
    cost = 50 * 1000.0 * CostModel.from_settings().round_trip_pct() / 100.0
    assert body["n_trades"] == 1 and body["n_wins"] == 1 and body["n_losses"] == 0
    assert body["hit_rate"] == 1.0 and body["profit_factor"] is None  # no losing trade yet
    assert body["total_net_pnl"] == pytest.approx(50 * 40.0 - cost)
    assert body["expectancy"] == pytest.approx(50 * 40.0 - cost)
    assert body["avg_win_pct"] == pytest.approx(4.0 - CostModel.from_settings().round_trip_pct())
    # Curve statistics start from the paper capital: two snapshots give two daily returns.
    assert body["n_days"] == 2 and body["n_returns"] == 2 and "sharpe" in body
    assert body["levered_days"] == 0 and body["min_cash"] is not None
    assert "max_exposure_pct" in body
    assert body["nifty_return_pct"] == pytest.approx(1.0)
    assert body["equity_curve"][1]["nifty_rebased"] == pytest.approx(1_010_000.0)
    assert body["open_trades"] == [] and len(body["recent_closed"]) == 1
    closed = body["recent_closed"][0]
    assert closed["status"] == "closed" and closed["exit_price"] == 1040.0
    assert closed["net_pnl"] == pytest.approx(50 * 40.0 - cost)
    assert closed["r_multiple"] == pytest.approx((50 * 40.0 - cost) / (50 * 20.0))
    assert closed["net_return_pct"] == pytest.approx(
        4.0 - CostModel.from_settings().round_trip_pct()
    )
    assert body["sample_warning"] is True
    trades = logged_in.get("/api/paper/trades").json()
    assert [t["status"] for t in trades] == ["closed"]
    assert logged_in.get("/api/paper/trades?status=open").json() == []
    assert len(logged_in.get("/api/paper/trades?status=closed").json()) == 1


# ----------------------------------------------------------------- /api/scorecard/ideas


def test_ideas_scorecard_endpoint_shape_when_empty(logged_in: TestClient) -> None:
    body = logged_in.get("/api/scorecard/ideas").json()
    assert body["horizon"] is None and body["horizons"] == ["swing", "positional"]
    assert body["n_ideas"] == 0 and body["n_graded"] == 0
    assert body["by_status"] == dict.fromkeys(IDEA_STATUSES, 0)
    for key in (
        "hit_rate",
        "win_rate",
        "avg_win_pct",
        "avg_loss_pct",
        "expectancy_pct",
        "profit_factor",
        "avg_r_multiple",
        "avg_days_held",
    ):
        assert body[key] is None, key
    assert body["by_setup"] == [] and body["calibration"] == [] and body["recent"] == []
    assert [r["horizon"] for r in body["by_horizon"]] == ["swing", "positional"]
    assert all(r["n"] == 0 and r["hit_rate"] is None for r in body["by_horizon"])
    assert body["champions"] == {"swing": None, "positional": None}
    assert body["sample_warning"] is True and body["min_sample_size"] == 100
    assert "unproven" in body["verdict"]
    one = logged_in.get("/api/scorecard/ideas?horizon=Positional").json()
    assert one["horizon"] == "positional" and one["champions"] == {"positional": None}
    assert [r["horizon"] for r in one["by_horizon"]] == ["positional"]


def test_ideas_scorecard_endpoint_aggregates_graded_ideas_and_the_champion(
    db: Session, logged_in: TestClient
) -> None:
    session = _current_session(db)
    cal = TradingCalendar.load(db)
    d1 = cal.previous_trading_day(session)
    d0 = cal.previous_trading_day(d1)
    hit = _record_idea(db, "RELIANCE", "swing", d0)
    grade_idea(db, hit, _bars(d0, (0, 1000.0, 1045.0, 998.0, 1030.0)), now=IDEA_NOW, source="t")
    miss = _record_idea(
        db, "TCS", "swing", d1, p_target=0.56, setup="pullback", strategy="pullback_20dma"
    )
    grade_idea(db, miss, _bars(d1, (0, 1000.0, 1005.0, 975.0, 990.0)), now=IDEA_NOW, source="t")
    _record_idea(db, "INFY", "positional", session)  # pending
    db.add(_ideas_champion("swing", IDEA_OOS_METRICS))
    db.commit()

    with captured_sql(db) as sql:
        body = logged_in.get("/api/scorecard/ideas").json()
    assert body["n_ideas"] == 3 and body["n_graded"] == 2
    assert body["by_status"]["target"] == 1 and body["by_status"]["stop"] == 1
    assert body["by_status"]["pending"] == 1
    assert body["hit_rate"] == 0.5 and body["win_rate"] == 0.5
    cost = CostModel.from_settings().round_trip_pct()
    assert body["avg_win_pct"] == pytest.approx(4.0 - cost)
    assert body["avg_loss_pct"] == pytest.approx(-2.0 - cost)
    assert body["expectancy_pct"] == pytest.approx((4.0 - cost - 2.0 - cost) / 2)
    assert body["profit_factor"] == pytest.approx((4.0 - cost) / (2.0 + cost))
    assert body["avg_days_held"] == 1.0
    setups = {r["setup"]: r for r in body["by_setup"]}
    assert setups["breakout"]["n"] == 1 and setups["breakout"]["hit_rate"] == 1.0
    assert setups["pullback"]["n"] == 1 and setups["pullback"]["hit_rate"] == 0.0
    by_h = {r["horizon"]: r for r in body["by_horizon"]}
    assert by_h["swing"]["n"] == 2 and by_h["positional"]["n"] == 0
    assert sum(b["n"] for b in body["calibration"]) == 2
    for b in body["calibration"]:
        assert set(b) == {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}
    assert "After 2 graded ideas" in body["verdict"] and "50%" in body["verdict"]
    assert body["sample_warning"] is True
    # Recent rows (newest first) mirror ideas_scorecard.row_dict.
    recent = body["recent"]
    assert [r["symbol"] for r in recent] == ["INFY", "TCS", "RELIANCE"]
    assert recent[0]["status"] == "pending" and recent[0]["net_return_pct"] is None
    assert recent[2]["status"] == "target" and recent[2]["exit_price"] == 1040.0
    assert recent[2]["surveillance_unknown"] is True
    # The champion box per horizon, from the stored metrics only (never the artifact).
    champ = body["champions"]["swing"]
    assert body["champions"]["positional"] is None
    assert champ["name"] == "ideas_swing" and champ["horizon"] == "swing"
    assert champ["version"] == "lgbm-ideas-swing" and champ["train_rows"] == 5000
    assert champ["trained_through"] == "2026-10-30" and champ["promoted_at"] is not None
    assert champ["oos"]["hit_rate"] == 0.58 and champ["oos"]["n_selected"] == 410
    assert champ["oos"]["expectancy_after_costs_pct"] == 0.82 and champ["oos"]["n_oos"] == 3200
    assert champ["oos"]["n_sessions"] is None  # not stored: not invented
    assert [b["name"] for b in champ["baselines"]] == ["all_candidates", "random_universe"]
    assert champ["baselines"][0]["hit_rate"] == 0.41 and champ["baselines"][1]["brier"] is None
    assert champ["per_strategy"] == {"breakout_20d": {"n": 900, "hit_rate": 0.44}}
    assert champ["calibration"][0]["n"] == 300
    assert champ["class_frequencies"] == {"target": 0.41, "stop": 0.47, "expired": 0.12}
    assert champ["notes"] == IDEA_OOS_METRICS["notes"]
    assert champ["oos_verdict"].startswith("Out of sample 410 ideas cleared the bar")
    selects = [q for q in sql if q.lstrip().upper().startswith("SELECT")]
    assert selects and all("artifact" not in q for q in selects)
    assert all("feature_snapshot" not in q and "drivers" not in q for q in selects)
    swing = logged_in.get("/api/scorecard/ideas?horizon=swing").json()
    assert swing["horizon"] == "swing" and swing["n_ideas"] == 2 and swing["n_graded"] == 2
    assert list(swing["champions"]) == ["swing"]


def test_ideas_scorecard_champion_without_an_oos_hit_rate_says_so(
    db: Session, logged_in: TestClient
) -> None:
    metrics = dict(IDEA_OOS_METRICS)
    metrics["hit_rate"] = None  # no candidate cleared the bar out of sample (NaN -> null)
    db.add(_ideas_champion("positional", metrics))
    db.commit()
    champ = logged_in.get("/api/scorecard/ideas").json()["champions"]["positional"]
    assert champ["oos"]["hit_rate"] is None and champ["oos"]["brier"] == 0.231
    assert champ["oos_verdict"] == "No out-of-sample hit rate was stored with this model."


# ----------------------------------------------------------------- factors, ledger chains


def test_market_call_factors_endpoint(db: Session, logged_in: TestClient) -> None:
    row = _record_call(db, "NIFTY 50")
    body = logged_in.get(f"/api/today/market-call/{row.id}/factors").json()
    assert body["drivers_top_n"] == get_settings().drivers_top_n == 8
    assert [d["feature"] for d in body["drivers"]] == ["es_fut_ret_1d", "vix_level"]
    assert [f["feature"] for f in body["all_factors"]] == ["es_fut_ret_1d", "vix_level", "ret_1d"]
    first, vix, ret = body["all_factors"]
    assert first["contribution"] == 0.12 and first["value"] == 0.8
    assert first["label"] == "S&P 500 futures 1-day return (%)"
    assert first["text"] == "S&P 500 futures 1-day return at +0.80%"
    assert vix["value"] is None and vix["contribution"] == -0.02  # missing stays missing
    assert ret == {
        "feature": "ret_1d",
        "label": "1-day return (%)",
        "value": 0.3,
        "contribution": None,
        "text": None,
    }
    assert body["feature_snapshot"] == {"ret_1d": 0.3, "vix_level": None}
    assert body["n_features"] == 3 and body["n_with_contribution"] == 2
    assert logged_in.get("/api/today/market-call/999/factors").status_code == 404


def test_verify_ledger_endpoint_walks_the_idea_chain_too(
    db: Session, logged_in: TestClient
) -> None:
    body = logged_in.post("/api/scorecard/verify-ledger?chain=idea").json()
    assert body == {"ok": True, "checked": 0, "first_bad_seq": None, "message": "ledger is empty"}
    _record_idea(db, "RELIANCE", "swing")
    _record_idea(db, "TCS", "positional")
    body = logged_in.post("/api/scorecard/verify-ledger?chain=idea").json()
    assert body["ok"] is True and body["checked"] == 2 and body["message"] == "2 records verified"
    # The default is still the market-call chain, which is empty here.
    assert logged_in.post("/api/scorecard/verify-ledger").json()["checked"] == 0
    assert logged_in.post("/api/scorecard/verify-ledger?chain=bogus").status_code == 404


# ----------------------------------------------------------------- web setup


TOKEN = "s3tup-token-with-enough-length"


def test_setup_disabled_without_token(client: TestClient) -> None:
    assert client.get("/api/auth/setup-status").json() == {
        "needs_setup": True,
        "setup_enabled": False,
    }
    resp = client.post("/api/auth/setup", json={"token": "x", "username": "me", "password": PW})
    assert resp.status_code == 403


def test_setup_creates_single_user_once(
    db: Session, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "setup_token", TOKEN)
    bad = client.post("/api/auth/setup", json={"token": "wrong", "username": "me", "password": PW})
    assert bad.status_code == 403
    ok = client.post("/api/auth/setup", json={"token": TOKEN, "username": "me", "password": PW})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["qr_svg_data_url"].startswith("data:image/svg+xml;base64,")
    assert body["provisioning_uri"].startswith("otpauth://totp/Dalal:me")
    # The new login works with the returned secret.
    login = client.post(
        "/api/auth/login",
        json={"username": "me", "password": PW, "totp": pyotp.TOTP(body["totp_secret"]).now()},
    )
    assert login.status_code == 200
    # Setup can never run again once a user exists.
    again = client.post("/api/auth/setup", json={"token": TOKEN, "username": "x", "password": PW})
    assert again.status_code == 409
    assert client.get("/api/auth/setup-status").json()["needs_setup"] is False


def test_setup_attempts_are_rate_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "setup_token", TOKEN)
    codes = [
        client.post(
            "/api/auth/setup", json={"token": "nope", "username": "me", "password": PW}
        ).status_code
        for _ in range(11)
    ]
    assert codes[:10] == [403] * 10 and codes[10] == 429
    # Strangers exhausting the limit must not lock out the owner, who has the token.
    owner = client.post("/api/auth/setup", json={"token": TOKEN, "username": "me", "password": PW})
    assert owner.status_code == 200, owner.text


def test_health_external_scheduler_mode(
    db: Session, logged_in: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings
    from app.models import JobRun

    monkeypatch.setattr(get_settings(), "scheduler_mode", "external")
    body = logged_in.get("/api/health").json()
    assert body["scheduler_mode"] == "external" and body["scheduler_alive"] is False
    db.add(JobRun(job_name="global_cues_refresh", run_date=date.today(), status="success"))
    db.commit()
    body = logged_in.get("/api/health").json()
    assert body["scheduler_alive"] is True
    cues = next(j for j in body["jobs"] if j["name"] == "global_cues_refresh")
    assert "GitHub Actions" in cues["schedule"]


def test_full_login_flow_on_postgres_kv(
    db: Session, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sessions, TOTP replay and lockout work with no Redis at all (the Vercel setup)."""
    from app.cache import get_kv
    from app.config import get_settings
    from app.kv import PostgresKV

    monkeypatch.setattr(get_settings(), "cache_backend", "postgres")
    get_kv.cache_clear()
    try:
        assert isinstance(get_kv(), PostgresKV)
        new = create_user(db, "me", PW)
        db.commit()
        code = pyotp.TOTP(new.totp_secret).now()
        ok = client.post("/api/auth/login", json={"username": "me", "password": PW, "totp": code})
        assert ok.status_code == 200
        assert client.get("/api/auth/me").json() == {"username": "me"}
        client.cookies.clear()
        replay = client.post(
            "/api/auth/login", json={"username": "me", "password": PW, "totp": code}
        )
        assert replay.status_code == 401, "a TOTP code works once"
        for _ in range(5):
            client.post(
                "/api/auth/login", json={"username": "me", "password": "x", "totp": "000000"}
            )
        locked = client.post(
            "/api/auth/login", json={"username": "me", "password": PW, "totp": code}
        )
        assert locked.status_code == 429, "failure counter survives the failing requests"
    finally:
        get_kv.cache_clear()

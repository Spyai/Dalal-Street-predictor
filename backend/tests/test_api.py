"""API: auth gate, login, Today (incl. the market call), Scorecard, Stock page, Health.

No network: providers are faked. Ledger rows are seeded through
`app.services.ledger.record_market_call` with a minimal `MarketCallRecordIn`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.health import market_call_freshness
from app.cache import get_kv
from app.config import IST
from app.main import app
from app.ml.core import (
    BASELINE_NAMES,
    CONFIDENCE_LEVELS,
    MODEL_NAME,
    Driver,
    MarketCallOutput,
    MarketCallRecordIn,
    session_for,
)
from app.models import ModelVersion, Prediction
from app.providers.base import Bar
from app.security.auth import create_user
from app.services import instruments, ledger
from app.services.calendar import TradingCalendar
from app.services.flows import store as store_flows
from app.services.grading import grade_close
from app.services.market_data import upsert_daily_bars
from app.services.scorecard import MIN_PAIRED_N
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

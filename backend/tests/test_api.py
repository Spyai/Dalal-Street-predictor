"""API: auth gate, login flow, Today, Stock page, Health. No network: providers are faked."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.cache import get_redis
from app.main import app
from app.providers.base import Bar
from app.security.auth import create_user
from app.services import instruments
from app.services.flows import store as store_flows
from app.services.market_data import upsert_daily_bars

PW = "correct horse battery staple"


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
    ],
)
def test_everything_else_requires_login(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


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
    assert body["market_call"] is None
    assert "Phase 2" in body["market_call_note"]
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
    get_redis().set("stockfetch:NSE:TESTCO", "1", ex=60)
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
    assert names["premarket_prediction"]["built"] is False
    assert names["premarket_prediction"]["phase"] == 2
    assert names["evening_data_pull"]["built"] is True
    assert {s["source"] for s in body["sources"]} >= {"kite", "yfinance", "nse", "telegram"}
    assert body["scheduler_alive"] is False
    assert "Phase 2" in body["models_note"]
    datasets = {f["dataset"]: f for f in body["freshness"]}
    assert datasets["NSE holiday calendar"]["status"] == "missing"

"""Kite daily token: expiry, encrypted storage, account check."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.config import IST, Settings
from app.models import KiteSession
from app.services import kite_auth


def test_token_expiry_rolls_to_next_morning() -> None:
    login = datetime(2026, 10, 30, 7, 10, tzinfo=IST)
    assert kite_auth.token_expiry(login) == datetime(2026, 10, 31, 5, 55, tzinfo=IST)
    early = datetime(2026, 10, 30, 3, 0, tzinfo=IST)
    assert kite_auth.token_expiry(early) == datetime(2026, 10, 30, 5, 55, tzinfo=IST)


class FakeKite:
    def __init__(self, user_id: str = "AB1234") -> None:
        self.user_id = user_id

    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
        assert api_secret == "secret"
        return {
            "user_id": self.user_id,
            "access_token": "tok-" + request_token,
            "login_time": "2026-10-30 07:12:00",
        }


def _settings(**kw: Any) -> Settings:
    return Settings(kite_api_key="key", kite_api_secret="secret", **kw)


def test_complete_login_stores_encrypted_token(db: Session) -> None:
    sess = kite_auth.complete_login(db, "req1", _settings(), client=FakeKite())
    db.commit()
    row = db.get(KiteSession, sess.id)
    assert row is not None
    assert "tok-req1" not in row.access_token_enc
    assert row.login_time == datetime(2026, 10, 30, 7, 12, tzinfo=IST)
    assert row.expires_at == datetime(2026, 10, 31, 5, 55, tzinfo=IST)


def test_wrong_zerodha_account_rejected(db: Session) -> None:
    with pytest.raises(PermissionError):
        kite_auth.complete_login(db, "req", _settings(kite_user_id="ZZ9999"), client=FakeKite())


def test_active_session_respects_expiry(db: Session) -> None:
    kite_auth.complete_login(db, "req", _settings(), client=FakeKite())
    db.commit()
    during = datetime(2026, 10, 30, 12, 0, tzinfo=IST)
    after = datetime(2026, 10, 31, 6, 0, tzinfo=IST)
    assert kite_auth.active_session(db, during) is not None
    assert kite_auth.active_session(db, after) is None
    assert kite_auth.active_session(db, after - timedelta(minutes=10)) is not None


def test_not_configured() -> None:
    from app.providers.base import NotConfigured

    with pytest.raises(NotConfigured):
        kite_auth.login_url(Settings(kite_api_key="", kite_api_secret=""))

"""Kite Connect daily login: login URL, request-token exchange, encrypted token storage.

Flow: you open the login URL -> log in on kite.zerodha.com -> Kite redirects to
APP_BASE_URL/api/kite/callback?request_token=... -> we exchange it with the API secret
for an access token, encrypt it, and store it until ~06:00 IST the next day.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import IST, Settings, get_settings
from app.models import KiteSession
from app.providers.base import AuthExpired, NotConfigured
from app.security.crypto import decrypt, encrypt

# Zerodha: access tokens are invalidated at ~06:00 IST the next day. Treat 05:55 as the
# end so a job never starts with a token that dies mid-run.
TOKEN_ROLLOVER = time(5, 55)


def token_expiry(login_time: datetime) -> datetime:
    local = login_time.astimezone(IST)
    rollover_today = datetime.combine(local.date(), TOKEN_ROLLOVER, tzinfo=IST)
    expiry = rollover_today if local < rollover_today else rollover_today + timedelta(days=1)
    return expiry.astimezone(UTC)


def _new_client(settings: Settings, access_token: str | None = None) -> Any:
    from kiteconnect import KiteConnect

    return KiteConnect(api_key=settings.kite_api_key, access_token=access_token, timeout=15)


def login_url(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.kite_configured:
        raise NotConfigured("kite", "KITE_API_KEY / KITE_API_SECRET not set")
    return str(_new_client(settings).login_url())


def complete_login(
    db: Session, request_token: str, settings: Settings | None = None, client: Any = None
) -> KiteSession:
    settings = settings or get_settings()
    if not settings.kite_configured:
        raise NotConfigured("kite", "KITE_API_KEY / KITE_API_SECRET not set")
    kite = client or _new_client(settings)
    data = kite.generate_session(request_token, api_secret=settings.kite_api_secret)
    user_id = str(data.get("user_id", ""))
    if settings.kite_user_id and user_id.upper() != settings.kite_user_id.upper():
        raise PermissionError(f"Kite login was for {user_id}, expected {settings.kite_user_id}")
    login_time = _parse_login_time(data.get("login_time"))
    session = KiteSession(
        kite_user_id=user_id,
        access_token_enc=encrypt(str(data["access_token"])),
        login_time=login_time,
        expires_at=token_expiry(login_time),
    )
    db.add(session)
    db.flush()
    return session


def active_session(db: Session, now: datetime | None = None) -> KiteSession | None:
    now = now or datetime.now(UTC)
    return db.scalar(
        select(KiteSession)
        .where(KiteSession.expires_at > now)
        .order_by(KiteSession.login_time.desc())
        .limit(1)
    )


def authenticated_client(db: Session, settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    if not settings.kite_configured:
        raise NotConfigured("kite", "KITE_API_KEY / KITE_API_SECRET not set")
    sess = active_session(db)
    if sess is None:
        raise AuthExpired("kite", "no valid Kite session today", "Log in to Kite")
    return _new_client(settings, access_token=decrypt(sess.access_token_enc))


def _parse_login_time(value: Any) -> datetime:
    """Kite returns login_time as an IST wall-clock string or naive datetime."""
    if isinstance(value, str) and value:
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value.replace(tzinfo=IST) if value.tzinfo is None else value
    return datetime.now(UTC)

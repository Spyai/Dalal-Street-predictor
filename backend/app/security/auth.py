"""Single-user authentication: argon2 password + TOTP, Redis-backed sessions.

- Login needs password AND a current TOTP code.
- A TOTP time-step can be used once (replay protection).
- 5 failed attempts in 15 minutes locks login for 15 minutes.
- Sessions are random 256-bit tokens stored in Redis; the cookie holds only the token.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

import pyotp
import redis
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache import KV
from app.models import AppUser
from app.security.crypto import decrypt, encrypt

SESSION_COOKIE = "dalal_session"
MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60
TOTP_ISSUER = "Dalal"

_hasher = PasswordHasher()


class LoginLocked(Exception):
    pass


class InvalidCredentials(Exception):
    pass


@dataclass(frozen=True)
class NewUser:
    username: str
    totp_secret: str
    provisioning_uri: str


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def create_user(db: Session, username: str, password: str) -> NewUser:
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")
    if db.scalar(select(AppUser).limit(1)) is not None:
        raise ValueError("A user already exists. Dalal is single-user; use reset-user to replace.")
    secret = pyotp.random_base32()
    db.add(
        AppUser(
            username=username,
            password_hash=hash_password(password),
            totp_secret_enc=encrypt(secret),
        )
    )
    db.flush()
    uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=TOTP_ISSUER)
    return NewUser(username=username, totp_secret=secret, provisioning_uri=uri)


def _fail_key(username: str) -> str:
    return f"auth:fail:{username.lower()}"


def authenticate(db: Session, r: KV, username: str, password: str, totp_code: str) -> AppUser:
    fail_key = _fail_key(username)
    failures = int(r.get(fail_key) or 0)
    if failures >= MAX_FAILURES:
        raise LoginLocked("Too many failed attempts. Try again in 15 minutes.")

    user = db.scalar(select(AppUser).where(AppUser.username == username))
    ok = user is not None and verify_password(user.password_hash, password)
    if ok and user is not None:
        totp = pyotp.TOTP(decrypt(user.totp_secret_enc))
        code = totp_code.strip().replace(" ", "")
        ok = totp.verify(code, valid_window=1)
        if ok:
            # Replay protection: each code (time-step) is single-use.
            step = int(datetime.now(UTC).timestamp() // totp.interval)
            used_key = f"auth:totp_used:{user.id}:{code}"
            ok = bool(r.set(used_key, str(step), nx=True, ex=totp.interval * 3))

    if not ok or user is None:
        pipe = r.pipeline()
        pipe.incr(fail_key)
        pipe.expire(fail_key, LOCKOUT_SECONDS)
        pipe.execute()
        raise InvalidCredentials("Invalid username, password or code.")

    r.delete(fail_key)
    user.last_login_at = datetime.now(UTC)
    return user


def create_session(r: KV, user_id: int, ttl_hours: int) -> str:
    token = secrets.token_urlsafe(32)
    r.set(f"session:{token}", str(user_id), ex=ttl_hours * 3600)
    return token


def resolve_session(r: KV, token: str | None) -> int | None:
    if not token:
        return None
    try:
        raw = r.get(f"session:{token}")
    except redis.RedisError:
        return None
    return int(raw) if raw else None


def destroy_session(r: KV, token: str | None) -> None:
    if token:
        r.delete(f"session:{token}")

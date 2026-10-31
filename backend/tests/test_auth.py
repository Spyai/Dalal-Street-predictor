"""Password + TOTP login, replay protection, lockout; secret encryption."""

from __future__ import annotations

import pyotp
import pytest
from sqlalchemy.orm import Session

from app.cache import get_kv
from app.security import auth
from app.security.crypto import SecretKeyMissing, decrypt, encrypt

PW = "correct horse battery staple"


def test_encrypt_roundtrip_and_key_required() -> None:
    tok = encrypt("kite-access-token")
    assert tok != "kite-access-token"
    assert decrypt(tok) == "kite-access-token"
    with pytest.raises(SecretKeyMissing):
        encrypt("x", secret_key="short")


def test_password_hashing() -> None:
    h = auth.hash_password(PW)
    assert h.startswith("$argon2")
    assert auth.verify_password(h, PW)
    assert not auth.verify_password(h, PW + "!")


def _setup(db: Session) -> tuple[auth.NewUser, pyotp.TOTP]:
    new = auth.create_user(db, "me", PW)
    db.commit()
    return new, pyotp.TOTP(new.totp_secret)


def test_create_user_is_single_user(db: Session) -> None:
    new, _ = _setup(db)
    assert new.provisioning_uri.startswith("otpauth://totp/Dalal:me")
    with pytest.raises(ValueError, match="single-user"):
        auth.create_user(db, "other", PW)
    with pytest.raises(ValueError, match="12 characters"):
        auth.create_user(db, "x", "short")


def test_login_requires_password_and_totp(db: Session) -> None:
    _, totp = _setup(db)
    r = get_kv()
    with pytest.raises(auth.InvalidCredentials):
        auth.authenticate(db, r, "me", "wrong password", totp.now())
    with pytest.raises(auth.InvalidCredentials):
        auth.authenticate(db, r, "me", PW, "000000" if totp.now() != "000000" else "111111")
    user = auth.authenticate(db, r, "me", PW, totp.now())
    assert user.username == "me" and user.last_login_at is not None


def test_totp_code_is_single_use(db: Session) -> None:
    _, totp = _setup(db)
    r = get_kv()
    code = totp.now()
    auth.authenticate(db, r, "me", PW, code)
    with pytest.raises(auth.InvalidCredentials):
        auth.authenticate(db, r, "me", PW, code)


def test_lockout_after_repeated_failures(db: Session) -> None:
    _, totp = _setup(db)
    r = get_kv()
    for _ in range(auth.MAX_FAILURES):
        with pytest.raises(auth.InvalidCredentials):
            auth.authenticate(db, r, "me", "nope", "123456")
    with pytest.raises(auth.LoginLocked):
        auth.authenticate(db, r, "me", PW, totp.now())


def test_sessions(db: Session) -> None:
    r = get_kv()
    tok = auth.create_session(r, 42, ttl_hours=1)
    assert auth.resolve_session(r, tok) == 42
    assert auth.resolve_session(r, "forged") is None
    auth.destroy_session(r, tok)
    assert auth.resolve_session(r, tok) is None


def test_totp_replay_cannot_use_unicode_lookalike_digits(db: Session) -> None:
    """Regression: pyotp NFKC-normalises codes, so full-width digits of a used code
    passed verify() under a different replay key."""
    _, totp = _setup(db)
    r = get_kv()
    code = totp.now()
    auth.authenticate(db, r, "me", PW, code)
    fullwidth = "".join(chr(ord(ch) - ord("0") + 0xFF10) for ch in code)
    with pytest.raises(auth.InvalidCredentials):
        auth.authenticate(db, r, "me", PW, fullwidth)
    with pytest.raises(auth.InvalidCredentials):
        auth.authenticate(db, r, "me", PW, code)

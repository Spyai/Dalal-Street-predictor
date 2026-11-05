"""Symmetric encryption for secrets stored in the database (TOTP seed, Kite token)."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from app.config import get_settings


class SecretKeyMissing(RuntimeError):
    pass


def _fernet(secret_key: str | None = None) -> Fernet:
    key = secret_key if secret_key is not None else get_settings().secret_key
    if not key or len(key) < 32:
        raise SecretKeyMissing("SECRET_KEY must be set (>= 32 chars). See .env.example.")
    digest = hashlib.sha256(("dalal-fernet:" + key).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(plaintext: str, secret_key: str | None = None) -> str:
    return _fernet(secret_key).encrypt(plaintext.encode()).decode()


def decrypt(token: str, secret_key: str | None = None) -> str:
    return _fernet(secret_key).decrypt(token.encode()).decode()

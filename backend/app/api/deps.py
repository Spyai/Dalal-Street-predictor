"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.cache import KV, get_kv
from app.db import get_db
from app.models import AppUser
from app.security.auth import SESSION_COOKIE, resolve_session


def kv_dep() -> KV:
    return get_kv()


def current_user(
    db: Session = Depends(get_db),
    r: KV = Depends(kv_dep),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> AppUser:
    user_id = resolve_session(r, session_token)
    user = db.get(AppUser, user_id) if user_id is not None else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")
    return user


def optional_user(
    db: Session = Depends(get_db),
    r: KV = Depends(kv_dep),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> AppUser | None:
    user_id = resolve_session(r, session_token)
    return db.get(AppUser, user_id) if user_id is not None else None

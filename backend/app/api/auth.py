"""Login / logout / me."""

from __future__ import annotations

import redis
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import current_user, redis_dep
from app.config import get_settings
from app.db import get_db
from app.models import AppUser
from app.schemas import LoginIn, MeOut
from app.security.auth import (
    SESSION_COOKIE,
    InvalidCredentials,
    LoginLocked,
    authenticate,
    create_session,
    destroy_session,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=MeOut)
def login(
    body: LoginIn,
    response: Response,
    db: Session = Depends(get_db),
    r: redis.Redis = Depends(redis_dep),
) -> MeOut:
    s = get_settings()
    try:
        user = authenticate(db, r, body.username, body.password, body.totp)
    except LoginLocked as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc
    except InvalidCredentials as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    token = create_session(r, user.id, s.session_ttl_hours)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=s.session_ttl_hours * 3600,
        httponly=True,
        secure=s.cookie_secure,
        samesite="lax",  # lax so the Kite redirect back to /api/kite/callback carries it
        path="/",
    )
    return MeOut(username=user.username)


@router.post("/logout", status_code=204)
def logout(
    response: Response,
    r: redis.Redis = Depends(redis_dep),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> Response:
    destroy_session(r, session_token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.status_code = 204
    return response


@router.get("/me", response_model=MeOut)
def me(user: AppUser = Depends(current_user)) -> MeOut:
    return MeOut(username=user.username)

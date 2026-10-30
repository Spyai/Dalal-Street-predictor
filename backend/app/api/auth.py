"""Login / logout / me, and one-time first-user setup."""

from __future__ import annotations

import base64
import secrets

import qrcode
import qrcode.image.svg
import redis
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, redis_dep
from app.config import get_settings
from app.db import get_db
from app.models import AppUser
from app.schemas import LoginIn, MeOut, SetupIn, SetupOut, SetupStatusOut
from app.security.auth import (
    SESSION_COOKIE,
    InvalidCredentials,
    LoginLocked,
    authenticate,
    create_session,
    create_user,
    destroy_session,
)

SETUP_MAX_ATTEMPTS = 10
SETUP_WINDOW_SECONDS = 3600
MIN_SETUP_TOKEN = 16

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


def _has_user(db: Session) -> bool:
    return db.scalar(select(AppUser.id).limit(1)) is not None


@router.get("/setup-status", response_model=SetupStatusOut)
def setup_status(db: Session = Depends(get_db)) -> SetupStatusOut:
    token = get_settings().setup_token
    return SetupStatusOut(
        needs_setup=not _has_user(db), setup_enabled=len(token) >= MIN_SETUP_TOKEN
    )


@router.post("/setup", response_model=SetupOut)
def setup(
    body: SetupIn, db: Session = Depends(get_db), r: redis.Redis = Depends(redis_dep)
) -> SetupOut:
    """Create the single login from the browser, for hosts without a terminal (Vercel).

    Needs SETUP_TOKEN (>= 16 chars) from the server environment and only works while no
    user exists, so a public URL cannot be claimed by someone else.
    """
    expected = get_settings().setup_token
    if len(expected) < MIN_SETUP_TOKEN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Web setup is off. Set SETUP_TOKEN.")
    attempts = r.incr("auth:setup_attempts")
    if attempts == 1:
        r.expire("auth:setup_attempts", SETUP_WINDOW_SECONDS)
    if attempts > SETUP_MAX_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many setup attempts.")
    if not secrets.compare_digest(body.token.encode(), expected.encode()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Wrong setup token.")
    if _has_user(db):
        raise HTTPException(status.HTTP_409_CONFLICT, "Already set up. Log in instead.")
    try:
        new = create_user(db, body.username, body.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    svg = qrcode.make(
        new.provisioning_uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=10
    ).to_string()
    return SetupOut(
        username=new.username,
        totp_secret=new.totp_secret,
        provisioning_uri=new.provisioning_uri,
        qr_svg_data_url="data:image/svg+xml;base64," + base64.b64encode(svg).decode(),
    )

"""Kite Connect daily login endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import current_user, optional_user
from app.config import get_settings
from app.db import get_db
from app.models import AppUser
from app.providers.base import NotConfigured, SourceUnavailable
from app.schemas import KiteStatusOut
from app.services import kite_auth
from app.services.source_status import mark_error, mark_ok

router = APIRouter(prefix="/api/kite", tags=["kite"])


def kite_status(db: Session) -> KiteStatusOut:
    s = get_settings()
    sess = kite_auth.active_session(db) if s.kite_configured else None
    return KiteStatusOut(
        configured=s.kite_configured,
        connected=sess is not None,
        kite_user_id=sess.kite_user_id if sess else None,
        expires_at=sess.expires_at if sess else None,
        redirect_url=s.kite_redirect_url,
    )


@router.get("/status", response_model=KiteStatusOut)
def status_(db: Session = Depends(get_db), _: AppUser = Depends(current_user)) -> KiteStatusOut:
    return kite_status(db)


@router.get("/login")
def login(_: AppUser = Depends(current_user)) -> RedirectResponse:
    try:
        return RedirectResponse(kite_auth.login_url(), status_code=302)
    except NotConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.reason) from exc


@router.get("/callback")
def callback(
    request_token: str | None = Query(default=None),
    status_param: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
    user: AppUser | None = Depends(optional_user),
) -> RedirectResponse:
    """Kite redirects here after login.

    Accepted if you are logged in to Dalal in this browser, OR if KITE_USER_ID is set
    (then only that Zerodha account's token is accepted). This lets the Telegram
    login link work from your phone without logging in to Dalal first.
    """
    s = get_settings()
    if status_param != "success" or not request_token:
        return RedirectResponse("/?kite=cancelled", status_code=302)
    if user is None and not s.kite_user_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Log in to Dalal first, or set KITE_USER_ID so the callback can verify the account.",
        )
    try:
        kite_auth.complete_login(db, request_token, s)
    except PermissionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except Exception as exc:
        mark_error(
            "kite", SourceUnavailable("kite", f"token exchange failed: {exc.__class__.__name__}")
        )
        return RedirectResponse("/?kite=failed", status_code=302)
    mark_ok("kite")
    return RedirectResponse("/?kite=connected", status_code=302)

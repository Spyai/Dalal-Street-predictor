"""Telegram alerts, sent only to the configured private chat."""

from __future__ import annotations

import httpx

from app.config import Settings, get_settings
from app.logging import get_logger
from app.services.source_status import mark_error, mark_ok

log = get_logger(__name__)
MAX_LEN = 4000  # Telegram limit is 4096 chars


class TelegramNotConfigured(RuntimeError):
    pass


def send(
    text: str,
    *,
    settings: Settings | None = None,
    transport: httpx.BaseTransport | None = None,
    disable_preview: bool = True,
) -> bool:
    """Send a message. Returns False (and logs) if Telegram is not configured or fails.

    Alerts must never crash the job that raised them.
    """
    s = settings or get_settings()
    if not s.telegram_configured:
        # Never log message text: job logs may be public (GitHub Actions on a public repo).
        log.info("telegram_not_configured", chars=len(text))
        return False
    url = f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage"
    chunks = [text[i : i + MAX_LEN] for i in range(0, len(text), MAX_LEN)] or [""]
    try:
        with httpx.Client(timeout=15, transport=transport) as client:
            for chunk in chunks:
                resp = client.post(
                    url,
                    json={
                        "chat_id": s.telegram_chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": disable_preview,
                    },
                )
                resp.raise_for_status()
    except httpx.HTTPError as exc:
        # Never log the URL: it contains the bot token.
        log.warning("telegram_send_failed", error=exc.__class__.__name__)
        mark_error("telegram", RuntimeError(f"send failed: {exc.__class__.__name__}"))
        return False
    mark_ok("telegram")
    return True

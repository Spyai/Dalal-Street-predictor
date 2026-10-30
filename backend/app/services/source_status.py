"""Track health and freshness of every external source.

Status writes use their own short transaction so a failure record survives even when
the calling job rolls back.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db import session_scope
from app.logging import get_logger
from app.models import SourceStatus
from app.providers.base import AuthExpired, NotConfigured, ProviderError

log = get_logger(__name__)

# Human-readable names and what to do if a source is down.
SOURCE_INFO: dict[str, dict[str, str]] = {
    "kite": {
        "label": "Zerodha Kite Connect",
        "hint": "Set KITE_API_KEY/KITE_API_SECRET and log in to Kite each morning.",
    },
    "yfinance": {
        "label": "Yahoo Finance (yfinance)",
        "hint": "Unofficial. Kite covers Indian prices; global cues have no free replacement.",
    },
    "nse": {
        "label": "NSE public data",
        "hint": "If NSE blocks Dalal, import NSE CSVs into /imports manually.",
    },
    "telegram": {
        "label": "Telegram alerts",
        "hint": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
    },
}


def _upsert(source: str, values: dict[str, object]) -> None:
    try:
        with session_scope() as db:
            stmt = insert(SourceStatus).values(source=source, **values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[SourceStatus.source],
                set_={**values, "updated_at": datetime.now(UTC)},
            )
            db.execute(stmt)
    except Exception as exc:  # status tracking must never break a job
        log.warning("source_status_write_failed", source=source, error=str(exc))


def mark_ok(source: str, data_as_of: datetime | None = None) -> None:
    values: dict[str, object] = {"status": "ok", "last_success_at": datetime.now(UTC)}
    if data_as_of is not None:
        values["data_as_of"] = data_as_of
    _upsert(source, values)


def mark_error(source: str, err: Exception) -> None:
    if isinstance(err, NotConfigured):
        status = "not_configured"
    elif isinstance(err, AuthExpired):
        status = "auth_required"
    else:
        status = "unavailable"
    hint = err.hint if isinstance(err, ProviderError) and err.hint else None
    _upsert(
        source,
        {
            "status": status,
            "last_error_at": datetime.now(UTC),
            "last_error": str(err)[:2000],
            "hint": hint or SOURCE_INFO.get(source, {}).get("hint"),
        },
    )
    log.warning("source_error", source=source, status=status, error=str(err))


def all_statuses() -> dict[str, SourceStatus]:
    with session_scope() as db:
        rows = db.scalars(select(SourceStatus)).all()
        return {r.source: r for r in rows}

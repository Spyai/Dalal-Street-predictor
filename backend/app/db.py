"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _serverless_kwargs() -> dict[str, Any]:
    # Warm serverless instances serve many requests: small pools, checked before use and
    # recycled often, and no server-side prepared statements (they break behind a
    # transaction-mode pooler such as Neon's).
    return {
        "pool_pre_ping": True,
        "pool_recycle": 240,
        "pool_timeout": 10,
        "connect_args": {"prepare_threshold": None},
    }


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    if s.is_serverless:
        return create_engine(s.sqlalchemy_url, pool_size=2, max_overflow=10, **_serverless_kwargs())
    return create_engine(s.sqlalchemy_url, pool_pre_ping=True, pool_size=5)


@lru_cache
def get_kv_engine() -> Engine:
    """Separate pool for app.kv.PostgresKV.

    A request holds its session's connection while it makes key-value calls. Sharing one
    pool lets enough concurrent requests take every connection and then wait on each
    other for a second one. Key-value transactions are short and never nest, so on
    their own pool they can't hold-and-wait.
    """
    s = get_settings()
    extra = _serverless_kwargs() if s.is_serverless else {"pool_pre_ping": True}
    return create_engine(s.sqlalchemy_url, pool_size=1, max_overflow=9, **extra)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for jobs and scripts: commit on success, rollback on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

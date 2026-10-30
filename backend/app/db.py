"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    if s.is_serverless:
        # Warm serverless instances serve many requests, so keep a tiny pool (checked
        # before use, recycled often), and no server-side prepared statements: they break
        # behind a transaction-mode pooler such as Neon's.
        return create_engine(
            s.sqlalchemy_url,
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
            pool_recycle=240,
            connect_args={"prepare_threshold": None},
        )
    return create_engine(s.sqlalchemy_url, pool_pre_ping=True, pool_size=5)


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

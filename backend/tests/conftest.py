"""Test configuration.

Pure unit tests run anywhere. Tests marked `db` need PostgreSQL (TimescaleDB optional)
and Redis; they are skipped if those are not reachable. Defaults match
`docker compose -f docker-compose.test.yml up -d` (see Makefile `make test`).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

TEST_DB = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://dalal:devpass@127.0.0.1:55432/dalal_test"
)
TEST_REDIS = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:56379/15")

os.environ["DATABASE_URL"] = TEST_DB
os.environ["REDIS_URL"] = TEST_REDIS
os.environ["SECRET_KEY"] = "test-secret-key-0123456789-abcdefghijklmnopqrstuvwxyz"
os.environ["IMPORTS_DIR"] = tempfile.mkdtemp(prefix="dalal-imports-")
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dalal-data-")
for key in (
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_USER_ID",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "WATCHLIST",
):
    os.environ[key] = ""
os.environ["JOB_BACKOFF_SECONDS"] = "30"
os.environ["JOB_MAX_ATTEMPTS"] = "3"

FIXTURES = Path(__file__).parent / "fixtures"


def _infra_available() -> str | None:
    try:
        import psycopg
        import redis

        url = TEST_DB.replace("postgresql+psycopg://", "postgresql://")
        admin = url.rsplit("/", 1)[0] + "/postgres"
        dbname = url.rsplit("/", 1)[1]
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{dbname}"')
        redis.Redis.from_url(TEST_REDIS, socket_connect_timeout=2).ping()
    except Exception as exc:  # pragma: no cover - depends on environment
        return f"{exc.__class__.__name__}: {exc}"
    return None


_INFRA_ERROR: str | None = None
_INFRA_CHECKED = False


@pytest.fixture(scope="session")
def migrated_db() -> Iterator[None]:
    global _INFRA_ERROR, _INFRA_CHECKED
    if not _INFRA_CHECKED:
        _INFRA_ERROR = _infra_available()
        _INFRA_CHECKED = True
    if _INFRA_ERROR:
        pytest.skip(f"database/redis not available: {_INFRA_ERROR}")
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(cfg, "head")
    yield


@pytest.fixture()
def db(migrated_db: None) -> Iterator[object]:
    """A clean database + Redis for each test."""
    from sqlalchemy import text

    from app.cache import get_kv
    from app.db import Base, get_engine, get_sessionmaker

    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    with get_engine().begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    get_kv().flushdb()
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "db" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "db: needs PostgreSQL + Redis")

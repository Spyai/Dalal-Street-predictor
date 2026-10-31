"""The Postgres key-value store must behave like the Redis commands it replaces.

Every test runs against both a real Redis and PostgresKV, so a semantic difference
between them fails the suite.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
import redis
from sqlalchemy.orm import Session

from app.kv import KVError, PostgresKV
from tests.conftest import TEST_REDIS


@pytest.fixture(params=["redis", "postgres"])
def kv(request: pytest.FixtureRequest, db: Session) -> Iterator[Any]:
    if request.param == "redis":
        client = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
        client.flushdb()
        yield client
        client.flushdb()
    else:
        from app.db import get_engine

        store = PostgresKV(get_engine())
        store.flushdb()
        yield store


def test_get_set_delete(kv: Any) -> None:
    assert kv.get("a") is None
    assert kv.set("a", "1")
    assert kv.get("a") == "1"
    assert kv.set("a", 2)  # overwrite, non-string value
    assert kv.get("a") == "2"
    assert kv.delete("a", "missing") == 1
    assert kv.get("a") is None


def test_set_nx_only_when_absent(kv: Any) -> None:
    assert kv.set("k", "first", nx=True)
    assert not kv.set("k", "second", nx=True)
    assert kv.get("k") == "first"


def test_expiry_ex_and_px(kv: Any) -> None:
    kv.set("s", "x", ex=1)
    kv.set("m", "x", px=150)
    assert kv.get("s") == "x" and kv.get("m") == "x"
    time.sleep(0.3)
    assert kv.get("m") is None, "px expiry"
    assert kv.exists("m") == 0
    # An expired key can be claimed again with nx.
    assert kv.set("m", "again", nx=True, px=1000)
    time.sleep(0.9)
    assert kv.get("s") is None, "ex expiry"


def test_set_without_ttl_clears_ttl(kv: Any) -> None:
    kv.set("t", "x", ex=100)
    assert 0 < kv.pttl("t") <= 100_000
    kv.set("t", "y")
    assert kv.pttl("t") == -1


def test_pttl_sentinels(kv: Any) -> None:
    assert kv.pttl("nope") == -2
    kv.set("forever", "1")
    assert kv.pttl("forever") == -1
    kv.set("soon", "1", px=5000)
    assert 0 < kv.pttl("soon") <= 5000


def test_incr_and_expire(kv: Any) -> None:
    assert kv.incr("c") == 1
    assert kv.incr("c") == 2
    assert kv.expire("c", 100)
    assert kv.incr("c") == 3, "incr keeps the ttl"
    assert 0 < kv.pttl("c") <= 100_000
    assert not kv.expire("missing", 10)


def test_incr_restarts_after_expiry(kv: Any) -> None:
    kv.set("c", "5", px=100)
    time.sleep(0.25)
    assert kv.incr("c") == 1
    assert kv.pttl("c") == -1


def test_incr_non_integer_raises_redis_error(kv: Any) -> None:
    kv.set("str", "abc")
    with pytest.raises(redis.RedisError):
        kv.incr("str")


def test_exists_counts_live_keys(kv: Any) -> None:
    kv.set("x", "1")
    kv.set("y", "1")
    assert kv.exists("x") == 1
    assert kv.exists("x", "y", "z") == 2


def test_pipeline_incr_expire(kv: Any) -> None:
    pipe = kv.pipeline()
    pipe.incr("fails")
    pipe.expire("fails", 60)
    res = pipe.execute()
    assert res[0] == 1 and res[1] in (True, 1)
    assert 0 < kv.pttl("fails") <= 60_000


def test_expire_zero_deletes(kv: Any) -> None:
    kv.set("gone", "1")
    kv.expire("gone", 0)
    assert kv.get("gone") is None


def test_ping_and_flush(kv: Any) -> None:
    kv.set("a", "1")
    assert kv.ping()
    kv.flushdb()
    assert kv.get("a") is None


def test_postgres_errors_are_redis_errors() -> None:
    from sqlalchemy import create_engine

    broken = PostgresKV(create_engine("postgresql+psycopg://nobody:x@127.0.0.1:1/none"))
    with pytest.raises(KVError):
        broken.get("a")
    assert issubclass(KVError, redis.RedisError)


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("KV_URL", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert Settings().kv_backend == "redis", "Docker Compose default"
    monkeypatch.setenv("VERCEL", "1")
    assert Settings().kv_backend == "postgres", "Vercel without Redis"
    assert Settings(kv_url="rediss://up").kv_backend == "redis", "Vercel with Upstash"
    assert Settings(cache_backend="postgres").kv_backend == "postgres"


def test_empty_redis_url_does_not_select_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.delenv("KV_URL", raising=False)
    assert Settings().kv_backend == "postgres"


def test_concurrent_requests_cannot_starve_kv_of_connections(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: KV calls used the request pool, so requests holding a session while
    making KV calls could exhaust it and wait on each other (seen as 30 s hangs, 401s)."""
    import threading

    from sqlalchemy import text

    from app import cache
    from app import db as app_db
    from app.config import get_settings

    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("REDIS_URL", raising=False)
    caches = (
        get_settings,
        app_db.get_engine,
        app_db.get_kv_engine,
        app_db.get_sessionmaker,
        cache.get_kv,
    )
    for fn in caches:
        fn.cache_clear()
    try:
        assert get_settings().kv_backend == "postgres"
        kv_store = cache.get_kv()
        assert isinstance(kv_store, PostgresKV)
        assert kv_store._engine is not app_db.get_engine(), "KV must not share the request pool"
        n = 12  # the request pool's full capacity (2 + 10 overflow)
        held = threading.Barrier(n)
        errors: list[BaseException] = []

        def request(i: int) -> None:
            session = app_db.get_sessionmaker()()
            try:
                session.execute(text("SELECT 1"))  # the request now holds a connection
                held.wait(timeout=20)  # ...and so does every other request
                kv_store.set(f"session:{i}", "1", ex=60)
                assert kv_store.get(f"session:{i}") == "1"
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                session.close()

        threads = [threading.Thread(target=request, args=(i,)) for i in range(n)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors
        assert time.monotonic() - start < 15
    finally:
        monkeypatch.undo()
        for fn in caches:
            fn.cache_clear()

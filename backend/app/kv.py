"""Postgres-backed stand-in for the small subset of Redis that Dalal uses.

Used on hosts without Redis (Vercel without Upstash), so the database is the only
external service. Same semantics as the Redis commands it mirrors: keys expire,
`set(nx=True)` is atomic, `incr` creates missing keys, and errors are raised as
`redis.RedisError` subclasses so existing `except redis.RedisError` handlers still work.
Every operation runs in its own transaction, like a Redis command: a login-failure
counter survives the request that raised the error.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import redis
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import SQLAlchemyError

LIVE = "(expires_at IS NULL OR expires_at > now())"
EXPIRED = "(kv_entry.expires_at IS NOT NULL AND kv_entry.expires_at <= now())"


class KVError(redis.RedisError):
    pass


def _expiry_sql(ex: int | None, px: int | None) -> tuple[str, dict[str, Any]]:
    if px is not None:
        return "now() + :ttl_ms * interval '1 millisecond'", {"ttl_ms": int(px)}
    if ex is not None:
        return "now() + :ttl_ms * interval '1 millisecond'", {"ttl_ms": int(ex) * 1000}
    return "NULL", {}


class PostgresKV:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @contextmanager
    def _tx(self) -> Iterator[Connection]:
        try:
            with self._engine.begin() as conn:
                yield conn
        except SQLAlchemyError as exc:
            raise KVError(f"kv store error: {exc.__class__.__name__}") from exc

    # ------------------------------------------------------------------ commands

    def get(self, key: str) -> str | None:
        with self._tx() as c:
            return self._get(c, key)

    def set(
        self,
        key: str,
        value: object,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        with self._tx() as c:
            return self._set(c, key, value, ex=ex, px=px, nx=nx)

    def delete(self, *keys: str) -> int:
        with self._tx() as c:
            return self._delete(c, *keys)

    def exists(self, *keys: str) -> int:
        with self._tx() as c:
            return int(
                c.execute(
                    text(f"SELECT count(*) FROM kv_entry WHERE key = ANY(:keys) AND {LIVE}"),
                    {"keys": list(keys)},
                ).scalar_one()
            )

    def incr(self, key: str) -> int:
        with self._tx() as c:
            return self._incr(c, key)

    def expire(self, key: str, seconds: int) -> bool:
        with self._tx() as c:
            return self._expire(c, key, seconds)

    def pttl(self, key: str) -> int:
        """Milliseconds to expiry; -1 if the key has no expiry, -2 if it doesn't exist."""
        with self._tx() as c:
            row = c.execute(
                text(
                    "SELECT expires_at IS NULL AS forever, "
                    "GREATEST(0, CEIL(EXTRACT(EPOCH FROM (expires_at - now())) * 1000))::bigint "
                    f"AS ms FROM kv_entry WHERE key = :k AND {LIVE}"
                ),
                {"k": key},
            ).one_or_none()
        if row is None:
            return -2
        return -1 if row.forever else int(row.ms)

    def ping(self) -> bool:
        with self._tx() as c:
            c.execute(text("SELECT 1"))
        return True

    def flushdb(self) -> bool:
        with self._tx() as c:
            c.execute(text("DELETE FROM kv_entry"))
        return True

    def pipeline(self) -> Pipeline:
        return Pipeline(self)

    # ------------------------------------------------------------ implementation

    def _get(self, c: Connection, key: str) -> str | None:
        return c.execute(
            text(f"SELECT value FROM kv_entry WHERE key = :k AND {LIVE}"), {"k": key}
        ).scalar_one_or_none()

    def _set(
        self,
        c: Connection,
        key: str,
        value: object,
        *,
        ex: int | None,
        px: int | None,
        nx: bool,
    ) -> bool | None:
        expiry, params = _expiry_sql(ex, px)
        # Housekeeping: drop expired rows so the table stays small.
        c.execute(text("DELETE FROM kv_entry WHERE expires_at <= now()"))
        guard = f" WHERE {EXPIRED}" if nx else ""
        row = c.execute(
            text(
                "INSERT INTO kv_entry (key, value, expires_at) "
                f"VALUES (:k, :v, {expiry}) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                f"expires_at = EXCLUDED.expires_at{guard} RETURNING 1"
            ),
            {"k": key, "v": str(value), **params},
        ).first()
        if nx:
            return True if row is not None else None
        return True

    def _delete(self, c: Connection, *keys: str) -> int:
        rows = c.execute(
            text(f"DELETE FROM kv_entry WHERE key = ANY(:keys) RETURNING {LIVE} AS live"),
            {"keys": list(keys)},
        ).all()
        return sum(1 for r in rows if r.live)

    def _incr(self, c: Connection, key: str) -> int:
        value = c.execute(
            text(
                "INSERT INTO kv_entry (key, value, expires_at) VALUES (:k, '1', NULL) "
                "ON CONFLICT (key) DO UPDATE SET "
                f"value = CASE WHEN {EXPIRED} THEN '1' "
                "ELSE (kv_entry.value::bigint + 1)::text END, "
                f"expires_at = CASE WHEN {EXPIRED} THEN NULL ELSE kv_entry.expires_at END "
                "RETURNING value"
            ),
            {"k": key},
        ).scalar_one()
        return int(value)

    def _expire(self, c: Connection, key: str, seconds: int) -> bool:
        if seconds <= 0:
            return self._delete(c, key) > 0
        row = c.execute(
            text(
                "UPDATE kv_entry SET expires_at = now() + :s * interval '1 second' "
                f"WHERE key = :k AND {LIVE} RETURNING 1"
            ),
            {"k": key, "s": int(seconds)},
        ).first()
        return row is not None


class Pipeline:
    """Queues commands and runs them in one transaction on execute(), like redis-py."""

    def __init__(self, kv: PostgresKV) -> None:
        self._kv = kv
        self._ops: list[Callable[[Connection], Any]] = []

    def incr(self, key: str) -> Pipeline:
        self._ops.append(lambda c: self._kv._incr(c, key))
        return self

    def expire(self, key: str, seconds: int) -> Pipeline:
        self._ops.append(lambda c: self._kv._expire(c, key, seconds))
        return self

    def set(
        self,
        key: str,
        value: object,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> Pipeline:
        self._ops.append(lambda c: self._kv._set(c, key, value, ex=ex, px=px, nx=nx))
        return self

    def delete(self, *keys: str) -> Pipeline:
        self._ops.append(lambda c: self._kv._delete(c, *keys))
        return self

    def execute(self) -> list[Any]:
        ops, self._ops = self._ops, []
        with self._kv._tx() as c:
            return [op(c) for op in ops]

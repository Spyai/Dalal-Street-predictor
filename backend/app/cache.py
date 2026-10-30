"""Key-value store (Redis, or Postgres via app.kv) plus caching and rate-limit helpers."""

from __future__ import annotations

import contextlib
import json
import time
from functools import lru_cache
from typing import Any

import redis

from app.config import get_settings
from app.kv import PostgresKV

HEARTBEAT_KEY = "scheduler:heartbeat"

KV = redis.Redis | PostgresKV


@lru_cache
def get_kv() -> KV:
    s = get_settings()
    if s.kv_backend == "postgres":
        from app.db import get_engine

        return PostgresKV(get_engine())
    return redis.Redis.from_url(
        s.cache_url,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


def cache_get_json(key: str, r: KV | None = None) -> Any | None:
    r = r or get_kv()
    try:
        raw = r.get(key)
    except redis.RedisError:
        return None
    return json.loads(raw) if raw else None


def cache_set_json(key: str, value: Any, ttl_seconds: int, r: KV | None = None) -> None:
    r = r or get_kv()
    # The cache is an optimisation, never a dependency for correctness.
    with contextlib.suppress(redis.RedisError):
        r.set(key, json.dumps(value, default=str), ex=ttl_seconds)


def wait_for_slot(key: str, min_interval_seconds: float, r: KV | None = None) -> None:
    """Block until at least `min_interval_seconds` have passed since the last call for `key`.

    Shared across processes (API + scheduler) via Redis so the combined request rate to
    any one host stays under the limit. Falls back to no coordination if Redis is down;
    callers also keep an in-process limiter.
    """
    r = r or get_kv()
    redis_key = f"ratelimit:{key}"
    ms = int(min_interval_seconds * 1000)
    for _ in range(600):
        try:
            if r.set(redis_key, "1", nx=True, px=ms):
                return
            ttl = r.pttl(redis_key)
        except redis.RedisError:
            return
        time.sleep(max(ttl, 50) / 1000 if ttl and ttl > 0 else 0.05)

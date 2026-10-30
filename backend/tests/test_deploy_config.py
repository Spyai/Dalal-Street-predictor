"""Hosted-deployment settings: database URL normalisation, Redis fallback, scheduler mode."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.api.health import external_scheduler_alive
from app.config import IST, Settings, normalize_db_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgres://u:p@h/db?sslmode=require", "postgresql+psycopg://u:p@h/db?sslmode=require"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ],
)
def test_normalize_db_url(raw: str, expected: str) -> None:
    assert normalize_db_url(raw) == expected


def test_postgres_url_used_when_database_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings(postgres_url="postgres://u:p@neon/db")
    assert s.sqlalchemy_url == "postgresql+psycopg://u:p@neon/db"


def test_redis_url_falls_back_to_kv_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert Settings(kv_url="rediss://default:t@upstash:6379").cache_url.startswith("rediss://")
    explicit = Settings(redis_url="redis://mine:6379/0", kv_url="rediss://other")
    assert explicit.cache_url == "redis://mine:6379/0"


def test_external_scheduler_liveness() -> None:
    day = datetime(2026, 10, 30, 12, 0, tzinfo=IST)
    assert external_scheduler_alive(day - timedelta(minutes=40), day)
    assert not external_scheduler_alive(day - timedelta(minutes=90), day)
    night = datetime(2026, 10, 31, 3, 0, tzinfo=IST)  # last cue run 23:30
    assert external_scheduler_alive(night - timedelta(hours=3, minutes=30), night)
    assert not external_scheduler_alive(None, day)


def test_vercel_entrypoint_exposes_app() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "api" / "index.py"
    spec = importlib.util.spec_from_file_location("vercel_index", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fastapi import FastAPI

    assert isinstance(mod.app, FastAPI)


def test_vercel_requirements_cover_runtime_imports() -> None:
    """Every pyproject runtime dependency the API needs is in requirements.txt."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    reqs = (root / "requirements.txt").read_text().splitlines()
    names = {r.split(">")[0].split("[")[0].split("=")[0].strip().lower() for r in reqs}
    not_needed_on_vercel = {"uvicorn", "alembic", "apscheduler"}
    for dep in deps:
        name = dep.split(">")[0].split("[")[0].split("=")[0].strip().lower()
        if name not in not_needed_on_vercel:
            assert name in names, f"{name} missing from requirements.txt"

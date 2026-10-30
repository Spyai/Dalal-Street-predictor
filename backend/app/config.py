"""Application settings, loaded from environment variables (see .env.example)."""

from __future__ import annotations

import os
from datetime import time
from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

IST = ZoneInfo("Asia/Kolkata")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Core
    secret_key: str = Field(default="", min_length=0)
    app_base_url: str = "http://127.0.0.1:3000"
    cookie_secure: bool = False
    session_ttl_hours: int = 168
    log_level: str = "info"

    # Database / cache. On Vercel the Neon/Upstash integrations inject POSTGRES_URL /
    # KV_URL; DATABASE_URL / REDIS_URL win when both are set.
    database_url: str | None = None
    postgres_url: str | None = None
    kv_url: str | None = None
    postgres_user: str = "dalal"
    postgres_password: str = ""
    postgres_db: str = "dalal"
    postgres_host: str = "db"
    postgres_port: int = 5432
    redis_url: str = "redis://redis:6379/0"
    # auto: Redis, except on Vercel with no REDIS_URL/KV_URL, where a Postgres table is
    # used instead (see app/kv.py). Or force "redis" / "postgres".
    cache_backend: str = "auto"

    # Broker
    broker: str = "kite"
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_user_id: str = ""
    kite_login_reminder_time: time = time(7, 0)

    # LLM
    anthropic_api_key: str = ""
    anthropic_model_writer: str = "claude-opus-5-5"
    anthropic_model_fast: str = "claude-haiku-5-5"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Optional keys
    fred_api_key: str = ""
    newsapi_key: str = ""

    # Public sources
    nse_user_agent: str = "Dalal/0.1 (+personal research tool; single user; <=1 req/s)"
    public_source_min_interval_seconds: float = 1.0
    public_source_timeout_seconds: float = 20.0

    # Universe / backfill
    universe: str = "nifty500"
    watchlist: str = ""
    backfill_years: int = 5
    min_avg_turnover_cr: float = 10.0

    # Jobs. "internal": the scheduler container runs APScheduler (Docker Compose).
    # "external": something else triggers `app.cli run-job` (GitHub Actions on Vercel).
    scheduler_mode: str = "internal"
    job_max_attempts: int = 3
    job_backoff_seconds: float = 30.0
    global_cues_refresh_minutes: int = 10

    # Prediction / risk (used from Phase 2+)
    market_flat_threshold_pct: float = 0.30
    paper_capital: float = 1_000_000.0
    risk_per_trade_pct: float = 1.0
    min_sample_size: int = 100

    # One-time first-user setup through the web UI (/setup?token=...). Empty = disabled.
    setup_token: str = ""

    # Paths
    imports_dir: str = "/imports"
    data_dir: str = "/data"

    @field_validator("kite_login_reminder_time", mode="before")
    @classmethod
    def _parse_time(cls, v: object) -> object:
        if isinstance(v, str) and len(v) == 5:
            hh, mm = v.split(":")
            return time(int(hh), int(mm))
        return v

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url or self.postgres_url
        if url:
            return normalize_db_url(url)
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cache_url(self) -> str:
        if self._redis_url_given or not self.kv_url:
            return self.redis_url
        return self.kv_url

    @property
    def _redis_url_given(self) -> bool:
        # An empty REDIS_URL (e.g. an unset GitHub secret) counts as not given.
        return "redis_url" in self.model_fields_set and bool(self.redis_url)

    @property
    def kv_backend(self) -> str:
        if self.cache_backend in {"redis", "postgres"}:
            return self.cache_backend
        explicit_redis = self._redis_url_given or bool(self.kv_url)
        return "postgres" if self.is_serverless and not explicit_redis else "redis"

    @property
    def is_serverless(self) -> bool:
        return bool(os.environ.get("VERCEL"))

    @property
    def watchlist_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.watchlist.split(",") if s.strip()]

    @property
    def kite_configured(self) -> bool:
        return bool(self.kite_api_key and self.kite_api_secret)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def kite_redirect_url(self) -> str:
        return f"{self.app_base_url.rstrip('/')}/api/kite/callback"


def normalize_db_url(url: str) -> str:
    """Hosted Postgres (Neon, Vercel) hands out postgres:// URLs; SQLAlchemy needs a driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


@lru_cache
def get_settings() -> Settings:
    return Settings()

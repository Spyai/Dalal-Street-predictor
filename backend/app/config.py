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
    market_flat_threshold_pct: float = 0.30  # |close-to-close %| beyond this is UP/DOWN
    gap_threshold_pct: float = 0.25  # |open vs previous close %| beyond this is a gap
    no_edge_margin: float = 0.08  # top-two class probability gap below this = "No edge"
    confidence_high_prob: float = 0.55  # max probability for a "high" confidence label
    ledger_lock_time: time = time(9, 15)  # IST; a record written later is marked late
    prediction_targets: str = "NIFTY 50,NIFTY BANK"  # comma-separated INDEX symbols
    wf_min_train_days: int = 500  # walk-forward: first fold trains on this many sessions
    wf_step_days: int = 21  # walk-forward: refit every N sessions
    retrain_min_brier_gain: float = 0.005  # challenger must beat champion's Brier by this
    mlflow_tracking_uri: str = ""  # optional; empty = models are registered in Postgres only
    paper_capital: float = 1_000_000.0
    risk_per_trade_pct: float = 1.0
    min_sample_size: int = 100

    # Stock ideas (Phase 3). Horizons are volatility-scaled (ATR multiples); costs are the
    # round-trip delivery cost model used to rank ideas by expected value after costs.
    ideas_max_per_horizon: int = 3  # brief §1: 0-3 high-conviction ideas, never 10 weak ones
    ideas_min_p_target: float = 0.55  # P(target before stop) below this = not an idea
    ideas_min_ev_pct: float = 0.5  # expected value after costs (%) below this = not an idea
    ideas_universe_min_bars: int = 250  # a stock needs this much history to be considered
    drivers_top_n: int = 8  # drivers shown per call/idea; the full list is one click away
    swing_horizon_days: int = 10
    swing_target_atr: float = 2.0
    swing_stop_atr: float = 1.0
    positional_horizon_days: int = 60
    positional_target_atr: float = 4.0
    positional_stop_atr: float = 2.0
    brokerage_pct: float = 0.03  # per side, % of trade value (discount broker, delivery)
    stt_delivery_pct: float = 0.1  # per side on delivery (buy and sell)
    exchange_txn_pct: float = 0.00345  # NSE transaction charge per side
    gst_pct: float = 18.0  # on brokerage + exchange charges
    stamp_duty_pct: float = 0.015  # buy side only
    sebi_fee_pct: float = 0.0001  # per side
    slippage_pct: float = 0.1  # per side, assumed
    paper_max_position_pct: float = 10.0  # one idea may not exceed this share of paper capital

    # One-time first-user setup through the web UI (/setup?token=...). Empty = disabled.
    setup_token: str = ""

    # Paths
    imports_dir: str = "/imports"
    data_dir: str = "/data"

    @property
    def prediction_target_symbols(self) -> list[str]:
        return [t.strip().upper() for t in self.prediction_targets.split(",") if t.strip()]

    @field_validator("kite_login_reminder_time", "ledger_lock_time", mode="before")
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

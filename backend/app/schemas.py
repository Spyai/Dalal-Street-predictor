"""Pydantic schemas for all API input/output."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- auth


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    totp: str = Field(min_length=6, max_length=8)


class MeOut(BaseModel):
    username: str


class SetupStatusOut(BaseModel):
    needs_setup: bool
    setup_enabled: bool


class SetupIn(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=12, max_length=256)


class SetupOut(BaseModel):
    username: str
    totp_secret: str
    provisioning_uri: str
    qr_svg_data_url: str


# ----------------------------------------------------------------------------- today


class MarketClockOut(BaseModel):
    now_ist: datetime
    session: str
    is_trading_day: bool
    day_reason: str | None
    calendar_verified: bool
    next_trading_day: date
    previous_trading_day: date


class CueOut(BaseModel):
    key: str
    label: str
    group: str
    unit: str
    last: float | None
    prev_close: float | None
    change_pct: float | None
    change_bps: float | None
    as_of: datetime | None
    status: str
    source: str | None
    note: str | None


class FlowDayOut(BaseModel):
    trade_date: date
    fii_net_cr: float | None
    dii_net_cr: float | None
    fii_buy_cr: float | None
    fii_sell_cr: float | None
    dii_buy_cr: float | None
    dii_sell_cr: float | None
    provisional: bool
    source: str


class FlowsOut(BaseModel):
    days: list[FlowDayOut]
    status: str  # ok | stale | unavailable
    expected_latest: date
    note: str | None


class KiteStatusOut(BaseModel):
    configured: bool
    connected: bool
    kite_user_id: str | None
    expires_at: datetime | None
    redirect_url: str


class TodayOut(BaseModel):
    clock: MarketClockOut
    cues: list[CueOut]
    flows: FlowsOut
    kite: KiteStatusOut
    market_call: None = None  # Phase 2
    market_call_note: str = "Market call starts in Phase 2. No prediction is made yet."


# ----------------------------------------------------------------------------- stocks


class InstrumentOut(BaseModel):
    symbol: str
    exchange: str
    name: str | None
    industry: str | None
    in_universe: bool
    is_index: bool


class CandleOut(BaseModel):
    time: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    delivery_pct: float | None


class SeriesPoint(BaseModel):
    time: date
    value: float


class LevelOut(BaseModel):
    price: float
    touches: int
    kind: str


class StockOut(BaseModel):
    instrument: InstrumentOut
    candles: list[CandleOut]
    indicators: dict[str, list[SeriesPoint]]
    levels: list[LevelOut]
    last_close: float | None
    prev_close: float | None
    change_pct: float | None
    data_as_of: date | None
    data_status: str  # ok | stale | missing
    sources: list[str]
    notes: list[str]


# ----------------------------------------------------------------------------- health


class JobHealthOut(BaseModel):
    name: str
    label: str
    schedule: str
    phase: int
    built: bool
    last_status: str | None
    last_run_at: datetime | None
    last_finished_at: datetime | None
    last_message: str | None
    last_success_at: datetime | None


class SourceHealthOut(BaseModel):
    source: str
    label: str
    status: str
    last_success_at: datetime | None
    last_error_at: datetime | None
    last_error: str | None
    data_as_of: datetime | None
    hint: str | None


class FreshnessOut(BaseModel):
    dataset: str
    latest: date | datetime | None
    expected: date | None
    status: str
    detail: str | None = None


class HealthOut(BaseModel):
    now: datetime
    scheduler_mode: str
    scheduler_alive: bool
    scheduler_heartbeat: datetime | None
    database: str
    redis: str
    cache_backend: str
    jobs: list[JobHealthOut]
    sources: list[SourceHealthOut]
    freshness: list[FreshnessOut]
    models: list[dict[str, str]]
    models_note: str
    holiday_coverage: dict[int, int]

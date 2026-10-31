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


# ----------------------------------------------------------------------------- market call


class DriverOut(BaseModel):
    """One plain-English reason behind a call (app.ml.core.Driver)."""

    feature: str
    value: float | None
    contribution: float | None
    text: str


class GradeOut(BaseModel):
    """What the market did, written after the fact. All None until graded."""

    actual_open: float | None
    actual_close: float | None
    actual_pct: float | None
    actual_class: str | None
    correct: bool | None
    brier: float | None
    log_loss: float | None
    gap_actual_pct: float | None
    gap_correct: bool | None
    graded_at: datetime | None


class TrackOut(BaseModel):
    """The live track record that sits beside every call (honesty rules, brief §1)."""

    n_graded: int
    accuracy: float | None
    best_baseline: str | None
    best_baseline_label: str | None
    best_baseline_accuracy: float | None
    sample_warning: bool
    min_sample_size: int


class MarketCallOut(BaseModel):
    """The ledger's current call for one target and session, plus grade and track record."""

    id: int
    seq: int
    target: str
    session_date: date
    as_of: datetime
    is_late: bool
    reference_date: date
    reference_close: float
    predicted_class: str
    probs: dict[str, float]
    range_p10_pct: float | None
    range_p90_pct: float | None
    gap_call: str | None
    gap_expected_pct: float | None
    confidence: str
    no_edge: bool
    drivers: list[DriverOut]
    timing_guidance: str | None
    baselines: dict[str, str | None]
    model_version: str | None
    note: str | None
    grade: GradeOut | None
    track: TrackOut | None


class TodayOut(BaseModel):
    clock: MarketClockOut
    cues: list[CueOut]
    flows: FlowsOut
    kite: KiteStatusOut
    market_call: list[MarketCallOut] = Field(default_factory=list)
    market_call_note: str = ""  # why a target has no call; empty when every target has one


# ----------------------------------------------------------------------------- scorecard


class ConfidenceBucketOut(BaseModel):
    confidence: str
    n: int
    accuracy: float | None = None  # the service omits the key when n == 0


class CalibrationBinOut(BaseModel):
    bin_lo: float
    bin_hi: float
    n: int
    predicted_mean: float | None
    observed: float | None


class BaselineScoreOut(BaseModel):
    name: str
    label: str
    n: int
    accuracy: float | None


class OosMetricsOut(BaseModel):
    """The champion's walk-forward, out-of-sample metrics."""

    n_oos: int | None
    accuracy: float | None
    brier: float | None
    log_loss: float | None


class ChampionOut(BaseModel):
    id: int
    version: str
    kind: str
    trained_through: date
    train_rows: int
    oos: OosMetricsOut
    promoted_at: datetime | None
    created_at: datetime | None


class PredictionRowOut(BaseModel):
    """One ledger record with its grade (grade fields None until graded)."""

    id: int
    seq: int
    target: str
    session_date: date
    as_of: datetime | None
    is_late: bool
    predicted_class: str
    probs: dict[str, float]
    range_p10_pct: float | None
    range_p90_pct: float | None
    confidence: str
    no_edge: bool
    gap_call: str | None
    gap_expected_pct: float | None
    model_version_id: int | None
    corrects_prediction_id: int | None
    actual_open: float | None
    actual_close: float | None
    actual_pct: float | None
    actual_class: str | None
    correct: bool | None
    brier: float | None
    log_loss: float | None
    gap_actual_pct: float | None
    gap_correct: bool | None
    graded_at: datetime | None


class ScorecardOut(BaseModel):
    """Mirrors app.services.scorecard.Scorecard, plus the targets the switch can offer."""

    target: str
    targets: list[str]
    n_graded: int
    accuracy: float | None
    brier: float | None
    log_loss: float | None
    gap_n: int
    gap_accuracy: float | None
    by_confidence: list[ConfidenceBucketOut]
    calibration: list[CalibrationBinOut]
    baselines: list[BaselineScoreOut]
    verdict: str
    sample_warning: bool
    min_sample_size: int
    champion: ChampionOut | None
    recent: list[PredictionRowOut]


class LedgerVerifyOut(BaseModel):
    ok: bool
    checked: int
    first_bad_seq: int | None
    message: str


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


class ModelHealthOut(BaseModel):
    """A champion model in production and its out-of-sample metrics."""

    name: str
    target: str
    version: str
    trained_through: date
    oos_brier: float | None
    oos_accuracy: float | None
    n_oos: int | None
    promoted_at: datetime | None


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
    models: list[ModelHealthOut]
    models_note: str
    holiday_coverage: dict[int, int]

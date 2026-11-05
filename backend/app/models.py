"""SQLAlchemy ORM models.

Time-series tables (daily_bar, intraday_bar) become TimescaleDB hypertables in the
initial Alembic migration. Every row records its `source` so the UI can always say
where a number came from.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AppUser(Base):
    """The single user. Password is argon2-hashed; TOTP secret is Fernet-encrypted."""

    __tablename__ = "app_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    totp_secret_enc: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Instrument(Base):
    """A tradable or reference series: NSE/BSE equity, index, or global cue."""

    __tablename__ = "instrument"
    __table_args__ = (UniqueConstraint("exchange", "symbol", name="uq_instrument_exchange_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    exchange: Mapped[str] = mapped_column(String(16))  # NSE | BSE | INDEX | GLOBAL
    name: Mapped[str | None] = mapped_column(String(256))
    isin: Mapped[str | None] = mapped_column(String(16), index=True)
    series: Mapped[str | None] = mapped_column(String(8))
    industry: Mapped[str | None] = mapped_column(String(128))
    kite_token: Mapped[int | None] = mapped_column(BigInteger)
    yf_ticker: Mapped[str | None] = mapped_column(String(32))
    is_index: Mapped[bool] = mapped_column(Boolean, default=False)
    in_universe: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    lot_size: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DailyBar(Base):
    """End-of-day OHLCV. Hypertable on trade_date."""

    __tablename__ = "daily_bar"

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instrument.id", ondelete="CASCADE"), primary_key=True
    )
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    delivery_qty: Mapped[float | None] = mapped_column(Float)
    delivery_pct: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IntradayBar(Base):
    """Intraday candles. Hypertable on ts."""

    __tablename__ = "intraday_bar"

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instrument.id", ondelete="CASCADE"), primary_key=True
    )
    interval: Mapped[str] = mapped_column(String(16), primary_key=True)  # minute | 5minute ...
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))


class GlobalCueSnapshot(Base):
    """A point-in-time reading of an overnight cue, kept forever so a later prediction's
    feature snapshot can reference exactly what was known at the time."""

    __tablename__ = "global_cue_snapshot"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    key: Mapped[str] = mapped_column(String(32), index=True)  # e.g. sp500, crude_brent
    ticker: Mapped[str] = mapped_column(String(32))
    last: Mapped[float | None] = mapped_column(Float)
    prev_close: Mapped[float | None] = mapped_column(Float)
    change_pct: Mapped[float | None] = mapped_column(Float)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    source: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # ok | stale | unavailable
    note: Mapped[str | None] = mapped_column(Text)


class FiiDiiFlow(Base):
    """FII/FPI and DII cash-market activity, INR crore."""

    __tablename__ = "fii_dii_flow"
    __table_args__ = (
        UniqueConstraint("trade_date", "category", "segment", name="uq_fii_dii_day_cat_seg"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    category: Mapped[str] = mapped_column(String(16))  # FII | DII
    segment: Mapped[str] = mapped_column(String(16), default="cash")
    buy_value_cr: Mapped[float | None] = mapped_column(Float)
    sell_value_cr: Mapped[float | None] = mapped_column(Float)
    net_value_cr: Mapped[float | None] = mapped_column(Float)
    provisional: Mapped[bool] = mapped_column(Boolean, default=True)
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MarketHoliday(Base):
    __tablename__ = "market_holiday"
    __table_args__ = (UniqueConstraint("holiday_date", "segment", name="uq_holiday_date_segment"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    holiday_date: Mapped[date] = mapped_column(Date, index=True)
    segment: Mapped[str] = mapped_column(String(8), default="CM")  # CM = equity cash market
    description: Mapped[str | None] = mapped_column(String(256))
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class JobRun(Base):
    """One execution of a scheduled job. Skipped days are recorded with a reason."""

    __tablename__ = "job_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64), index=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)  # IST calendar date
    status: Mapped[str] = mapped_column(String(16))  # running | success | failed | skipped
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, object] | None] = mapped_column(JSON)


class SourceStatus(Base):
    """Health and freshness of each external data source."""

    __tablename__ = "source_status"

    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))  # ok | degraded | unavailable | not_configured
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    data_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    hint: Mapped[str | None] = mapped_column(Text)  # e.g. which API key would replace it
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KVEntry(Base):
    """Key-value rows for hosts without Redis (see app/kv.py)."""

    __tablename__ = "kv_entry"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class KiteSession(Base):
    """Daily Kite access token (Fernet-encrypted). Expires ~06:00 IST next day."""

    __tablename__ = "kite_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kite_user_id: Mapped[str] = mapped_column(String(32))
    access_token_enc: Mapped[str] = mapped_column(Text)
    login_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --------------------------------------------------------------------- Phase 2


class ModelVersion(Base):
    """A trained model (or a baseline) and its walk-forward, out-of-sample metrics.

    The artifact is stored here (LightGBM `model_to_string`), because GitHub Actions
    runners keep no files between runs. Exactly one champion per (name, target).
    """

    __tablename__ = "model_version"
    __table_args__ = (
        UniqueConstraint("name", "target", "version", name="uq_model_version_name_target_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(32), index=True)  # "market_call"
    target: Mapped[str] = mapped_column(String(32))  # "NIFTY 50" | "NIFTY BANK"
    version: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))  # lightgbm | baseline
    params: Mapped[dict | None] = mapped_column(JSON)
    feature_names: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    artifact: Mapped[str | None] = mapped_column(Text)
    trained_through: Mapped[date] = mapped_column(Date)
    train_rows: Mapped[int] = mapped_column(Integer)
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promotion_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    """One immutable ledger record (brief §5). Rows are never updated or deleted: a
    database trigger (migration 0003) rejects both. Corrections are new rows that set
    `corrects_prediction_id`. `hash` covers the record and `prev_hash`, so the chain
    (ordered by `seq`) makes any later edit detectable; see app/services/ledger.py.
    """

    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, unique=True)
    kind: Mapped[str] = mapped_column(String(32))  # "market_call"
    target: Mapped[str] = mapped_column(String(32), index=True)
    session_date: Mapped[date] = mapped_column(Date, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_late: Mapped[bool] = mapped_column(Boolean, default=False)  # written after 09:15 IST
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"))
    reference_date: Mapped[date] = mapped_column(Date)
    reference_close: Mapped[float] = mapped_column(Float)
    prob_up: Mapped[float] = mapped_column(Float)
    prob_flat: Mapped[float] = mapped_column(Float)
    prob_down: Mapped[float] = mapped_column(Float)
    predicted_class: Mapped[str] = mapped_column(String(8))  # UP | FLAT | DOWN
    range_p10_pct: Mapped[float | None] = mapped_column(Float)
    range_p90_pct: Mapped[float | None] = mapped_column(Float)
    gap_call: Mapped[str | None] = mapped_column(String(8))  # gap_up | gap_down | flat
    gap_expected_pct: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(8))  # high | medium | low | none
    no_edge: Mapped[bool] = mapped_column(Boolean, default=False)
    drivers: Mapped[list] = mapped_column(JSON)
    timing_guidance: Mapped[str | None] = mapped_column(Text)
    baselines: Mapped[dict] = mapped_column(JSON)
    feature_snapshot: Mapped[dict] = mapped_column(JSON)
    corrects_prediction_id: Mapped[int | None] = mapped_column(ForeignKey("prediction.id"))
    note: Mapped[str | None] = mapped_column(Text)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PredictionGrade(Base):
    """The outcome of a prediction, written after the fact (mutable: the open grade comes
    at 09:20, the close grade at 15:45; re-grading overwrites)."""

    __tablename__ = "prediction_grade"
    __table_args__ = (UniqueConstraint("prediction_id", name="uq_prediction_grade_prediction"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("prediction.id"), index=True)
    graded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    actual_open: Mapped[float | None] = mapped_column(Float)
    actual_close: Mapped[float | None] = mapped_column(Float)
    actual_pct: Mapped[float | None] = mapped_column(Float)
    actual_class: Mapped[str | None] = mapped_column(String(8))
    correct: Mapped[bool | None] = mapped_column(Boolean)
    brier: Mapped[float | None] = mapped_column(Float)
    log_loss: Mapped[float | None] = mapped_column(Float)
    gap_actual_pct: Mapped[float | None] = mapped_column(Float)
    gap_correct: Mapped[bool | None] = mapped_column(Boolean)
    gap_graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    baseline_correct: Mapped[dict | None] = mapped_column(JSON)
    error_category: Mapped[str | None] = mapped_column(String(32))  # Phase 4
    notes: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32))


# --------------------------------------------------------------------- Phase 3


class Idea(Base):
    """One immutable stock-idea record (brief §4.2, §5): its own hash chain ("idea"), the
    same append-only trigger as `prediction`. Outcomes live in `idea_outcome`."""

    __tablename__ = "idea"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, unique=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    session_date: Mapped[date] = mapped_column(Date, index=True)  # first session to enter
    is_late: Mapped[bool] = mapped_column(Boolean, default=False)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instrument.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    exchange: Mapped[str] = mapped_column(String(16))
    horizon: Mapped[str] = mapped_column(String(16))  # swing | positional
    horizon_days: Mapped[int] = mapped_column(Integer)
    setup: Mapped[str] = mapped_column(String(32))  # breakout | momentum | mean_reversion | ...
    direction: Mapped[str] = mapped_column(String(8), default="long")
    entry_low: Mapped[float] = mapped_column(Float)
    entry_high: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target_1: Mapped[float] = mapped_column(Float)
    target_2: Mapped[float | None] = mapped_column(Float)
    p_target: Mapped[float] = mapped_column(Float)  # P(target_1 before stop within horizon)
    expected_return_low_pct: Mapped[float] = mapped_column(Float)
    expected_return_high_pct: Mapped[float] = mapped_column(Float)
    risk_reward: Mapped[float] = mapped_column(Float)
    ev_pct_after_costs: Mapped[float] = mapped_column(Float)
    position_qty: Mapped[int] = mapped_column(Integer)
    position_value: Mapped[float] = mapped_column(Float)
    risk_amount: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON)
    invalidation: Mapped[str] = mapped_column(Text)
    drivers: Mapped[list] = mapped_column(JSON)
    feature_snapshot: Mapped[dict] = mapped_column(JSON)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"))
    strategy: Mapped[str] = mapped_column(String(32))
    rank: Mapped[int] = mapped_column(Integer)
    surveillance_unknown: Mapped[bool] = mapped_column(Boolean, default=True)
    corrects_idea_id: Mapped[int | None] = mapped_column(ForeignKey("idea.id"))
    note: Mapped[str | None] = mapped_column(Text)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IdeaOutcome(Base):
    """What happened to an idea (mutable; one row per idea). Statuses: pending (not yet
    entered), open, target, stop, expired, invalidated (gapped past the zone), void."""

    __tablename__ = "idea_outcome"
    __table_args__ = (UniqueConstraint("idea_id", name="uq_idea_outcome_idea"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    idea_id: Mapped[int] = mapped_column(ForeignKey("idea.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    entry_date: Mapped[date | None] = mapped_column(Date)
    entry_price: Mapped[float | None] = mapped_column(Float)
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(Float)
    gross_return_pct: Mapped[float | None] = mapped_column(Float)
    costs_pct: Mapped[float | None] = mapped_column(Float)
    net_return_pct: Mapped[float | None] = mapped_column(Float)
    r_multiple: Mapped[float | None] = mapped_column(Float)
    days_held: Mapped[int | None] = mapped_column(Integer)
    max_favorable_pct: Mapped[float | None] = mapped_column(Float)
    max_adverse_pct: Mapped[float | None] = mapped_column(Float)
    graded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    error_category: Mapped[str | None] = mapped_column(String(32))  # Phase 4


class PaperTrade(Base):
    """A simulated position with costs (brief §7.10). Auto-opened from ideas; manual ones
    have idea_id NULL. Never a real order."""

    __tablename__ = "paper_trade"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    idea_id: Mapped[int | None] = mapped_column(ForeignKey("idea.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    exchange: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(8), default="buy")
    qty: Mapped[int] = mapped_column(Integer)
    entry_date: Mapped[date] = mapped_column(Date)
    entry_price: Mapped[float] = mapped_column(Float)
    stop: Mapped[float | None] = mapped_column(Float)
    target: Mapped[float | None] = mapped_column(Float)
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(8), default="open", index=True)
    gross_pnl: Mapped[float | None] = mapped_column(Float)
    costs: Mapped[float | None] = mapped_column(Float)
    net_pnl: Mapped[float | None] = mapped_column(Float)
    r_multiple: Mapped[float | None] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PaperEquity(Base):
    """Daily paper-portfolio snapshot for the equity curve. `nifty_close` is the Nifty 50
    price index (the total-return index is not available free; the UI says so)."""

    __tablename__ = "paper_equity"

    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    invested: Mapped[float] = mapped_column(Float)
    open_positions: Mapped[int] = mapped_column(Integer)
    nifty_close: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))

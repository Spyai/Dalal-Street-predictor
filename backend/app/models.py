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

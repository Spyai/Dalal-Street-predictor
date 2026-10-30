"""Builds provider instances from settings. The only place that knows concrete classes."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache import get_kv, wait_for_slot
from app.config import get_settings
from app.models import Instrument
from app.providers.base import MarketDataProvider, NotConfigured, ProviderError
from app.providers.http import PoliteClient, RedisCooldownStore
from app.providers.kite import KiteProvider
from app.providers.nse import NSEProvider
from app.providers.yfinance_provider import YFinanceProvider
from app.services import kite_auth
from app.services.source_status import mark_error


def nse_provider() -> NSEProvider:
    s = get_settings()
    r = get_kv()
    client = PoliteClient(
        source="nse",
        user_agent=s.nse_user_agent,
        min_interval_seconds=s.public_source_min_interval_seconds,
        timeout_seconds=s.public_source_timeout_seconds,
        slot_waiter=lambda key, interval: wait_for_slot(key, interval, r),
        cooldown_store=RedisCooldownStore(r),
    )
    return NSEProvider(client)


def yfinance_provider() -> YFinanceProvider:
    return YFinanceProvider()


def kite_provider(db: Session) -> KiteProvider:
    s = get_settings()
    if s.broker != "kite":
        raise NotConfigured("kite", f"BROKER={s.broker}; only 'kite' is implemented")
    client = kite_auth.authenticated_client(db, s)

    def lookup(symbol: str, exchange: str) -> int | None:
        return db.scalar(
            select(Instrument.kite_token).where(
                Instrument.symbol == symbol, Instrument.exchange == exchange
            )
        )

    return KiteProvider(client, lookup)


def market_data_chain(db: Session, *, include_broker: bool = True) -> list[MarketDataProvider]:
    """Broker first (if configured and logged in), then yfinance as EOD fallback."""
    chain: list[MarketDataProvider] = []
    if include_broker:
        try:
            chain.append(kite_provider(db))
        except ProviderError as exc:
            mark_error("kite", exc)
    chain.append(yfinance_provider())
    return chain

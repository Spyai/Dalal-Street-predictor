"""Stock page data: any NSE/BSE symbol, daily candles, indicators, support/resistance."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import current_user, kv_dep
from app.cache import KV
from app.config import get_settings
from app.db import get_db
from app.models import AppUser
from app.providers import registry
from app.schemas import CandleOut, InstrumentOut, LevelOut, SeriesPoint, StockOut
from app.services import indicators as ind
from app.services import instruments
from app.services.calendar import TradingCalendar
from app.services.market_data import daily_frame, ensure_recent

router = APIRouter(prefix="/api/stocks", tags=["stocks"])

RANGES = {"6m": 183, "1y": 365, "3y": 365 * 3, "5y": 365 * 5, "max": 365 * 30}
WARMUP_DAYS = 320  # calendar days (~220 sessions) so SMA200 exists from the first visible bar
REFETCH_COOLDOWN = 15 * 60  # don't re-hit providers for the same symbol more often


def _inst_out(i: instruments.Instrument) -> InstrumentOut:
    return InstrumentOut(
        symbol=i.symbol,
        exchange=i.exchange,
        name=i.name,
        industry=i.industry,
        in_universe=i.in_universe,
        is_index=i.is_index,
    )


@router.get("/search", response_model=list[InstrumentOut])
def search(
    q: str = Query(min_length=1, max_length=40),
    db: Session = Depends(get_db),
    _: AppUser = Depends(current_user),
) -> list[InstrumentOut]:
    return [_inst_out(i) for i in instruments.search(db, q)]


def _series(s: pd.Series) -> list[SeriesPoint]:
    s = s.dropna()
    idx = pd.DatetimeIndex(s.index)
    return [
        SeriesPoint(time=ts.date(), value=round(float(v), 4))
        for ts, v in zip(idx, s.to_numpy(), strict=True)
    ]


@router.get("/{exchange}/{symbol}", response_model=StockOut)
def stock(
    exchange: str,
    symbol: str,
    range_: str = Query(default="1y", alias="range"),
    db: Session = Depends(get_db),
    r: KV = Depends(kv_dep),
    _: AppUser = Depends(current_user),
) -> StockOut:
    exchange, symbol = exchange.upper(), symbol.upper().strip()
    if exchange not in {"NSE", "BSE", "INDEX"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "exchange must be NSE, BSE or INDEX")
    if range_ not in RANGES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"range must be one of {list(RANGES)}")

    inst = instruments.get_instrument(db, symbol, exchange)
    created = False
    if inst is None:
        if exchange == "INDEX":
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown index {symbol}")
        inst = instruments.get_or_create_equity(db, symbol, exchange)
        created = True

    cal = TradingCalendar.load(db)
    expected = cal.last_completed_session(datetime.now(UTC))
    notes: list[str] = []
    fetch_key = f"stockfetch:{exchange}:{symbol}"
    if r.set(fetch_key, "1", nx=True, ex=REFETCH_COOLDOWN):
        source, errors = ensure_recent(
            db, inst, expected, registry.market_data_chain(db), get_settings().backfill_years
        )
        if source is None and errors:
            notes.extend(errors)
            r.set(f"{fetch_key}:errors", "\n".join(errors), ex=REFETCH_COOLDOWN)
    else:
        cached = r.get(f"{fetch_key}:errors")
        if cached:
            notes.extend(str(cached).split("\n"))

    start = date.today() - timedelta(days=RANGES[range_] + WARMUP_DAYS)
    df = daily_frame(db, inst.id, start)
    if df.empty:
        if created:
            db.delete(inst)
            db.flush()
        detail = "; ".join(notes) or "no data from any source"
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No price data for {exchange}:{symbol} ({detail})"
        )

    close = df["close"]
    upper, mid, lower = ind.bollinger(close)
    macd_line, macd_sig, macd_hist = ind.macd(close)
    series = {
        "sma20": ind.sma(close, 20),
        "sma50": ind.sma(close, 50),
        "sma200": ind.sma(close, 200),
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "rsi14": ind.rsi(close, 14),
        "macd": macd_line,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
    }
    visible_from = pd.Timestamp(date.today() - timedelta(days=RANGES[range_]))
    vis = df[df.index >= visible_from]
    if vis.empty:
        vis = df.tail(1)
    indicators_out = {k: _series(v[v.index >= vis.index[0]]) for k, v in series.items()}
    one_year = df[df.index >= pd.Timestamp(date.today() - timedelta(days=365))]
    levels = ind.support_resistance(one_year if len(one_year) > 30 else df)

    latest = df.index[-1].date()
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else None
    if latest < expected:
        notes.insert(0, f"Latest bar is {latest}; expected {expected}.")
    return StockOut(
        instrument=_inst_out(inst),
        candles=[
            CandleOut(
                time=pd.Timestamp(d).date(),  # type: ignore[arg-type]
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=None if pd.isna(row.volume) else row.volume,
                delivery_pct=None if pd.isna(row.delivery_pct) else row.delivery_pct,
            )
            for d, row in vis.iterrows()
        ],
        indicators=indicators_out,
        levels=[LevelOut(**lv) for lv in levels],  # type: ignore[arg-type]
        last_close=last_close,
        prev_close=prev_close,
        change_pct=(last_close / prev_close - 1) * 100 if prev_close else None,
        data_as_of=latest,
        data_status="ok" if latest >= expected else "stale",
        sources=sorted(set(vis["source"].dropna())),
        notes=notes,
    )

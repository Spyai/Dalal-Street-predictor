"""Phase 1 job implementations."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from app.jobs.runner import JobContext, JobResult, JobSkipped
from app.models import DailyBar, Instrument
from app.providers import registry
from app.providers.base import AuthExpired, NotConfigured, ProviderError
from app.services import calendar, flows, global_cues, instruments, kite_auth, telegram
from app.services.backfill import run_backfill
from app.services.market_data import ingest_bhavcopy
from app.services.source_status import mark_error, mark_ok


def _broker_or_none(ctx: JobContext) -> registry.MarketDataProvider | None:
    try:
        return registry.kite_provider(ctx.db)
    except ProviderError as exc:
        mark_error("kite", exc)
        return None


# ----------------------------------------------------------------------------- jobs


def holiday_refresh(ctx: JobContext) -> JobResult:
    imported = 0
    for path in sorted(Path(ctx.settings.imports_dir).glob("nse_holidays*.csv")):
        imported += calendar.import_holiday_csv(ctx.db, path)
    try:
        n = calendar.refresh_from_exchange(ctx.db, registry.nse_provider())
        mark_ok("nse")
    except ProviderError as exc:
        mark_error("nse", exc)
        cov = calendar.coverage(ctx.db)
        if cov.get(ctx.today.year):
            return JobResult(
                "partial",
                f"NSE holiday API unavailable ({exc.reason}); using stored calendar "
                f"({cov[ctx.today.year]} holidays for {ctx.today.year})",
            )
        raise RuntimeError(
            f"No holiday calendar for {ctx.today.year} and NSE is unavailable ({exc.reason}). "
            f"Put nse_holidays_{ctx.today.year}.csv (date,description) in imports/."
        ) from exc
    return JobResult("success", f"{n} NSE holidays stored; {imported} imported from CSV")


def kite_login_reminder(ctx: JobContext) -> JobResult:
    if not ctx.settings.kite_configured:
        raise JobSkipped("Kite API key/secret not configured")
    sess = kite_auth.active_session(ctx.db)
    if sess is not None:
        return JobResult("success", f"Kite session valid until {sess.expires_at.isoformat()}")
    url = kite_auth.login_url(ctx.settings)
    sent = telegram.send(
        "🔑 <b>Kite login needed</b> for today's 07:30 run.\n"
        f'<a href="{url}">Log in to Kite</a> (opens Zerodha, then returns to Dalal).'
    )
    return JobResult(
        "partial",
        "no Kite session yet; " + ("reminder sent" if sent else "Telegram not configured"),
    )


def global_cues_refresh(ctx: JobContext) -> JobResult:
    broker = _broker_or_none(ctx)
    snaps = global_cues.refresh(ctx.db, registry.yfinance_provider(), broker, now=ctx.now)
    missing = [s.key for s in snaps if s.status == "unavailable" and s.key != "gift_nifty"]
    stale = [s.key for s in snaps if s.status == "stale"]
    msg = f"{sum(s.status == 'ok' for s in snaps)} cues ok"
    if stale:
        msg += f"; stale: {', '.join(stale)}"
    if missing:
        msg += f"; unavailable: {', '.join(missing)}"
    return JobResult("partial" if missing else "success", msg)


def instrument_sync(ctx: JobContext) -> JobResult:
    details: dict[str, object] = {}
    instruments.ensure_reference_instruments(ctx.db)
    try:
        kite = registry.kite_provider(ctx.db)
        details["kite_instruments"] = instruments.sync_broker_instruments(ctx.db, kite)
        mark_ok("kite")
    except (NotConfigured, AuthExpired) as exc:
        details["kite_instruments"] = f"skipped: {exc.reason}"
    except ProviderError as exc:
        mark_error("kite", exc)
        details["kite_instruments"] = f"failed: {exc.reason}"
    try:
        n, src = instruments.sync_universe(ctx.db, registry.nse_provider())
        details["universe"] = f"{n} symbols from {src}"
    except ProviderError as exc:
        mark_error("nse", exc)
        details["universe"] = f"unchanged; NSE unavailable: {exc.reason}"
    ok = all(not str(v).startswith(("failed", "unchanged")) for v in details.values())
    return JobResult(
        "success" if ok else "partial", "; ".join(f"{k}: {v}" for k, v in details.items()), details
    )


def evening_data_pull(ctx: JobContext) -> JobResult:
    """19:30: FII/DII, bhavcopy + delivery %, reference series, universe gap-fill."""
    session_day = ctx.calendar.last_completed_session(ctx.now)
    steps: dict[str, str] = {}
    nse = registry.nse_provider()

    try:
        n = flows.refresh(ctx.db, nse)
        steps["fii_dii"] = f"ok ({n} rows)"
        mark_ok("nse")
    except ProviderError as exc:
        mark_error("nse", exc)
        steps["fii_dii"] = f"unavailable: {exc.reason}"

    try:
        records = nse.bhavcopy(session_day)
        stored, unknown = ingest_bhavcopy(ctx.db, records)
        steps["bhavcopy"] = f"ok ({stored} bars for {session_day}, {unknown} unmapped symbols)"
        mark_ok("nse")
    except ProviderError as exc:
        mark_error("nse", exc)
        steps["bhavcopy"] = f"unavailable: {exc.reason}"
    ctx.db.commit()

    chain = registry.market_data_chain(ctx.db)
    ref_ids = list(
        ctx.db.scalars(select(Instrument.id).where(Instrument.exchange.in_(["INDEX", "GLOBAL"])))
    )
    rep = run_backfill(ref_ids, session_day - timedelta(days=10), session_day, chain)
    steps["reference_series"] = rep.summary()

    have = set(
        ctx.db.scalars(select(DailyBar.instrument_id).where(DailyBar.trade_date == session_day))
    )
    gaps = [i.id for i in instruments.universe_instruments(ctx.db) if i.id not in have]
    if gaps:
        rep = run_backfill(gaps, session_day - timedelta(days=10), session_day, chain)
        steps["universe_gap_fill"] = rep.summary()
    else:
        steps["universe_gap_fill"] = "nothing missing"
    steps["not_built_yet"] = "F&O OI, participant OI, bulk/block deals (Phase 2)"

    degraded = any("unavailable" in v or "no data" in v for v in steps.values())
    return JobResult(
        "partial" if degraded else "success",
        "; ".join(f"{k}: {v}" for k, v in steps.items()),
        {"session": session_day.isoformat(), **steps},
    )

"""Job implementations: Phase 1 data jobs and the Phase 2 market-call loop.

Phase 2 (brief §3, §4.1, §5): `premarket_prediction` at 07:30 makes and locks the call,
`open_check` at 09:20 grades the gap call, `close_grading` at 15:45 grades the close, and
`weekly_retrain` on Saturday trains a challenger and promotes it only when it beats the
champion out of sample. Every job returns per-target details for /health, says "partial"
whenever a number it wanted was missing, and never substitutes a guess for it.

Honest skips beat mis-referenced records: the 07:30 job refuses to write a call when the
previous session's target bar is missing, when the session's year has no holiday calendar
(unless forced), or when it runs on the reference session itself before 18:30 IST, and it
says so in the run details instead. Global cues live on other exchanges, so a missing bar
on one of their holidays is a note, not a problem, and their bars are only ever fetched
through the last complete UTC day, then re-requested at 07:30 so a provisional print is
replaced by the final close.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import IST, Settings
from app.jobs.runner import JobContext, JobResult, JobSkipped
from app.ml import baselines, core, features, models
from app.ml import registry as model_registry
from app.ml.core import MarketCallRecordIn, finite
from app.models import DailyBar, Instrument, Prediction, PredictionGrade
from app.providers import registry
from app.providers.base import AuthExpired, MarketDataProvider, NotConfigured, ProviderError
from app.services import (
    brief,
    calendar,
    flows,
    global_cues,
    grading,
    instruments,
    kite_auth,
    ledger,
    scorecard,
    telegram,
)
from app.services.backfill import run_backfill
from app.services.market_data import (
    ensure_recent,
    ingest_bhavcopy,
    last_final_global_day,
    latest_bar_date,
)
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
    index_ids = list(ctx.db.scalars(select(Instrument.id).where(Instrument.exchange == "INDEX")))
    rep = run_backfill(index_ids, session_day - timedelta(days=10), session_day, chain)
    steps["reference_series"] = rep.summary()
    # Global cues trade while this job runs (the US session is open at 19:30 IST): fetch
    # them only through the last complete UTC day, never the run day. The 07:30 job
    # re-requests the last days and the same-source upsert replaces any earlier print.
    global_ids = list(ctx.db.scalars(select(Instrument.id).where(Instrument.exchange == "GLOBAL")))
    global_through = min(session_day, last_final_global_day(ctx.now))
    rep = run_backfill(global_ids, global_through - timedelta(days=10), global_through, chain)
    steps["global_cues"] = f"{rep.summary()} (through {global_through}, the last complete UTC day)"

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


# ----------------------------------------------------------------------------- Phase 2

TRAIN_HINT = "no model: run `python -m app.cli train`"
RETRAIN_HINT = "retrain needed (run `python -m app.cli train` or wait for Saturday)"
# Calendar days of GLOBAL-cue history re-requested at 07:30 so a bar first stored while
# its market was open is replaced by the final one (market_data.ensure_recent).
GLOBAL_REFETCH_DAYS = 3
# The previous session's own close may still be revised by the exchange / the vendor for
# a while after 15:30 IST; a call written on that day before this time would lock in a
# provisional reference close.
REFERENCE_FINAL_TIME = time(18, 30)
# Ungraded calls older than this many sessions will not resolve on their own: list them
# as unresolvable (void them with `grade --void`) instead of alerting every afternoon.
UNRESOLVED_SESSIONS = 5
GAP_GUIDANCE: dict[str, str] = {
    "gap_up": "Gap-up likely; avoid buying the first 15 minutes.",
    "gap_down": "Gap-down likely; avoid selling into the first 15 minutes.",
    "flat": "Flat open likely; let the first 15 minutes set the range.",
}
# 10-day realised volatility this far above / below its 20-day level reads as a regime.
VOL_EXPANDING = 1.25
VOL_CONTRACTING = 0.75


def timing_guidance(gap_call: str | None, row: Mapping[str, float | None]) -> str | None:
    """Rule-based entry-timing text (brief §4.1), at most two sentences; None without a gap call.

    Transparent on purpose: the gap sentence follows the gap call, the second sentence
    compares 10- and 20-day realised volatility. Phase 4 grades this against a naive rule.
    """
    if gap_call is None:
        return None
    first = GAP_GUIDANCE.get(gap_call)
    if first is None:
        return None
    rv10, rv20 = finite(row.get("rvol_10d")), finite(row.get("rvol_20d"))
    second: str | None = None
    if rv10 is not None and rv20 is not None and rv20 > 0:
        if rv10 >= VOL_EXPANDING * rv20:
            second = "Trend day probable (10-day volatility well above its 20-day level)."
        elif rv10 <= VOL_CONTRACTING * rv20:
            second = "Range day likely (10-day volatility well below its 20-day level)."
    return first if second is None else f"{first} {second}"


def _reference_instruments(db: Session, targets: list[str]) -> list[Instrument]:
    """The prediction targets, India VIX and every global cue, in that order."""
    out: list[Instrument] = []
    for symbol in [*targets, features.VIX_SYMBOL]:
        inst = instruments.get_instrument(db, symbol, "INDEX")
        if inst is not None:
            out.append(inst)
    out += list(
        db.scalars(
            select(Instrument).where(Instrument.exchange == "GLOBAL").order_by(Instrument.symbol)
        )
    )
    return out


@dataclass
class RefreshReport:
    """What the pre-run data refresh found. `problems` make the run partial (a target or
    VIX bar is missing); `notes` are reported but harmless (a global cue's exchange was
    closed, or its re-fetch failed and the stored bar stands)."""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _refresh_bars(
    db: Session,
    insts: list[Instrument],
    through: date,
    chain: list[MarketDataProvider],
    years: int,
    *,
    session: date,
    now: datetime,
) -> RefreshReport:
    """Best-effort `ensure_recent` for each instrument, then say what is still missing.

    INDEX instruments (targets, India VIX) must have a bar for `through`; a missing one is
    a problem the caller reports (never fills in). GLOBAL instruments follow their own
    calendars: they are fetched only through the last complete UTC day, the last
    `GLOBAL_REFETCH_DAYS` are re-requested so a provisional bar is replaced, and a missing
    bar is only a problem once the latest bar is older than `features.MAX_CUE_AGE_DAYS`
    (the age at which the feature pipeline treats the cue as missing anyway).
    """
    report = RefreshReport()
    global_through = min(through, last_final_global_day(now))
    for inst in insts:
        is_global = inst.exchange == "GLOBAL"
        want = global_through if is_global else through
        _, errors = ensure_recent(
            db,
            inst,
            want,
            chain,
            years,
            refetch_days=GLOBAL_REFETCH_DAYS if is_global else 0,
        )
        latest = latest_bar_date(db, inst.id)
        name = f"{inst.exchange}:{inst.symbol}"
        why = errors[-1] if errors else "no bars returned"
        if latest is not None and latest >= want:
            if errors:
                report.notes.append(
                    f"{name}: re-fetch returned nothing ({why}); the stored bar for {want} stands"
                )
            continue
        if not is_global:
            report.problems.append(f"{name} has no bar for {want}: {why}")
        elif latest is not None and (session - latest).days <= features.MAX_CUE_AGE_DAYS:
            report.notes.append(f"{name}: no bar for {want} (market closed?); latest is {latest}")
        else:
            report.problems.append(
                f"{name} has no bar for {want} and its latest bar ({latest or 'none'}) is older "
                f"than {features.MAX_CUE_AGE_DAYS} days: {why}"
            )
    return report


def _class_frequencies(mv: Any) -> dict[str, float]:
    raw = (mv.metrics or {}).get("class_frequencies") if mv is not None else None
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): v for k, v in raw.items() if finite(v) is not None}


def _call_details(p: Prediction, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "seq": p.seq,
        "class": p.predicted_class,
        "confidence": p.confidence,
        "no_edge": p.no_edge,
        "is_late": p.is_late,
        "probs": {"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down},
        "gap_call": p.gap_call,
        "reference_date": p.reference_date.isoformat(),
        **extra,
    }


def _call_summary(p: Prediction) -> str:
    if p.no_edge:
        return f"{p.target}: No edge today (leans {p.predicted_class})"
    top = {"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down}[p.predicted_class]
    late = ", late" if p.is_late else ""
    return f"{p.target}: {p.predicted_class} {top:.0%} ({p.confidence}{late})"


@dataclass
class TargetOutcome:
    """One target's result in the 07:30 run."""

    row: Prediction | None
    details: dict[str, Any]
    problems: list[str] = field(default_factory=list)
    recorded: bool = False  # a new ledger record was written in this run


def _skipped(target: str, status: str, problem: str | None = None) -> TargetOutcome:
    return TargetOutcome(None, {"status": status}, [problem] if problem else [])


def _names(items: list[str], limit: int = 5) -> str:
    return ", ".join(items[:limit]) + ("..." if len(items) > limit else "")


def recording_block(ctx: JobContext, session: date, reference: date) -> str | None:
    """Why no call may be recorded for `session` right now, or None if recording is fine.

    - The session's year has no NSE holiday rows: `session` itself may be a holiday, so
      the call would be for a day that never trades. `--force` overrides (the operator
      has checked), and the run record says so.
    - The run happens on the reference session itself before REFERENCE_FINAL_TIME: the
      just-closed bar may still be provisional, and the ledger is immutable.
    """
    if not ctx.calendar.info(session).calendar_verified and not ctx.force:
        return (
            f"NSE holiday calendar for {session.year} not loaded, so {session} may not be a "
            f"trading day; import nse_holidays_{session.year}.csv or run with --force"
        )
    now_ist = ctx.now.astimezone(IST)
    if now_ist.date() == reference and now_ist.time() < REFERENCE_FINAL_TIME:
        return (
            f"run at {now_ist:%H:%M} IST on the reference session {reference}; its close may "
            f"still be provisional before {REFERENCE_FINAL_TIME:%H:%M} IST, rerun after that"
        )
    return None


def _predict_target(db: Session, target: str, session: date, ctx: JobContext) -> TargetOutcome:
    """Make and record one target's call; never records against a stale reference."""
    s = ctx.settings
    champion = model_registry.champion(db, target)
    if champion is None or not champion.artifact:
        return _skipped(target, TRAIN_HINT, f"{target}: {TRAIN_HINT}")
    try:
        row, ref_date, ref_close = features.live_row(db, target, session, settings=s)
    except LookupError as exc:
        return _skipped(target, f"no features: {exc}", f"{target}: {exc}")
    expected_ref = ctx.calendar.previous_trading_day(session)
    if ref_date != expected_ref:
        why = (
            f"previous session bar missing ({expected_ref}); call not recorded rather than "
            f"referenced to {ref_date}"
        )
        return _skipped(target, f"not recorded: {why}", f"{target}: {why}")
    trained_names = [str(f) for f in champion.feature_names]
    missing = [f for f in trained_names if f not in row]
    if missing:
        why = (
            f"{RETRAIN_HINT}: model {champion.version} expects features the pipeline no "
            f"longer produces ({_names(missing)})"
        )
        return _skipped(target, f"not predicted: {why}", f"{target}: {why}")
    problems: list[str] = []
    extra = [f for f in row if f not in trained_names]
    if extra:
        problems.append(
            f"{target}: {RETRAIN_HINT}: the pipeline now produces features model "
            f"{champion.version} was not trained on ({_names(extra)}); predicted without them"
        )

    out = models.predict(champion.artifact, trained_names, row, settings=s)
    gap_pct = baselines.gap_expected_pct(row)
    gap_call = core.gap_class_of(gap_pct, s.gap_threshold_pct) if gap_pct is not None else None
    calls = baselines.baseline_calls(row, target, session, _class_frequencies(champion), s)
    rec = MarketCallRecordIn(
        target=target,
        session_date=session,
        as_of=ctx.now,
        model_version_id=champion.id,
        reference_date=ref_date,
        reference_close=ref_close,
        output=out,
        gap_call=gap_call,
        gap_expected_pct=gap_pct,
        timing_guidance=timing_guidance(gap_call, row),
        baselines=calls,
        feature_snapshot=row,
    )
    try:
        p = ledger.record_market_call(db, rec, settings=s)
    except ledger.LedgerConflict as conflict:
        # Another run recorded this session while we were computing: theirs stands.
        existing = conflict.existing
        return TargetOutcome(existing, _call_details(existing, "already recorded"))
    details = _call_details(
        p,
        "recorded",
        model_version=champion.version,
        range_p10_pct=p.range_p10_pct,
        range_p90_pct=p.range_p90_pct,
        gap_expected_pct=p.gap_expected_pct,
        drivers=[d["text"] for d in (p.drivers or []) if isinstance(d, dict) and d.get("text")],
    )
    return TargetOutcome(p, details, problems, recorded=True)


def premarket_prediction(ctx: JobContext) -> JobResult:
    """07:30: refresh inputs, make and lock the market call per target, send the brief.

    Data refresh and the FII/DII pull are best effort: a missing target or VIX bar is a
    problem (partial run), a global cue's closed market is a note. A call is recorded only
    against the previous session's own bar (see `recording_block` and `_predict_target`);
    otherwise the target's status says why it was not recorded. A target with no champion
    model is reported with the training hint; an existing call for the session is left
    alone (re-runs are idempotent, race-safe through the ledger lock, and do not re-send
    the brief).
    """
    s = ctx.settings
    db = ctx.db
    session = core.session_for(ctx.calendar, ctx.now)
    through = ctx.calendar.previous_trading_day(session)
    targets = s.prediction_target_symbols
    details: dict[str, Any] = {"session": session.isoformat(), "reference_session": str(through)}
    problems: list[str] = []

    chain = registry.market_data_chain(db, include_broker=False)
    refresh = _refresh_bars(
        db,
        _reference_instruments(db, targets),
        through,
        chain,
        s.backfill_years,
        session=session,
        now=ctx.now,
    )
    details["refresh"] = refresh.problems or "ok"
    if refresh.notes:
        details["refresh_notes"] = refresh.notes
    problems += refresh.problems
    try:
        n = flows.refresh(db, registry.nse_provider())
        details["fii_dii"] = f"ok ({n} rows)"
        mark_ok("nse")
    except ProviderError as exc:
        mark_error("nse", exc)
        details["fii_dii"] = f"unavailable: {exc.reason}"
    db.commit()  # keep what was fetched even if the prediction step fails and retries

    blocked = recording_block(ctx, session, through)
    if blocked is not None:
        problems.append(f"call not recorded: {blocked}")
    elif ctx.force and not ctx.calendar.info(session).calendar_verified:
        details["calendar"] = f"unverified for {session.year}; recorded anyway (--force)"

    current: list[Prediction] = []
    recorded = 0
    for target in targets:
        # Fast path for re-runs; the race between two live runs is closed inside
        # ledger.record_market_call (LedgerConflict under the ledger lock).
        existing = ledger.latest_call(db, target, session)
        if existing is not None:
            details[target] = _call_details(existing, "already recorded")
            current.append(existing)
            continue
        if blocked is not None:
            details[target] = {"status": f"not recorded: {blocked}"}
            continue
        outcome = _predict_target(db, target, session, ctx)
        details[target] = outcome.details
        problems += outcome.problems
        if outcome.row is not None:
            current.append(outcome.row)
            recorded += int(outcome.recorded)
    db.commit()  # the ledger rows are final before anything is announced

    if recorded:
        cards: list[tuple[Prediction, scorecard.Scorecard | None]] = [
            (p, scorecard.scorecard(db, p.target, settings=s)) for p in current
        ]
        text = brief.morning_brief(cards, session, s, head=ledger.head(db))
        sent = telegram.send(text)
        details["brief"] = "sent" if sent else "not sent (Telegram not configured or failed)"
    else:
        details["brief"] = "not sent (nothing new recorded)"

    summary = [_call_summary(p) for p in current] or [f"no call recorded for {session}"]
    message = "; ".join([*summary, *problems])
    return JobResult("partial" if problems else "success", message, details)


def _session_open(
    chain: list[MarketDataProvider], symbol: str, session: date
) -> tuple[float, str] | None:
    """The session's opening print from the first provider that has a bar dated `session`.

    The bar is partial (the session is running) and is NOT stored: a provisional close in
    `daily_bar` would later pass for the real one. A quote's last price is not an open and
    is never used as one.
    """
    for provider in chain:
        try:
            bars = provider.daily_bars(symbol, "INDEX", session, session)
        except ProviderError as exc:
            mark_error(provider.name, exc)
            continue
        for bar in bars:
            open_ = finite(bar.open)
            if bar.trade_date == session and open_ is not None and open_ > 0:
                mark_ok(provider.name)
                return open_, provider.name
    return None


def _grade_row(db: Session, prediction_id: int) -> PredictionGrade | None:
    return db.scalars(
        select(PredictionGrade).where(PredictionGrade.prediction_id == prediction_id)
    ).first()


def open_check(ctx: JobContext) -> JobResult:
    """09:20: grade every gap call of the session against the actual open."""
    db = ctx.db
    session = core.session_for(ctx.calendar, ctx.now)
    calls = ledger.calls_for_session(db, session)
    if not calls:
        raise JobSkipped(f"no market call recorded for {session}")
    chain = registry.market_data_chain(db)
    details: dict[str, Any] = {"session": session.isoformat()}
    problems: list[str] = []
    opens: dict[str, tuple[float, str] | None] = {}
    graded = 0
    for p in calls:
        key = f"{p.target} seq {p.seq}"
        existing = _grade_row(db, p.id)
        if existing is not None and existing.gap_graded_at is not None:
            details[key] = {
                "status": "already graded",
                "gap_call": p.gap_call,
                "gap_actual_pct": existing.gap_actual_pct,
                "gap_correct": existing.gap_correct,
            }
            continue
        if p.target not in opens:
            opens[p.target] = _session_open(chain, p.target, session)
        found = opens[p.target]
        if found is None:
            details[key] = {"status": "open not available yet"}
            problems.append(f"{p.target}: open for {session} not available yet")
            continue
        open_, source = found
        g = grading.grade_gap(db, p, open_, now=ctx.now, source=source)
        graded += 1
        details[key] = {
            "status": "gap graded",
            "actual_open": open_,
            "gap_call": p.gap_call,
            "gap_expected_pct": p.gap_expected_pct,
            "gap_actual_pct": g.gap_actual_pct,
            "gap_correct": g.gap_correct,
            "source": source,
        }
    message = "; ".join([f"{graded} gap call(s) graded for {session}", *problems])
    return JobResult("partial" if problems else "success", message, details)


def _sessions_back(cal: calendar.TradingCalendar, session: date, n: int) -> date:
    """The session `n` trading days before `session`."""
    d = session
    for _ in range(n):
        d = cal.previous_trading_day(d)
    return d


def close_grading(ctx: JobContext) -> JobResult:
    """15:45: fetch the session close, grade every ungraded call that has one, report.

    A call still without a close within the last UNRESOLVED_SESSIONS sessions is a
    problem (partial run). Older ones will not resolve by themselves (the session never
    traded, or the bar is gone for good): they are listed under `unresolvable` with the
    void command, without turning every afternoon's run partial.
    """
    s = ctx.settings
    db = ctx.db
    session = ctx.calendar.last_completed_session(ctx.now)
    pending = ledger.ungraded(db, through=session)
    if not pending:
        raise JobSkipped(f"no ungraded market calls through {session}")
    details: dict[str, Any] = {"session": session.isoformat()}
    oldest_recent = _sessions_back(ctx.calendar, session, UNRESOLVED_SESSIONS - 1)

    chain = registry.market_data_chain(db)
    targets = sorted({p.target for p in pending})
    insts = {t: instruments.get_instrument(db, t, "INDEX") for t in targets}
    refresh = _refresh_bars(
        db,
        [i for i in insts.values() if i is not None],
        session,
        chain,
        s.backfill_years,
        session=session,
        now=ctx.now,
    )
    details["refresh"] = refresh.problems or "ok"
    db.commit()

    graded: list[tuple[Prediction, PredictionGrade]] = []
    remaining: list[str] = []
    unresolvable: list[str] = []
    for p in pending:
        inst = insts.get(p.target)
        bar = db.get(DailyBar, (inst.id, p.session_date)) if inst is not None else None
        if bar is None or finite(bar.close) is None:
            if p.session_date < oldest_recent:
                unresolvable.append(
                    f"{p.target} {p.session_date} (seq {p.seq}): no close after "
                    f"{UNRESOLVED_SESSIONS} sessions; void it with "
                    f"`python -m app.cli grade --void {p.seq} --reason ...`"
                )
            else:
                remaining.append(f"{p.target} {p.session_date} (seq {p.seq}): no close yet")
            continue
        g = grading.grade_close(
            db, p, bar.open, bar.close, now=ctx.now, source=bar.source, settings=s
        )
        graded.append((p, g))
    db.commit()

    details["graded"] = [
        {
            "target": p.target,
            "session_date": p.session_date.isoformat(),
            "seq": p.seq,
            "predicted_class": p.predicted_class,
            "actual_pct": g.actual_pct,
            "actual_class": g.actual_class,
            "correct": g.correct,
            "brier": g.brier,
            "gap_correct": g.gap_correct,
            "source": g.source,
        }
        for p, g in graded
    ]
    details["ungraded"] = remaining
    details["unresolvable"] = unresolvable
    if graded:
        sent = telegram.send(brief.close_summary(graded))
        details["summary"] = "sent" if sent else "not sent (Telegram not configured or failed)"
    message = f"{len(graded)} call(s) graded; {len(remaining)} still ungraded"
    if remaining:
        message += ": " + "; ".join(remaining)
    if unresolvable:
        message += (
            f"; {len(unresolvable)} unresolvable (older than {UNRESOLVED_SESSIONS} sessions, "
            "see details)"
        )
    return JobResult("partial" if remaining else "success", message, details)


def _retrain_target(db: Session, target: str, s: Settings) -> tuple[dict[str, Any], str | None]:
    """Train a challenger for one target; returns (details, problem or None)."""
    frame = features.build_frame(db, target, settings=s)
    if frame.empty:
        why = "no daily bars: run `python -m app.cli backfill`"
        return {"status": f"not trained: {why}"}, f"{target}: {why}"
    labelled = int(frame["label_pct"].notna().sum())
    try:
        result = models.train_walk_forward(frame, settings=s, target=target)
    except ValueError as exc:
        return {"status": f"not trained: {exc}", "labelled_rows": labelled}, f"{target}: {exc}"
    champ = model_registry.champion(db, target)
    promote, reason = model_registry.should_promote(champ, result, s)
    mv = model_registry.save_model(db, target, result, promote=promote, note=reason)
    db.commit()
    model_registry.log_to_mlflow(result, target, promote, s)
    m = result.metrics
    return {
        "status": "promoted" if promote else "rejected",
        "version": mv.version,
        "trained_through": result.trained_through.isoformat(),
        "train_rows": result.train_rows,
        "n_oos": m.n_oos,
        "oos_brier": finite(m.brier),
        "oos_accuracy": finite(m.accuracy),
        "oos_log_loss": finite(m.log_loss),
        "baselines": {k: {kk: finite(vv) for kk, vv in v.items()} for k, v in m.baselines.items()},
        "previous_champion": champ.version if champ is not None else None,
        "reason": reason,
    }, None


def weekly_retrain(ctx: JobContext) -> JobResult:
    """Saturday 10:00: walk-forward retrain per target; promote only a better challenger.

    Every version is saved (and logged to MLflow when configured) with the promotion or
    rejection reason, so the registry is the audit trail the brief asks for (§6.2).
    """
    s = ctx.settings
    details: dict[str, Any] = {}
    problems: list[str] = []
    summary: list[str] = []
    for target in s.prediction_target_symbols:
        target_details, problem = _retrain_target(ctx.db, target, s)
        details[target] = target_details
        if problem is not None:
            problems.append(problem)
            continue
        summary.append(
            f"{target}: {target_details['version']} {target_details['status']} "
            f"(OOS Brier {target_details['oos_brier']:.4f}, "
            f"accuracy {target_details['oos_accuracy']:.3f}, n={target_details['n_oos']})"
        )
    message = "; ".join([*summary, *problems])
    return JobResult("partial" if problems else "success", message, details)

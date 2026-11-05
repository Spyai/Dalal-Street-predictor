"""Job implementations: Phase 1 data jobs, the Phase 2 market-call loop and the Phase 3
stock-ideas loop that rides on the same three jobs.

Phase 2 (brief §3, §4.1, §5): `premarket_prediction` at 07:30 makes and locks the call,
`open_check` at 09:20 grades the gap call, `close_grading` at 15:45 grades the close, and
`weekly_retrain` on Saturday trains a challenger and promotes it only when it beats the
champion out of sample. Every job returns per-target details for /health, says "partial"
whenever a number it wanted was missing, and never substitutes a guess for it.

Phase 3 (brief §4.2, §5, §6.3): after the market calls, the 07:30 run refreshes the
universe's bars through the previous session, runs the rule-based strategies on the
point-in-time features of that session, scores every candidate with the champion ideas
model of each horizon, ranks by expected value after costs and records the survivors in
the idea ledger (`generate_ideas`; "No edge today" for a horizon when nothing clears the
bar, the training hint when a horizon has no model). The 09:20 run fills each pending
idea at its session's open or invalidates it and opens the paper position; the 15:45 run
grades every open idea against the bars since entry, keeps the paper book in step and
snapshots the paper equity (a position without a close is reported, never marked at a
guess). Intraday monitoring stays off: it needs an always-on host (SETUP.md).

Honest skips beat mis-referenced records: the 07:30 job refuses to write a call when the
previous session's target bar is missing, when the session's year has no holiday calendar
(unless forced), or when it runs on the reference session itself before 18:30 IST, and it
says so in the run details instead. Global cues live on other exchanges, so a missing bar
on one of their holidays is a note, not a problem, and their bars are only ever fetched
through the last complete UTC day, then re-requested at 07:30 so a provisional print is
replaced by the final close.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import IST, Settings
from app.jobs.runner import JobContext, JobResult, JobSkipped
from app.ml import baselines, core, features, ideas_models, models, ranking, strategies
from app.ml import ideas_features as idf
from app.ml import registry as model_registry
from app.ml.core import FeatureRow, IdeaRecordIn, MarketCallRecordIn, finite
from app.models import (
    DailyBar,
    Idea,
    IdeaOutcome,
    Instrument,
    ModelVersion,
    PaperEquity,
    PaperTrade,
    Prediction,
    PredictionGrade,
)
from app.providers import registry
from app.providers.base import AuthExpired, MarketDataProvider, NotConfigured, ProviderError
from app.services import (
    brief,
    calendar,
    flows,
    global_cues,
    grading,
    idea_grading,
    ideas_scorecard,
    instruments,
    kite_auth,
    ledger,
    paper,
    scorecard,
    telegram,
)
from app.services.backfill import run_backfill
from app.services.market_data import (
    daily_frame,
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
    the brief). The stock ideas follow the calls (`generate_ideas`) under the same
    recording guard; the brief goes out when a call or an idea was newly recorded.
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

    ideas_out = generate_ideas(ctx, session, through, chain, blocked=blocked)
    details["ideas"] = ideas_out.details
    problems += ideas_out.problems
    db.commit()  # the idea ledger rows are final before anything is announced

    if recorded or ideas_out.recorded:
        cards: list[tuple[Prediction, scorecard.Scorecard | None]] = [
            (p, scorecard.scorecard(db, p.target, settings=s)) for p in current
        ]
        text = brief.morning_brief(
            cards,
            session,
            s,
            head=ledger.head(db),
            ideas=idea_tracks(db, ideas_out.ideas, s),
            idea_head=ledger.head_idea(db),
            idea_notes=ideas_out.notes,
        )
        sent = telegram.send(text)
        details["brief"] = "sent" if sent else "not sent (Telegram not configured or failed)"
    else:
        details["brief"] = "not sent (nothing new recorded)"

    summary = [_call_summary(p) for p in current] or [f"no call recorded for {session}"]
    summary.append(ideas_out.summary)
    message = "; ".join([*summary, *problems])
    return JobResult("partial" if problems else "success", message, details)


def _session_open(
    chain: list[MarketDataProvider], symbol: str, session: date, exchange: str = "INDEX"
) -> tuple[float, str] | None:
    """The session's opening print from the first provider that has a bar dated `session`.

    The bar is partial (the session is running) and is NOT stored: a provisional close in
    `daily_bar` would later pass for the real one. A quote's last price is not an open and
    is never used as one.
    """
    for provider in chain:
        try:
            bars = provider.daily_bars(symbol, exchange, session, session)
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
    """09:20: grade every gap call of the session against the actual open, then fill or
    invalidate the session's pending ideas at that open (`fill_ideas`)."""
    db = ctx.db
    session = core.session_for(ctx.calendar, ctx.now)
    calls = ledger.calls_for_session(db, session)
    ideas = ledger.ideas_for_session(db, session, current_only=True)
    if not calls and not ideas:
        raise JobSkipped(f"no market call recorded for {session} and no idea to fill")
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

    fills = fill_ideas(ctx, ideas, session, chain)
    details["ideas"] = fills.details
    problems += fills.problems
    db.commit()
    if fills.processed:
        sent = telegram.send(brief.fill_summary(fills.processed, session))
        details["ideas"]["summary"] = (
            "sent" if sent else "not sent (Telegram not configured or failed)"
        )
    summary = [f"{graded} gap call(s) graded for {session}"]
    if ideas:
        summary.append(
            f"{fills.filled} idea(s) filled, {fills.invalidated} invalidated, "
            f"{fills.pending} still pending"
        )
    message = "; ".join([*summary, *problems])
    return JobResult("partial" if problems else "success", message, details)


def _sessions_back(cal: calendar.TradingCalendar, session: date, n: int) -> date:
    """The session `n` trading days before `session`."""
    d = session
    for _ in range(n):
        d = cal.previous_trading_day(d)
    return d


def close_grading(ctx: JobContext) -> JobResult:
    """15:45: fetch the session close, grade every ungraded call that has one, report;
    then grade the open ideas, keep the paper book in step and snapshot the paper equity
    (`grade_ideas`).

    A call still without a close within the last UNRESOLVED_SESSIONS sessions is a
    problem (partial run). Older ones will not resolve by themselves (the session never
    traded, or the bar is gone for good): they are listed under `unresolvable` with the
    void command, without turning every afternoon's run partial.
    """
    s = ctx.settings
    db = ctx.db
    session = ctx.calendar.last_completed_session(ctx.now)
    pending = ledger.ungraded(db, through=session)
    open_ideas = ledger.open_ideas(db, through=session)
    has_book = db.scalar(select(PaperTrade.id).limit(1)) is not None
    if not pending and not open_ideas and not has_book:
        raise JobSkipped(
            f"no ungraded market calls, open ideas or paper positions through {session}"
        )
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

    ideas_out = grade_ideas(ctx, open_ideas, session, chain, oldest_recent=oldest_recent)
    details["ideas"] = ideas_out.details
    db.commit()

    if graded or ideas_out.graded:
        text = brief.close_summary(graded, ideas=ideas_out.graded, equity=ideas_out.equity)
        sent = telegram.send(text)
        details["summary"] = "sent" if sent else "not sent (Telegram not configured or failed)"
    message = f"{len(graded)} call(s) graded; {len(remaining)} still ungraded"
    if remaining:
        message += ": " + "; ".join(remaining)
    if unresolvable:
        message += (
            f"; {len(unresolvable)} unresolvable (older than {UNRESOLVED_SESSIONS} sessions, "
            "see details)"
        )
    if open_ideas or has_book:
        message += "; " + ideas_out.summary
    if ideas_out.problems:
        message += "; " + "; ".join(ideas_out.problems)
    partial = bool(remaining or ideas_out.problems)
    return JobResult("partial" if partial else "success", message, details)


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


# ----------------------------------------------------------------------------- Phase 3

TRAIN_IDEAS_HINT = "no ideas model: run `python -m app.cli train-ideas`"
RETRAIN_IDEAS_HINT = "retrain needed (run `python -m app.cli train-ideas`)"
UNIVERSE_HINT = "no universe instruments: run `python -m app.cli sync-instruments`"
# Share of the universe that must have a bar for the signal session before ideas are
# generated from it: the momentum decile is a cross-sectional rank, and "No edge today"
# read off a half-empty universe would be a verdict about missing data, not the market.
UNIVERSE_MIN_COVERAGE = 0.5
MAX_ERROR_REASONS = 5  # distinct provider reasons kept in the run details
STOCK_EXCHANGE = "NSE"
IDEA_VOID_HINT = "void it with `python -m app.cli ideas --void {seq} --reason ...`"


@dataclass
class UniverseRefresh:
    """What the pre-run universe refresh found: how much of the universe has the signal
    session's bar. Nothing is filled in; a stock without bars is simply not eligible."""

    n_members: int
    n_with_bar: int  # members with a bar dated `through` after the refresh
    n_fetched: int  # members a provider supplied bars for in this run
    n_without_bars: int  # members with no bars at all (the provider does not know them)
    errors: list[str] = field(default_factory=list)  # distinct provider reasons (capped)
    n_errors: int = 0

    @property
    def coverage(self) -> float:
        return self.n_with_bar / self.n_members if self.n_members else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruments": self.n_members,
            "with_bar": self.n_with_bar,
            "fetched": self.n_fetched,
            "without_bars": self.n_without_bars,
            "coverage": finite(round(self.coverage, 4)),
            "errors": self.n_errors,
            "error_reasons": list(self.errors),
        }


def refresh_universe(
    db: Session,
    members: Sequence[Instrument],
    through: date,
    chain: list[MarketDataProvider],
    years: int,
) -> UniverseRefresh:
    """Best-effort `ensure_recent` for every universe member through `through`, then a
    count of who actually has that session's bar (the two Nifty 500 names the provider
    does not carry stay without bars and never become candidates)."""
    fetched = 0
    reasons: list[str] = []
    n_errors = 0
    for inst in members:
        source, errors = ensure_recent(db, inst, through, chain, years)
        if source is not None:
            fetched += 1
        if errors:
            n_errors += 1
            why = errors[-1]
            if why not in reasons and len(reasons) < MAX_ERROR_REASONS:
                reasons.append(why)
    ids = [m.id for m in members]
    with_bar = db.scalar(
        select(func.count()).where(DailyBar.instrument_id.in_(ids), DailyBar.trade_date == through)
    )
    any_bar = db.scalar(
        select(func.count(func.distinct(DailyBar.instrument_id))).where(
            DailyBar.instrument_id.in_(ids)
        )
    )
    return UniverseRefresh(
        n_members=len(members),
        n_with_bar=int(with_bar or 0),
        n_fetched=fetched,
        n_without_bars=len(members) - int(any_bar or 0),
        errors=reasons,
        n_errors=n_errors,
    )


@dataclass
class IdeasOutcome:
    """The ideas step of the 07:30 run (`generate_ideas`)."""

    details: dict[str, Any] = field(default_factory=dict)
    ideas: list[Idea] = field(default_factory=list)  # the session's current ideas, seq order
    problems: list[str] = field(default_factory=list)
    recorded: int = 0  # new idea-ledger rows written in this run
    # horizon -> the line the brief shows when that horizon has no idea ("No edge today",
    # the training hint, ...). None means the step did not run at all, so the brief
    # carries no ideas section rather than a made-up verdict.
    notes: dict[str, str] | None = field(default_factory=dict)
    summary: str = "ideas: not generated"


def _idea_details(i: Idea) -> dict[str, Any]:
    return {
        "seq": i.seq,
        "symbol": i.symbol,
        "horizon": i.horizon,
        "setup": i.setup,
        "strategy": i.strategy,
        "rank": i.rank,
        "entry_low": i.entry_low,
        "entry_high": i.entry_high,
        "stop": i.stop,
        "target_1": i.target_1,
        "target_2": i.target_2,
        "p_target": i.p_target,
        "expected_return_low_pct": i.expected_return_low_pct,
        "expected_return_high_pct": i.expected_return_high_pct,
        "risk_reward": i.risk_reward,
        "ev_pct_after_costs": i.ev_pct_after_costs,
        "position_qty": i.position_qty,
        "position_value": i.position_value,
        "risk_amount": i.risk_amount,
        "is_late": i.is_late,
        "surveillance_unknown": i.surveillance_unknown,
    }


def _horizon_status(out: IdeasOutcome, horizon: str, status: str, *, problem: bool = False) -> None:
    out.details[horizon] = {"status": status}
    if out.notes is not None:
        out.notes[horizon] = status
    if problem:
        out.problems.append(f"{horizon} ideas: {status}")


def _ideas_summary(out: IdeasOutcome, horizons: Iterable[str]) -> str:
    parts: list[str] = []
    for h in horizons:
        d = out.details.get(h) or {}
        status = str(d.get("status", "not generated"))
        if d.get("ideas"):
            n = len(d["ideas"])
            extra = " (already recorded)" if status == "already recorded" else ""
            parts.append(f"{h} {n} idea{'' if n == 1 else 's'}{extra}")
        elif status.startswith(ranking.NO_EDGE_TEXT):
            parts.append(f"{h} {ranking.NO_EDGE_TEXT}")
        else:
            parts.append(f"{h} {status}")
    return "ideas: " + ", ".join(parts)


def _candidate_rows(feats: pd.DataFrame, through: date) -> dict[tuple[str, str], pd.Series]:
    """The candidate frame rows (features at D plus the strategy's columns) keyed by
    (symbol, strategy), the row `ideas_models.live_features` turns into a model input."""
    cf = strategies.candidate_frame(feats, through)
    return {(str(r["symbol"]), str(r["strategy"])): r for _, r in cf.iterrows()}


def _score_candidates(
    mv: ModelVersion,
    spec: core.HorizonSpec,
    cands: Sequence[strategies.Candidate],
    rows: Mapping[tuple[str, str], pd.Series],
    snapshots: dict[tuple[str, str], FeatureRow],
    s: Settings,
) -> tuple[list[ranking.ScoredCandidate], list[str]]:
    """P(target before stop) for every candidate under one horizon's champion.

    Raises LookupError when the model expects a feature the pipeline no longer produces
    (the horizon is then not predicted, with the retrain hint). Features the pipeline
    produces but the model never saw are ignored and returned, for the run details.
    """
    artifact = mv.artifact
    assert artifact, "champion without an artifact"
    trained = [str(f) for f in mv.feature_names]
    scored: list[ranking.ScoredCandidate] = []
    extra: list[str] = []
    for c in cands:
        series = rows.get((c.symbol, c.strategy))
        if series is None:
            continue
        row = ideas_models.live_features(series, None, c.strategy)
        missing = [f for f in trained if f not in row]
        if missing:
            raise LookupError(
                f"{RETRAIN_IDEAS_HINT}: model {mv.version} expects features the pipeline no "
                f"longer produces ({_names(missing)})"
            )
        if not extra:
            extra = sorted(f for f in row if f not in trained)
        pred = ideas_models.predict(artifact, trained, row, settings=s)
        snapshots[(c.symbol, c.strategy)] = row
        scored.append(
            ranking.ScoredCandidate(
                c, pred.p_target, spec, drivers=pred.all_drivers, exchange=STOCK_EXCHANGE
            )
        )
    return scored, extra


def generate_ideas(
    ctx: JobContext,
    session: date,
    through: date,
    chain: list[MarketDataProvider],
    *,
    blocked: str | None = None,
) -> IdeasOutcome:
    """07:30, after the calls: the stock ideas for `session`, per horizon, into the ledger.

    Steps: (1) ideas already recorded for the session are left alone, per horizon
    (re-runs are idempotent; the race between two live runs is closed by the idea ledger
    lock, `LedgerConflict`); (2) `blocked` (the calls' recording guard: unverified
    calendar, provisional reference close) blocks the ideas too; (3) a horizon without a
    champion model is reported with the training hint and makes the run partial; (4) the
    universe's bars are refreshed through `through` (the signal session, best effort,
    reported), and ideas are only generated when at least UNIVERSE_MIN_COVERAGE of the
    universe has that bar; (5) point-in-time features at `through`
    (`ideas_features.stock_features`), the rule-based candidates (`strategies`), the
    champion's P(target) per candidate (`ideas_models.predict`), the ranking by EV after
    costs with the p / EV bar, the one-per-symbol rule and the per-horizon cap
    (`ranking.rank_report`), position size from the paper risk settings, and one ledger
    row per idea with the full feature snapshot and every driver. A horizon with nothing
    over the bar gets "No edge today" and no row. `surveillance_unknown` is always True:
    no ASM/GSM source is reachable.
    """
    s, db = ctx.settings, ctx.db
    specs = core.horizons(s)
    out = IdeasOutcome(
        details={"session": session.isoformat(), "signal_session": through.isoformat()}
    )
    assert out.notes is not None
    existing = ledger.ideas_for_session(db, session, current_only=True)
    out.ideas = list(existing)
    for h in specs:
        mine = [i for i in existing if i.horizon == h]
        if mine:
            out.details[h] = {
                "status": "already recorded",
                "ideas": [_idea_details(i) for i in mine],
            }
    todo = [h for h in specs if h not in out.details]
    if not todo:
        out.summary = _ideas_summary(out, specs)
        return out
    if blocked is not None:
        for h in todo:
            _horizon_status(out, h, f"not generated: {blocked}")
        out.summary = _ideas_summary(out, specs)
        return out

    champions: dict[str, ModelVersion] = {}
    for h in todo:
        mv = ideas_models.ideas_champion(db, h, s)
        if mv is None or not mv.artifact:
            _horizon_status(out, h, TRAIN_IDEAS_HINT, problem=True)
        else:
            champions[h] = mv
    if not champions:
        out.summary = _ideas_summary(out, specs)
        return out

    members = instruments.universe_instruments(db)
    if not members:
        for h in champions:
            _horizon_status(out, h, f"not generated: {UNIVERSE_HINT}", problem=True)
        out.summary = _ideas_summary(out, specs)
        return out
    refresh = refresh_universe(db, members, through, chain, s.backfill_years)
    out.details["universe"] = refresh.as_dict()
    db.commit()  # keep what was fetched even if the scoring step fails and retries
    if refresh.coverage < UNIVERSE_MIN_COVERAGE:
        why = (
            f"only {refresh.n_with_bar} of {refresh.n_members} universe stocks have a bar "
            f"for {through}"
        )
        for h in champions:
            _horizon_status(out, h, f"not generated: {why}", problem=True)
        out.summary = _ideas_summary(out, specs)
        return out

    feats = idf.stock_features(db, [m.id for m in members], through, settings=s, start=through)
    cands = strategies.candidates(feats, through)
    rows = _candidate_rows(feats, through)
    eligible = int(feats["eligible"].fillna(False).astype(bool).sum()) if not feats.empty else 0
    out.details["candidates"] = {
        "eligible": eligible,
        "n": len(cands),
        "by_strategy": {
            name: sum(c.strategy == name for c in cands) for name in strategies.STRATEGY_NAMES
        },
    }
    ids_by_symbol = {m.symbol: m.id for m in members}
    for c in cands:
        if c.instrument_id is not None:
            ids_by_symbol[c.symbol] = c.instrument_id

    scored: list[ranking.ScoredCandidate] = []
    snapshots: dict[tuple[str, str], FeatureRow] = {}
    scoring: dict[str, ModelVersion] = {}
    for h, mv in champions.items():
        try:
            horizon_scored, extra = _score_candidates(mv, specs[h], cands, rows, snapshots, s)
        except LookupError as exc:
            _horizon_status(out, h, f"not predicted: {exc}", problem=True)
            continue
        scored += horizon_scored
        scoring[h] = mv
        out.details[h] = {
            "status": "scored",
            "model_version": mv.version,
            "n_candidates": len(horizon_scored),
        }
        if extra:
            out.details[h]["features_not_in_model"] = extra

    report = ranking.rank_report(scored, s)
    out.details["ranking"] = report.as_dict()
    for h, mv in scoring.items():
        kept = [o for o in report.ideas if o.horizon == h]
        if not kept:
            text = report.no_edge.get(h) or ranking.no_edge_text(h, s)
            out.details[h]["status"] = text
            out.notes[h] = text
            continue
        recorded_rows: list[dict[str, Any]] = []
        for rank, o in enumerate(kept, start=1):
            qty, value, risk = ranking.size_position(ranking.entry_mid(o), o.stop, s)
            rec = IdeaRecordIn(
                session_date=session,
                as_of=ctx.now,
                instrument_id=ids_by_symbol[o.symbol],
                output=o,
                position_qty=qty,
                position_value=value,
                risk_amount=risk,
                rank=rank,
                feature_snapshot=snapshots[(o.symbol, o.strategy)],
                model_version_id=mv.id,
                surveillance_unknown=True,
            )
            try:
                row = ledger.record_idea(db, rec, settings=s)
                out.recorded += 1
                status = "recorded"
            except ledger.LedgerConflict as conflict:
                # Another run recorded this idea while we were computing: theirs stands.
                row = conflict.existing
                status = "already recorded"
            out.ideas.append(row)
            recorded_rows.append({**_idea_details(row), "status": status})
        out.details[h]["status"] = "recorded"
        out.details[h]["ideas"] = recorded_rows
    out.ideas.sort(key=lambda i: i.seq)
    out.summary = _ideas_summary(out, specs)
    return out


def idea_tracks(
    db: Session, ideas: Sequence[Idea], settings: Settings
) -> list[tuple[Idea, ideas_scorecard.IdeaTrack]]:
    """Each idea with the live record of its setup at its horizon (for the brief)."""
    cache: dict[tuple[str, str], ideas_scorecard.IdeaTrack] = {}
    out: list[tuple[Idea, ideas_scorecard.IdeaTrack]] = []
    for i in ideas:
        key = (i.horizon, i.setup)
        if key not in cache:
            cache[key] = ideas_scorecard.idea_track(db, i.horizon, i.setup, settings=settings)
        out.append((i, cache[key]))
    return out


# --------------------------------------------------------------------------- 09:20 fills


@dataclass
class FillReport:
    """The ideas part of the 09:20 run (`fill_ideas`)."""

    details: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    processed: list[tuple[Idea, IdeaOutcome]] = field(default_factory=list)  # decided now
    filled: int = 0
    invalidated: int = 0
    pending: int = 0  # still waiting for the session's open


def _outcome_details(o: IdeaOutcome | None) -> dict[str, Any]:
    if o is None:
        return {}
    return {
        "entry_date": o.entry_date.isoformat() if o.entry_date else None,
        "entry_price": o.entry_price,
        "exit_date": o.exit_date.isoformat() if o.exit_date else None,
        "exit_price": o.exit_price,
        "gross_return_pct": o.gross_return_pct,
        "net_return_pct": o.net_return_pct,
        "r_multiple": o.r_multiple,
        "days_held": o.days_held,
        "max_favorable_pct": o.max_favorable_pct,
        "max_adverse_pct": o.max_adverse_pct,
        "notes": o.notes,
    }


def fill_ideas(
    ctx: JobContext, ideas: Sequence[Idea], session: date, chain: list[MarketDataProvider]
) -> FillReport:
    """09:20: the entry rule for each of the session's pending ideas at the session's open.

    The open comes from a provider's bar for the session (partial, never stored: see
    `_session_open`); inside [entry_low, entry_high] the idea is filled at it and its
    paper position opened (`paper.open_trade_for_idea`), outside it is invalidated
    (`idea_grading.open_check_fill`). An idea whose open is not available yet stays
    pending and makes the run partial; ideas already decided are reported as they are.
    """
    db = ctx.db
    rep = FillReport()
    outcomes = idea_grading.outcomes_for(db, ideas)
    opens: dict[tuple[str, str], tuple[float, str] | None] = {}
    for idea in ideas:
        key = f"seq {idea.seq} {idea.symbol} {idea.horizon}"
        existing = outcomes.get(idea.id)
        status = idea_grading.status_of(existing)
        if status != "pending":
            rep.details[key] = {"status": f"already {status}", **_outcome_details(existing)}
            continue
        ikey = (idea.symbol, idea.exchange)
        if ikey not in opens:
            opens[ikey] = _session_open(chain, idea.symbol, session, idea.exchange)
        found = opens[ikey]
        if found is None:
            rep.pending += 1
            rep.details[key] = {"status": "open not available yet"}
            rep.problems.append(
                f"{idea.symbol} ({idea.horizon}): open for {session} not available yet"
            )
            continue
        open_, source = found
        outcome = idea_grading.open_check_fill(db, idea, open_, now=ctx.now, source=source)
        entry: dict[str, Any] = {
            "status": outcome.status,
            "actual_open": open_,
            "entry_zone": [idea.entry_low, idea.entry_high],
            "entry_price": outcome.entry_price,
            "note": outcome.notes,
            "source": source,
        }
        if outcome.status == "open":
            rep.filled += 1
            try:
                trade = paper.open_trade_for_idea(db, idea, outcome, now=ctx.now)
                entry["paper_trade_id"] = trade.id
                entry["qty"] = trade.qty
            except ValueError as exc:
                entry["paper_trade"] = f"not opened: {exc}"
                rep.problems.append(
                    f"{idea.symbol} ({idea.horizon}): filled but no paper trade: {exc}"
                )
        else:
            rep.invalidated += 1
        rep.processed.append((idea, outcome))
        rep.details[key] = entry
    return rep


# --------------------------------------------------------------------------- 15:45 grading


@dataclass
class IdeaGradingReport:
    """The ideas and paper part of the 15:45 run (`grade_ideas`)."""

    details: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    graded: list[tuple[Idea, IdeaOutcome]] = field(default_factory=list)  # written this run
    equity: PaperEquity | None = None
    summary: str = ""


def _bar_dates(bars: pd.DataFrame) -> list[date]:
    return [pd.Timestamp(d).date() for d in bars.index]


def grade_ideas(
    ctx: JobContext,
    open_ideas: Sequence[Idea],
    session: date,
    chain: list[MarketDataProvider],
    *,
    oldest_recent: date,
) -> IdeaGradingReport:
    """15:45: settle or advance every open idea, keep the paper book in step, snapshot equity.

    Bars of the ideas' and open positions' instruments and the Nifty 50 are refreshed
    through the session first (best effort, reported). Each idea is graded on its bars
    from the entry session on (`idea_grading.grade_idea`: a pending idea is filled from
    its entry session's open first, a 09:20 run that never happened loses nothing), and
    `paper.sync_trade_with_outcome` opens or closes its paper trade. An idea whose
    instrument has no bar for the session yet waits (partial run); one that has waited
    longer than UNRESOLVED_SESSIONS sessions is listed as unresolvable with the void
    command. The equity row for the session is written only when every open position has
    a close (`paper.snapshot_equity`); otherwise the gap is reported, never papered over.
    """
    db, s = ctx.db, ctx.settings
    rep = IdeaGradingReport()
    insts: dict[int, Instrument] = {}
    for idea in open_ideas:
        inst = db.get(Instrument, idea.instrument_id)
        if inst is not None:
            insts[inst.id] = inst
    open_trades = paper.list_trades(db, status="open")
    for t in open_trades:
        inst = instruments.get_instrument(db, t.symbol, t.exchange)
        if inst is not None:
            insts[inst.id] = inst
    has_book = db.scalar(select(PaperTrade.id).limit(1)) is not None
    nifty = instruments.get_instrument(db, idf.NIFTY_SYMBOL, "INDEX")
    if nifty is not None and (has_book or open_ideas):
        insts[nifty.id] = nifty  # the equity row's benchmark close
    if insts:
        refresh = _refresh_bars(
            db,
            list(insts.values()),
            session,
            chain,
            s.backfill_years,
            session=session,
            now=ctx.now,
        )
        rep.details["refresh"] = refresh.problems or "ok"
        db.commit()

    waiting: list[str] = []
    unresolvable: list[str] = []
    settled = still_open = invalidated = 0
    for idea in open_ideas:
        key = f"seq {idea.seq} {idea.symbol} {idea.horizon}"
        bars = daily_frame(db, idea.instrument_id, start=idea.session_date)
        dates = _bar_dates(bars)
        if dates and dates[-1] > session:  # never read past the session being graded
            keep = [d <= session for d in dates]
            bars = bars[keep]
            dates = [d for d in dates if d <= session]
        latest = dates[-1] if dates else None
        existing = idea_grading.outcome_of(db, idea.id)
        pending = idea_grading.status_of(existing) == "pending"
        no_entry_bar = pending and (not dates or dates[0] != idea.session_date)
        if latest != session or no_entry_bar:
            what = (
                f"no bar for its entry session {idea.session_date}"
                if no_entry_bar
                else f"no bar for {session}"
            )
            if latest is not None and latest >= oldest_recent and not no_entry_bar:
                # The stock has traded recently; today's bar is simply not published yet.
                waiting.append(f"{idea.symbol} {idea.horizon} (seq {idea.seq}): {what} yet")
            elif idea.session_date < oldest_recent:
                unresolvable.append(
                    f"{idea.symbol} {idea.horizon} {idea.session_date} (seq {idea.seq}): {what} "
                    f"after {UNRESOLVED_SESSIONS} sessions; {IDEA_VOID_HINT.format(seq=idea.seq)}"
                )
            else:
                waiting.append(f"{idea.symbol} {idea.horizon} (seq {idea.seq}): {what} yet")
            rep.details[key] = {"status": idea_grading.status_of(existing), "problem": what}
            continue
        source = str(bars["source"].iloc[-1]) if "source" in bars.columns else "close_grading"
        try:
            outcome = idea_grading.grade_idea(
                db, idea, bars, now=ctx.now, source=source, settings=s
            )
        except ValueError as exc:
            rep.problems.append(f"{idea.symbol} {idea.horizon} (seq {idea.seq}): not graded: {exc}")
            rep.details[key] = {"status": idea_grading.status_of(existing), "problem": str(exc)}
            continue
        entry: dict[str, Any] = {"status": outcome.status, **_outcome_details(outcome)}
        try:
            trade = paper.sync_trade_with_outcome(db, idea, outcome, now=ctx.now, settings=s)
        except ValueError as exc:
            trade = None
            entry["paper_trade"] = f"not synced: {exc}"
            rep.problems.append(
                f"{idea.symbol} {idea.horizon} (seq {idea.seq}): paper trade not synced: {exc}"
            )
        if trade is not None:
            entry["paper_trade_id"] = trade.id
            entry["paper_trade_status"] = trade.status
            entry["net_pnl"] = trade.net_pnl
        if outcome.status in idea_grading.TERMINAL_STATUSES:
            settled += 1
        elif outcome.status == "open":
            still_open += 1
        else:
            invalidated += 1
        rep.graded.append((idea, outcome))
        rep.details[key] = entry
    rep.details["waiting"] = waiting
    rep.details["unresolvable"] = unresolvable
    rep.problems += waiting

    has_book = db.scalar(select(PaperTrade.id).limit(1)) is not None
    if has_book:
        closes: dict[str, float] = {}
        for t in paper.list_trades(db, status="open"):
            inst = instruments.get_instrument(db, t.symbol, t.exchange)
            bar = db.get(DailyBar, (inst.id, session)) if inst is not None else None
            close = finite(bar.close) if bar is not None else None
            if close is not None:
                closes[t.symbol] = close
        nifty_bar = db.get(DailyBar, (nifty.id, session)) if nifty is not None else None
        nifty_close = finite(nifty_bar.close) if nifty_bar is not None else None
        if nifty_close is None:
            rep.details["nifty_close"] = f"missing for {session}: equity row without the benchmark"
        try:
            eq = paper.snapshot_equity(
                db, session, closes, nifty_close, source="close_grading", settings=s
            )
            rep.equity = eq
            rep.details["paper_equity"] = {
                "trade_date": eq.trade_date.isoformat(),
                "equity": eq.equity,
                "cash": eq.cash,
                "invested": eq.invested,
                "open_positions": eq.open_positions,
                "nifty_close": eq.nifty_close,
            }
        except ValueError as exc:
            rep.details["paper_equity"] = f"not snapshotted: {exc}"
            rep.problems.append(f"paper equity for {session} not snapshotted: {exc}")

    parts = [f"{settled} idea(s) settled", f"{still_open} open"]
    if invalidated:
        parts.append(f"{invalidated} invalidated")
    if waiting:
        parts.append(f"{len(waiting)} waiting for a bar")
    if unresolvable:
        parts.append(f"{len(unresolvable)} unresolvable (see details)")
    if rep.equity is not None:
        parts.append(f"paper equity {rep.equity.equity:,.0f}")
    rep.summary = ", ".join(parts)
    return rep

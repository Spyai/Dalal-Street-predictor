"""Job implementations: Phase 1 data jobs and the Phase 2 market-call loop.

Phase 2 (brief §3, §4.1, §5): `premarket_prediction` at 07:30 makes and locks the call,
`open_check` at 09:20 grades the gap call, `close_grading` at 15:45 grades the close, and
`weekly_retrain` on Saturday trains a challenger and promotes it only when it beats the
champion out of sample. Every job returns per-target details for /health, says "partial"
whenever a number it wanted was missing, and never substitutes a guess for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
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
from app.services.market_data import ensure_recent, ingest_bhavcopy
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


# ----------------------------------------------------------------------------- Phase 2

TRAIN_HINT = "no model: run `python -m app.cli train`"
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


def _refresh_bars(
    db: Session,
    insts: list[Instrument],
    through: date,
    chain: list[MarketDataProvider],
    years: int,
) -> list[str]:
    """Best-effort `ensure_recent` for each instrument; returns one line per instrument
    that still has no bar for `through` (the caller reports them, never fills them in)."""
    problems: list[str] = []
    for inst in insts:
        source, errors = ensure_recent(db, inst, through, chain, years)
        if source is None and errors:
            problems.append(f"{inst.exchange}:{inst.symbol} has no bar for {through}: {errors[-1]}")
    return problems


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


def _predict_target(
    db: Session, target: str, session: date, ctx: JobContext
) -> tuple[Prediction | None, dict[str, Any], list[str]]:
    """Make and record one target's call. Returns (row or None, details, problems)."""
    s = ctx.settings
    champion = model_registry.champion(db, target)
    if champion is None or not champion.artifact:
        return None, {"status": TRAIN_HINT}, [f"{target}: {TRAIN_HINT}"]
    try:
        row, ref_date, ref_close = features.live_row(db, target, session, settings=s)
    except LookupError as exc:
        return None, {"status": f"no features: {exc}"}, [f"{target}: {exc}"]
    trained_names = [str(f) for f in champion.feature_names]
    missing = [f for f in trained_names if f not in row]
    if missing:
        why = (
            f"model {champion.version} expects features the pipeline no longer produces "
            f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}); retrain"
        )
        return None, {"status": f"not predicted: {why}"}, [f"{target}: {why}"]

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
    p = ledger.record_market_call(db, rec, settings=s)
    problems: list[str] = []
    expected_ref = ctx.calendar.previous_trading_day(session)
    if ref_date < expected_ref:
        problems.append(
            f"{target}: reference close is from {ref_date}, not {expected_ref} (bar missing)"
        )
    details = _call_details(
        p,
        "recorded",
        model_version=champion.version,
        range_p10_pct=p.range_p10_pct,
        range_p90_pct=p.range_p90_pct,
        gap_expected_pct=p.gap_expected_pct,
        drivers=[d["text"] for d in (p.drivers or []) if isinstance(d, dict) and d.get("text")],
    )
    return p, details, problems


def premarket_prediction(ctx: JobContext) -> JobResult:
    """07:30: refresh inputs, make and lock the market call per target, send the brief.

    Data refresh and the FII/DII pull are best effort: a missing bar is reported in the
    details (and the call still records which reference close it used). A target with no
    champion model is reported with the training hint; an existing call for the session
    is left alone (re-runs are idempotent and do not re-send the brief).
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
        db, _reference_instruments(db, targets), through, chain, s.backfill_years
    )
    details["refresh"] = refresh or "ok"
    problems += refresh
    try:
        n = flows.refresh(db, registry.nse_provider())
        details["fii_dii"] = f"ok ({n} rows)"
        mark_ok("nse")
    except ProviderError as exc:
        mark_error("nse", exc)
        details["fii_dii"] = f"unavailable: {exc.reason}"
    db.commit()  # keep what was fetched even if the prediction step fails and retries

    current: list[Prediction] = []
    recorded = 0
    for target in targets:
        existing = ledger.latest_call(db, target, session)
        if existing is not None:
            details[target] = _call_details(existing, "already recorded")
            current.append(existing)
            continue
        p, target_details, target_problems = _predict_target(db, target, session, ctx)
        details[target] = target_details
        problems += target_problems
        if p is not None:
            current.append(p)
            recorded += 1
    db.commit()  # the ledger rows are final before anything is announced

    if recorded:
        cards: list[tuple[Prediction, scorecard.Scorecard | None]] = [
            (p, scorecard.scorecard(db, p.target, settings=s)) for p in current
        ]
        sent = telegram.send(brief.morning_brief(cards, session, s))
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


def close_grading(ctx: JobContext) -> JobResult:
    """15:45: fetch the session close, grade every ungraded call that has one, report."""
    s = ctx.settings
    db = ctx.db
    session = ctx.calendar.last_completed_session(ctx.now)
    pending = ledger.ungraded(db, through=session)
    if not pending:
        raise JobSkipped(f"no ungraded market calls through {session}")
    details: dict[str, Any] = {"session": session.isoformat()}

    chain = registry.market_data_chain(db)
    targets = sorted({p.target for p in pending})
    insts = {t: instruments.get_instrument(db, t, "INDEX") for t in targets}
    refresh = _refresh_bars(
        db, [i for i in insts.values() if i is not None], session, chain, s.backfill_years
    )
    details["refresh"] = refresh or "ok"
    db.commit()

    graded: list[tuple[Prediction, PredictionGrade]] = []
    remaining: list[str] = []
    for p in pending:
        inst = insts.get(p.target)
        bar = db.get(DailyBar, (inst.id, p.session_date)) if inst is not None else None
        if bar is None or finite(bar.close) is None:
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
    if graded:
        sent = telegram.send(brief.close_summary(graded))
        details["summary"] = "sent" if sent else "not sent (Telegram not configured or failed)"
    message = f"{len(graded)} call(s) graded; {len(remaining)} still ungraded"
    if remaining:
        message += ": " + "; ".join(remaining)
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
    promote, reason = model_registry.should_promote(
        champ.metrics if champ is not None else None, result.metrics, s
    )
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

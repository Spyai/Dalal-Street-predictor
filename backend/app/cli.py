"""Command-line tools. Run inside the backend container:

    docker compose run --rm backend python -m app.cli <command>

Commands:
    migrate                     apply database migrations
    create-user                 create the single login (password + TOTP)
    reset-user                  delete the login and create a new one
    sync-instruments            Kite instrument tokens + universe membership
    import-holidays FILE        load an NSE holiday CSV (date,description)
    backfill [--years N]        load daily history for indices, global cues and universe
    run-job NAME [--force]      run one scheduled job now (--force ignores the calendar)
    jobs                        list scheduled jobs
    record-fixtures DIR         save live provider responses for adapter tests
    test-telegram               send a test message to your Telegram chat
    train [--target T] [--promote-always]
                                walk-forward train the market model; the first model is
                                promoted, later ones only if they beat the champion
    predict [--force]           the 07:30 pre-market prediction run, now
    grade [--force]             the 15:45 close grading, now
    grade --void SEQ --reason TEXT [--category no_session|data_error]
                                mark one ledger record as ungradeable (its session never
                                traded, or the data is unusable); the record itself stays
    verify-ledger               walk both hash chains (market calls, stock ideas); exit 1
                                if either is broken
    scorecard [--target T]      the live track record in plain words
    train-ideas [--horizon swing|positional] [--promote-always]
                                walk-forward train the stock-ideas model(s) on the
                                universe's historical strategy candidates; same
                                champion/challenger rule as `train`
    ideas [--force]             the 07:30 run (market calls + stock ideas), now
    ideas --dry-run             score and rank now, print, record nothing
    ideas --void SEQ --reason TEXT [--category no_session|data_error]
                                mark one idea as ungradeable; the ledger row stays
    paper [--summary]           the paper portfolio: trades, equity curve, metrics
    paper --trade SYMBOL QTY --price P [--date YYYY-MM-DD] [--stop S] [--target T]
          [--note TEXT]         enter a manual paper trade (a price is required: it is
                                never guessed); never a real order
    paper --close ID --price P [--date YYYY-MM-DD]
                                close a paper trade at that price (manual trades; ideas
                                close themselves when graded)
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import session_scope
from app.logging import configure_logging


def cmd_migrate(_: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent.parent / "alembic"))
    command.upgrade(cfg, "head")
    with session_scope() as db:
        from app.services.instruments import ensure_reference_instruments

        ensure_reference_instruments(db)
    print("Database migrated.")
    return 0


def _create_user(username: str | None) -> int:
    import qrcode

    from app.security.auth import create_user

    username = username or input("Username: ").strip()
    pw = getpass.getpass("Password (min 12 chars): ")
    if pw != getpass.getpass("Repeat password: "):
        print("Passwords do not match.", file=sys.stderr)
        return 1
    with session_scope() as db:
        new = create_user(db, username, pw)
    print(
        "\nScan this QR code with an authenticator app (Google Authenticator, 1Password, Aegis):\n"
    )
    qr = qrcode.QRCode(border=1)
    qr.add_data(new.provisioning_uri)
    qr.print_ascii(invert=True)
    print(f"\nOr enter this secret manually: {new.totp_secret}")
    print("Store it somewhere safe. It is shown once.\n")
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    return _create_user(args.username)


def cmd_reset_user(args: argparse.Namespace) -> int:
    from app.models import AppUser

    if input("Delete the existing login and create a new one? [y/N] ").strip().lower() != "y":
        return 1
    with session_scope() as db:
        db.execute(delete(AppUser))
    return _create_user(args.username)


def cmd_sync_instruments(_: argparse.Namespace) -> int:
    from app.jobs.definitions import get_spec
    from app.jobs.runner import run_job

    status = run_job(get_spec("instrument_sync"))
    print(f"instrument_sync: {status} (details on /health)")
    return 0 if status in {"success", "partial"} else 1


def cmd_import_holidays(args: argparse.Namespace) -> int:
    from app.services.calendar import import_holiday_csv

    with session_scope() as db:
        n = import_holiday_csv(db, Path(args.file))
    print(f"Imported {n} holidays from {args.file}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from app.models import Instrument
    from app.providers import registry
    from app.services import instruments
    from app.services.backfill import run_backfill
    from app.services.calendar import TradingCalendar
    from app.services.market_data import last_final_global_day

    years = args.years or get_settings().backfill_years
    with session_scope() as db:
        instruments.ensure_reference_instruments(db)
        if not db.scalar(select(Instrument.id).where(Instrument.in_universe).limit(1)):
            print("Universe empty; syncing from NSE (or imports/ind_<universe>list.csv) ...")
            try:
                n, src = instruments.sync_universe(db, registry.nse_provider())
                print(f"  {n} symbols from {src}")
            except Exception as exc:
                print(f"  could not load universe: {exc}", file=sys.stderr)
        now = datetime.now(UTC)
        cal = TradingCalendar.load(db)
        end = cal.last_completed_session(now)
        # Global cues trade in other time zones: a bar dated the current UTC day may be a
        # partial print, so they stop at the last complete UTC day (see market_data).
        global_end = min(end, last_final_global_day(now))
        start = end - timedelta(days=int(365.25 * years))
        index_ids = list(db.scalars(select(Instrument.id).where(Instrument.exchange == "INDEX")))
        global_ids = list(db.scalars(select(Instrument.id).where(Instrument.exchange == "GLOBAL")))
        uni_ids = [i.id for i in instruments.universe_instruments(db)]
        chain = registry.market_data_chain(db, include_broker=not args.no_broker)
    print(f"Backfilling {start} .. {end} with providers: {[p.name for p in chain]}")
    groups = (
        ("indices", index_ids, end),
        ("global cues", global_ids, global_end),
        ("universe", uni_ids, end),
    )
    for label, ids, through in groups:
        print(f"\n== {label}: {len(ids)} instruments through {through}")
        rep = run_backfill(ids, start, through, chain, progress=lambda m: print("  " + m))
        print("  " + rep.summary())
        for key, why in list(rep.missing.items())[:25]:
            print(f"  missing {key}: {why}")
        if len(rep.missing) > 25:
            print(f"  ... and {len(rep.missing) - 25} more")
    return 0


def cmd_run_job(args: argparse.Namespace) -> int:
    from app.jobs.definitions import get_spec
    from app.jobs.runner import run_job

    spec = get_spec(args.name)
    if spec.fn is None:
        print(f"{spec.name} is not built yet (Phase {spec.phase}).", file=sys.stderr)
        return 1
    status = run_job(
        spec,
        sleep=lambda s: None if args.no_retry_wait else __import__("time").sleep(s),
        force=bool(getattr(args, "force", False)),
    )
    print(f"{spec.name}: {status}")
    return 0 if status != "failed" else 1


def _run_named_job(name: str, force: bool) -> int:
    from app.jobs.definitions import get_spec
    from app.jobs.runner import run_job

    status = run_job(get_spec(name), force=force)
    print(f"{name}: {status} (details on /health)")
    return 0 if status != "failed" else 1


def cmd_predict(args: argparse.Namespace) -> int:
    return _run_named_job("premarket_prediction", args.force)


def cmd_grade(args: argparse.Namespace) -> int:
    if args.void is not None:
        return _void_call(args.void, args.reason, args.category)
    if args.reason or args.category:
        print("--reason / --category only make sense with --void SEQ", file=sys.stderr)
        return 2
    return _run_named_job("close_grading", args.force)


def _void_call(seq: int, reason: str | None, category: str | None) -> int:
    """Mark one ledger record as ungradeable. The record is immutable; only its grade row
    says that no close will come, and why."""
    from app.models import Prediction
    from app.services.grading import void_grade

    if not reason or not reason.strip():
        print("--void needs --reason TEXT (it is shown next to the call forever)", file=sys.stderr)
        return 2
    with session_scope() as db:
        p = db.scalar(select(Prediction).where(Prediction.seq == seq))
        if p is None:
            print(f"no ledger record with seq {seq}", file=sys.stderr)
            return 1
        try:
            g = void_grade(
                db, p, now=datetime.now(UTC), reason=reason, category=category or "no_session"
            )
        except ValueError as exc:
            print(f"not voided: {exc}", file=sys.stderr)
            return 1
        print(f"voided seq {p.seq} ({p.target}, session {p.session_date}): {g.notes}")
    return 0


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "\u2014" if x is None else format(x, spec)


def _pct(x: float | None) -> str:
    return "\u2014" if x is None else f"{x * 100:.0f}%"


def cmd_train(args: argparse.Namespace) -> int:
    """Walk-forward train the market model per target and register the result."""
    from app.ml import features, models
    from app.ml import registry as model_registry

    s = get_settings()
    targets = [args.target.strip().upper()] if args.target else s.prediction_target_symbols
    failures = 0
    with session_scope() as db:
        for target in targets:
            print(f"== {target}")
            frame = features.build_frame(db, target, settings=s)
            if frame.empty:
                print("  no daily bars; run `python -m app.cli backfill` first", file=sys.stderr)
                failures += 1
                continue
            labelled = int(frame["label_pct"].notna().sum())
            print(
                f"  {labelled} labelled sessions, {len(features.FEATURE_NAMES)} features; "
                f"walk-forward from {s.wf_min_train_days} sessions, step {s.wf_step_days} ..."
            )
            try:
                result = models.train_walk_forward(frame, settings=s, target=target)
            except ValueError as exc:
                print(f"  not trained: {exc}", file=sys.stderr)
                failures += 1
                continue
            champ = model_registry.champion(db, target)
            promote, reason = model_registry.should_promote(champ, result, s)
            if args.promote_always and not promote:
                promote, reason = True, f"--promote-always ({reason})"
            mv = model_registry.save_model(db, target, result, promote=promote, note=reason)
            db.commit()
            model_registry.log_to_mlflow(result, target, promote, s)
            m = result.metrics
            print(f"  version {mv.version}, trained through {result.trained_through}")
            print(
                f"  out of sample: n={m.n_oos} accuracy={_pct(m.accuracy)} "
                f"brier={_fmt(m.brier)} log_loss={_fmt(m.log_loss)}"
            )
            for name, entry in m.baselines.items():
                print(
                    f"    baseline {name:<18} n={int(entry.get('n', 0)):<4} "
                    f"accuracy={_pct(entry.get('accuracy'))} brier={_fmt(entry.get('brier'))}"
                )
            print(f"  {'PROMOTED' if promote else 'rejected'}: {reason}")
    return 1 if failures else 0


def cmd_verify_ledger(_: argparse.Namespace) -> int:
    """Walk both chains; exit 1 if either is broken (each chain is reported on its own)."""
    from app.services.ledger import verify_all

    with session_scope() as db:
        results = verify_all(db)
    ok = True
    for chain, v in results.items():
        print(f"{chain}: " + ("OK: " if v.ok else "BROKEN: ") + v.message)
        if not v.ok:
            ok = False
            print(f"  first bad seq: {v.first_bad_seq} ({v.checked} records verified before it)")
    return 0 if ok else 1


def cmd_scorecard(args: argparse.Namespace) -> int:
    from app.services.scorecard import scorecard

    s = get_settings()
    targets = [args.target.strip().upper()] if args.target else s.prediction_target_symbols
    with session_scope() as db:
        for target in targets:
            sc = scorecard(db, target, settings=s)
            print(f"== {target}")
            print(f"  {sc.verdict}")
            print(
                f"  graded: {sc.n_graded}  accuracy: {_pct(sc.accuracy)}  "
                f"brier: {_fmt(sc.brier)}  log loss: {_fmt(sc.log_loss)}"
            )
            print(f"  gap calls graded: {sc.gap_n}  gap accuracy: {_pct(sc.gap_accuracy)}")
            for b in sc.baselines:
                print(f"    baseline {b['label']:<42} n={b['n']:<4} accuracy={_pct(b['accuracy'])}")
            for row in sc.by_confidence:
                print(
                    f"    confidence {row['confidence']:<7} n={row['n']:<4} "
                    f"accuracy={_pct(row.get('accuracy'))}"
                )
            if sc.champion is None:
                print("  champion: none (run `python -m app.cli train`)")
            else:
                oos = sc.champion["oos"]
                print(
                    f"  champion: {sc.champion['version']} trained through "
                    f"{sc.champion['trained_through']}; OOS n={oos.get('n_oos')} "
                    f"accuracy={_pct(oos.get('accuracy'))} brier={_fmt(oos.get('brier'))} "
                    f"log loss={_fmt(oos.get('log_loss'))}"
                )
            if sc.sample_warning:
                print(f"  sample-size warning: fewer than {s.min_sample_size} graded calls")
    return 0


# ----------------------------------------------------------------------------- Phase 3


def cmd_train_ideas(args: argparse.Namespace) -> int:
    """Walk-forward train the ideas model per horizon on the universe's candidates."""
    from app.ml import core, ideas_models
    from app.ml import registry as model_registry

    s = get_settings()
    specs = core.horizons(s)
    names = [args.horizon] if args.horizon else list(specs)
    failures = 0
    with session_scope() as db:
        for h in names:
            spec = specs[h]
            print(
                f"== {h} ({spec.days} sessions, target {spec.target_atr:g} x ATR, "
                f"stop {spec.stop_atr:g} x ATR)"
            )
            try:
                data = ideas_models.training_data(db, h, settings=s)
            except ValueError as exc:
                print(f"  not trained: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(
                f"  {data.n_stocks} universe stocks with bars through {data.through}; "
                f"{data.n_candidates} strategy candidates, {data.n_labelled} labelled, "
                f"{data.n_invalidated} would not have filled (open outside the entry zone); "
                f"walk-forward from {s.wf_min_train_days} sessions, step {s.wf_step_days}, "
                f"embargo {spec.days} ..."
            )
            try:
                result = ideas_models.train_walk_forward(
                    data.frame, h, settings=s, universe=data.universe
                )
            except ValueError as exc:
                print(f"  not trained: {exc}", file=sys.stderr)
                failures += 1
                continue
            champ = ideas_models.ideas_champion(db, h, s)
            promote, reason = model_registry.should_promote(champ, result, s)
            if args.promote_always and not promote:
                promote, reason = True, f"--promote-always ({reason})"
            mv = ideas_models.save_ideas_model(db, result, promote=promote, note=reason, settings=s)
            db.commit()
            model_registry.log_to_mlflow(result, ideas_models.model_target(s), promote, s)
            m = result.metrics.as_dict()
            print(f"  version {mv.version}, trained through {result.trained_through}")
            print(
                f"  out of sample: n={m.get('n_oos')} hit rate of ideas over the bar="
                f"{_pct(_f(m.get('hit_rate')))} (n={m.get('n_selected')}) "
                f"ranked={_pct(_f(m.get('hit_rate_ranked')))} (n={m.get('n_ranked')}) "
                f"brier={_fmt(_f(m.get('brier')))} log_loss={_fmt(_f(m.get('log_loss')))}"
            )
            over_bar = _pct_pt(_f(m.get("expectancy_after_costs_pct")))
            ranked = _pct_pt(_f(m.get("expectancy_ranked_pct")))
            everyone = _pct_pt(_f(m.get("expectancy_all_candidates_pct")))
            round_trip = _pct_pt(_f(m.get("cost_round_trip_pct")))
            print(
                f"  expectancy after costs: over the bar {over_bar}, ranked {ranked}, "
                f"all candidates {everyone} (round trip {round_trip})"
            )
            baselines = m.get("baselines")
            if isinstance(baselines, dict):
                for name, entry in baselines.items():
                    if not isinstance(entry, dict):
                        continue
                    print(
                        f"    baseline {name:<18} n={int(entry.get('n') or 0):<6} "
                        f"hit rate={_pct(_f(entry.get('hit_rate')))} "
                        f"brier={_fmt(_f(entry.get('brier')))}"
                    )
            per_strategy = m.get("per_strategy")
            if isinstance(per_strategy, dict):
                for name, entry in per_strategy.items():
                    if not isinstance(entry, dict):
                        continue
                    print(
                        f"    strategy {name:<20} n={int(entry.get('n') or 0):<6} "
                        f"hit rate={_pct(_f(entry.get('hit_rate')))} "
                        f"selected={int(entry.get('n_selected') or 0)}"
                    )
            print(
                f"  ranking check, bar ignored: best {s.ideas_max_per_horizon} a session hit rate="
                f"{_pct(_f(m.get('hit_rate_top')))} (n={m.get('n_top')}) "
                f"expectancy={_pct_pt(_f(m.get('expectancy_top_pct')))}"
            )
            lift = m.get("lift")
            if isinstance(lift, list) and lift:
                cells = " ".join(
                    f"d{int(_f(r.get('decile')) or 0)}={_pct(_f(r.get('hit_rate')))}"
                    for r in lift
                    if isinstance(r, dict)
                )
                print(f"  hit rate by P(target) decile (1 = lowest): {cells}")
            print(
                f"  chosen out of sample: {result.params.get('n_estimators')} rounds, "
                f"shrink {result.params.get('shrink')} towards the base rate"
            )
            print(f"  {'PROMOTED' if promote else 'rejected'}: {reason}")
    return 1 if failures else 0


def _f(x: object) -> float | None:
    """A float or None (metrics dicts hold finite() values, i.e. float | None)."""
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else None


def _pct_pt(x: float | None) -> str:
    """A signed percentage-point number such as +1.20%."""
    return "\u2014" if x is None else f"{x:+.2f}%"


def cmd_ideas(args: argparse.Namespace) -> int:
    if args.void is not None:
        return _void_idea(args.void, args.reason, args.category)
    if args.reason or args.category:
        print("--reason / --category only make sense with --void SEQ", file=sys.stderr)
        return 2
    if args.dry_run:
        return _ideas_dry_run()
    return _run_named_job("premarket_prediction", args.force)


def _ideas_dry_run() -> int:
    """Score and rank the next session's candidates right now and print what the 07:30 run
    would record, the nearest misses included. Nothing is written to either ledger."""
    from app.config import IST
    from app.jobs import tasks
    from app.jobs.runner import JobContext
    from app.ml import core
    from app.services.calendar import TradingCalendar

    s = get_settings()
    now = datetime.now(UTC)
    with session_scope() as db:
        cal = TradingCalendar.load(db)
        ctx = JobContext(
            db=db, now=now, today=now.astimezone(IST).date(), calendar=cal, settings=s, force=True
        )
        out = tasks.ideas_dry_run(ctx)
        db.commit()  # refreshed bars stay; no ledger row was written
    d = out.details
    print(
        f"dry run for session {d.get('session')} (signal session {d.get('signal_session')}); "
        "nothing recorded"
    )
    if d.get("recording_block"):
        print(f"  a live run would not record: {d['recording_block']}")
    if d.get("already_recorded"):
        print(f"  {d['already_recorded']} idea(s) already in the ledger for this session")
    uni = d.get("universe") or {}
    if uni:
        print(
            f"  universe: {uni.get('with_bar')} of {uni.get('instruments')} stocks have a bar "
            f"for {d.get('signal_session')}"
        )
    cands = d.get("candidates") or {}
    if cands:
        print(
            f"  candidates: {cands.get('n')} from {cands.get('eligible')} eligible stocks; "
            f"by strategy {cands.get('by_strategy')}"
        )
    rk = d.get("ranking") or {}
    if rk:
        print(
            f"  ranking: {rk.get('n_cleared')} of {rk.get('n_candidates')} candidates cleared "
            f"the bar ({rk.get('n_unpriced')} unpriced, {rk.get('n_unsizable')} unsizable); "
            f"best P(target) {_pct(_f(rk.get('best_p_target')))}, "
            f"best EV {_pct_pt(_f(rk.get('best_ev_pct')))}"
        )
    for h in core.horizons(s):
        hd = d.get(h) or {}
        print(f"== {h}: {hd.get('status', 'not scored')}")
        for row in hd.get("ideas") or []:
            print(
                f"  #{row['rank']} {row['symbol']} {row['setup']} "
                f"P(target)={_pct(_f(row.get('p_target')))} "
                f"EV={_pct_pt(_f(row.get('ev_pct_after_costs')))} "
                f"entry {row['entry_low']:,.2f}-{row['entry_high']:,.2f} "
                f"stop {row['stop']:,.2f} target {row['target_1']:,.2f} "
                f"qty {row['position_qty']}"
            )
        closest = (rk.get("closest") or {}).get(h) or []
        if closest:
            print(
                "  best by EV: "
                + "; ".join(
                    f"{c['symbol']} ({c['strategy']}) P={_pct(_f(c.get('p_target')))} "
                    f"EV={_pct_pt(_f(c.get('ev_pct_after_costs')))}"
                    f"{'' if c.get('cleared') else ' (under the bar)'}"
                    for c in closest
                )
            )
    for problem in out.problems:
        print(f"  problem: {problem}")
    return 0


def _void_idea(seq: int, reason: str | None, category: str | None) -> int:
    """Mark one idea as ungradeable; the ledger row is immutable, only its outcome says so."""
    from app.models import Idea
    from app.services.idea_grading import void_idea

    if not reason or not reason.strip():
        print("--void needs --reason TEXT (it is shown next to the idea forever)", file=sys.stderr)
        return 2
    with session_scope() as db:
        idea = db.scalar(select(Idea).where(Idea.seq == seq))
        if idea is None:
            print(f"no idea with seq {seq}", file=sys.stderr)
            return 1
        try:
            o = void_idea(
                db, idea, now=datetime.now(UTC), reason=reason, category=category or "no_session"
            )
        except ValueError as exc:
            print(f"not voided: {exc}", file=sys.stderr)
            return 1
        where = f"{idea.symbol} {idea.horizon}, {idea.session_date}"
        print(f"voided idea seq {idea.seq} ({where}): {o.notes}")
    return 0


def _parse_day(text: str | None) -> date:
    if text is None:
        from app.config import IST

        return datetime.now(IST).date()
    return date.fromisoformat(text)


def cmd_paper(args: argparse.Namespace) -> int:
    from app.services import paper

    s = get_settings()
    if args.trade is not None:
        symbol, qty_text = args.trade
        if args.price is None:
            print("--trade needs --price P: a fill price is never guessed", file=sys.stderr)
            return 2
        try:
            qty = int(qty_text)
        except ValueError:
            print(f"QTY must be a whole number of shares, got {qty_text!r}", file=sys.stderr)
            return 2
        with session_scope() as db:
            try:
                t = paper.manual_trade(
                    db,
                    symbol=symbol,
                    qty=qty,
                    entry_date=_parse_day(args.date),
                    entry_price=args.price,
                    stop=args.stop,
                    target=args.target,
                    note=args.note,
                )
            except ValueError as exc:
                print(f"not entered: {exc}", file=sys.stderr)
                return 1
            db.flush()
            print(
                f"paper trade {t.id}: buy {t.qty} {t.symbol} at {t.entry_price:.2f} on "
                f"{t.entry_date} (stop {_fmt(t.stop, '.2f')}, target {_fmt(t.target, '.2f')})"
            )
        return 0
    if args.close is not None:
        from app.models import PaperTrade

        if args.price is None:
            print("--close needs --price P: an exit price is never guessed", file=sys.stderr)
            return 2
        with session_scope() as db:
            existing = db.get(PaperTrade, args.close)
            if existing is None:
                print(f"no paper trade with id {args.close}", file=sys.stderr)
                return 1
            try:
                closed = paper.close_trade(
                    db,
                    existing,
                    _parse_day(args.date),
                    args.price,
                    now=datetime.now(UTC),
                    settings=s,
                    note="closed from the CLI",
                )
            except ValueError as exc:
                print(f"not closed: {exc}", file=sys.stderr)
                return 1
            print(
                f"paper trade {closed.id}: {closed.symbol} closed at "
                f"{_fmt(closed.exit_price, '.2f')} on {closed.exit_date}; "
                f"gross {_fmt(closed.gross_pnl, ',.0f')}, costs {_fmt(closed.costs, ',.0f')}, "
                f"net {_fmt(closed.net_pnl, ',.0f')}, R {_fmt(closed.r_multiple, '.2f')}"
            )
        return 0
    with session_scope() as db:
        ps = paper.portfolio_summary(db, settings=s)
    _print_portfolio(ps, s.min_sample_size)
    return 0


def _print_portfolio(ps: Any, min_sample_size: int) -> None:
    """The paper book in plain words; every metric with its sample size, None as a dash."""
    print(f"== paper portfolio (capital {ps.capital:,.0f})")
    print(f"  {ps.note}")
    print(
        f"  closed trades: {ps.n_trades}  open: {ps.n_open}  "
        f"wins/losses: {ps.n_wins}/{ps.n_losses}  hit rate: {_pct(ps.hit_rate)}"
    )
    print(
        f"  avg win: {_fmt(ps.avg_win, ',.0f')} ({_pct_pt(ps.avg_win_pct)})  "
        f"avg loss: {_fmt(ps.avg_loss, ',.0f')} ({_pct_pt(ps.avg_loss_pct)})  "
        f"expectancy: {_fmt(ps.expectancy, ',.0f')} per trade ({_pct_pt(ps.expectancy_pct)})  "
        f"profit factor: {_fmt(ps.profit_factor, '.2f')}"
    )
    print(
        f"  equity snapshots: {ps.n_days}  total return: {_pct_pt(ps.total_return_pct)}  "
        f"Nifty 50 (price index) over the same days: {_pct_pt(ps.nifty_return_pct)}  "
        f"max drawdown: {_pct_pt(ps.max_drawdown_pct)}  sharpe: {_fmt(ps.sharpe, '.2f')}  "
        f"sortino: {_fmt(ps.sortino, '.2f')} (n={ps.n_returns} daily returns)"
    )
    for row in ps.open_trades:
        origin = f" (idea {row['idea_id']})" if row["idea_id"] is not None else " (manual)"
        print(
            f"    open  #{row['id']} {row['symbol']} x{row['qty']} at {row['entry_price']:.2f} "
            f"since {row['entry_date']} stop {_fmt(row['stop'], '.2f')} "
            f"target {_fmt(row['target'], '.2f')}{origin}"
        )
    for row in ps.recent_closed:
        print(
            f"    closed #{row['id']} {row['symbol']} x{row['qty']} "
            f"{row['entry_date']} -> {row['exit_date']} net {_fmt(row['net_pnl'], ',.0f')} "
            f"({_pct_pt(row['net_return_pct'])}) R {_fmt(row['r_multiple'], '.2f')}"
        )
    if ps.sample_warning:
        print(f"  sample-size warning: fewer than {min_sample_size} closed paper trades")


def cmd_jobs(_: argparse.Namespace) -> int:
    from app.jobs.definitions import job_specs

    for s in job_specs():
        built = "built" if s.fn else f"Phase {s.phase}"
        print(f"{s.name:<22} {s.schedule:<28} {built}")
    return 0


def cmd_record_fixtures(args: argparse.Namespace) -> int:
    """Save raw live responses so adapter tests run against real payloads."""
    from app.providers import registry
    from app.providers.nse import ARCHIVES, INDEX_LIST_URLS, WWW

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    nse = registry.nse_provider()
    http = nse._http  # noqa: SLF001
    http.get(WWW + "/", expect_html=True)
    targets = {
        "nse_fiidii.json": WWW + "/api/fiidiiTradeReact",
        "nse_holidays.json": WWW + "/api/holiday-master?type=trading",
        "nse_nifty50list.csv": INDEX_LIST_URLS["nifty50"],
    }
    day = date.today() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    targets["nse_bhavcopy_full.csv"] = (
        f"{ARCHIVES}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv"
    )
    for name, url in targets.items():
        try:
            resp = http.get(url, headers={"Referer": WWW + "/"})
            (out / name).write_bytes(resp.content)
            print(f"saved {name}")
        except Exception as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
    try:
        import yfinance as yf

        df = yf.download(
            ["^GSPC", "USDINR=X"],
            period="10d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
        df.to_csv(out / "yf_daily_multi.csv")
        print("saved yf_daily_multi.csv")
    except Exception as exc:
        print(f"FAILED yfinance: {exc}", file=sys.stderr)
    (out / "RECORDED_AT.json").write_text(
        json.dumps({"recorded_at": datetime.now(UTC).isoformat()})
    )
    return 0


def cmd_test_telegram(_: argparse.Namespace) -> int:
    from app.services.telegram import send

    s = get_settings()
    if not s.telegram_configured:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env", file=sys.stderr)
        return 1
    ok = send("✅ Dalal can reach you. Alerts will arrive in this chat.")
    print("sent" if ok else "failed (check the token and chat ID; see /health)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    configure_logging("warning")
    p = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    for name, fn in (("create-user", cmd_create_user), ("reset-user", cmd_reset_user)):
        sp = sub.add_parser(name)
        sp.add_argument("--username")
        sp.set_defaults(fn=fn)
    sub.add_parser("sync-instruments").set_defaults(fn=cmd_sync_instruments)
    sp = sub.add_parser("import-holidays")
    sp.add_argument("file")
    sp.set_defaults(fn=cmd_import_holidays)
    sp = sub.add_parser("backfill")
    sp.add_argument("--years", type=int)
    sp.add_argument("--no-broker", action="store_true", help="skip Kite, use yfinance only")
    sp.set_defaults(fn=cmd_backfill)
    sp = sub.add_parser("run-job")
    sp.add_argument("name")
    sp.add_argument("--no-retry-wait", action="store_true")
    sp.add_argument("--force", action="store_true", help="run even on a weekend or holiday")
    sp.set_defaults(fn=cmd_run_job)
    sub.add_parser("jobs").set_defaults(fn=cmd_jobs)
    sp = sub.add_parser("train")
    sp.add_argument("--target", help='an INDEX symbol, e.g. "NIFTY 50" (default: all targets)')
    sp.add_argument(
        "--promote-always",
        action="store_true",
        help="promote even if it does not beat the champion",
    )
    sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("predict")
    sp.add_argument("--force", action="store_true", help="run even on a weekend or holiday")
    sp.set_defaults(fn=cmd_predict)
    sp = sub.add_parser("grade")
    sp.add_argument("--force", action="store_true", help="run even on a weekend or holiday")
    sp.add_argument("--void", type=int, metavar="SEQ", help="mark this record as ungradeable")
    sp.add_argument("--reason", help="why no close will come (required with --void)")
    sp.add_argument(
        "--category",
        choices=("no_session", "data_error"),
        help="what kind of void (default no_session)",
    )
    sp.set_defaults(fn=cmd_grade)
    sub.add_parser("verify-ledger").set_defaults(fn=cmd_verify_ledger)
    sp = sub.add_parser("scorecard")
    sp.add_argument("--target", help='an INDEX symbol, e.g. "NIFTY 50" (default: all targets)')
    sp.set_defaults(fn=cmd_scorecard)
    sp = sub.add_parser("train-ideas")
    sp.add_argument("--horizon", choices=("swing", "positional"), help="default: both")
    sp.add_argument(
        "--promote-always",
        action="store_true",
        help="promote even if it does not beat the champion",
    )
    sp.set_defaults(fn=cmd_train_ideas)
    sp = sub.add_parser("ideas")
    sp.add_argument("--force", action="store_true", help="run even on a weekend or holiday")
    sp.add_argument(
        "--dry-run",
        action="store_true",
        help="score and rank the candidates now, print them, record nothing",
    )
    sp.add_argument("--void", type=int, metavar="SEQ", help="mark this idea as ungradeable")
    sp.add_argument("--reason", help="why it can never be graded (required with --void)")
    sp.add_argument(
        "--category",
        choices=("no_session", "data_error"),
        help="what kind of void (default no_session)",
    )
    sp.set_defaults(fn=cmd_ideas)
    sp = sub.add_parser("paper")
    sp.add_argument("--summary", action="store_true", help="the portfolio summary (default)")
    sp.add_argument("--trade", nargs=2, metavar=("SYMBOL", "QTY"), help="enter a manual trade")
    sp.add_argument("--close", type=int, metavar="ID", help="close this paper trade")
    sp.add_argument("--price", type=float, help="fill / exit price (required with --trade/--close)")
    sp.add_argument("--date", help="trade date YYYY-MM-DD (default: today, IST)")
    sp.add_argument("--stop", type=float, help="stop-loss price (manual trade)")
    sp.add_argument("--target", type=float, help="target price (manual trade)")
    sp.add_argument("--note", help="free text kept with the trade")
    sp.set_defaults(fn=cmd_paper)
    sub.add_parser("test-telegram").set_defaults(fn=cmd_test_telegram)
    sp = sub.add_parser("record-fixtures")
    sp.add_argument("dir")
    sp.set_defaults(fn=cmd_record_fixtures)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line tools. Run inside the backend container:

    docker compose run --rm backend python -m app.cli <command>

Commands:
    migrate                     apply database migrations
    create-user                 create the single login (password + TOTP)
    reset-user                  delete the login and create a new one
    sync-instruments            Kite instrument tokens + universe membership
    import-holidays FILE        load an NSE holiday CSV (date,description)
    backfill [--years N]        load daily history for indices, global cues and universe
    run-job NAME                run one scheduled job now
    jobs                        list scheduled jobs
    record-fixtures DIR         save live provider responses for adapter tests
    test-telegram               send a test message to your Telegram chat
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

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
        cal = TradingCalendar.load(db)
        end = cal.last_completed_session(datetime.now(UTC))
        start = end - timedelta(days=int(365.25 * years))
        ref_ids = list(
            db.scalars(select(Instrument.id).where(Instrument.exchange.in_(["INDEX", "GLOBAL"])))
        )
        uni_ids = [i.id for i in instruments.universe_instruments(db)]
        chain = registry.market_data_chain(db, include_broker=not args.no_broker)
    print(f"Backfilling {start} .. {end} with providers: {[p.name for p in chain]}")
    for label, ids in (("indices + global cues", ref_ids), ("universe", uni_ids)):
        print(f"\n== {label}: {len(ids)} instruments")
        rep = run_backfill(ids, start, end, chain, progress=lambda m: print("  " + m))
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
        spec, sleep=lambda s: None if args.no_retry_wait else __import__("time").sleep(s)
    )
    print(f"{spec.name}: {status}")
    return 0 if status != "failed" else 1


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
    sp.set_defaults(fn=cmd_run_job)
    sub.add_parser("jobs").set_defaults(fn=cmd_jobs)
    sub.add_parser("test-telegram").set_defaults(fn=cmd_test_telegram)
    sp = sub.add_parser("record-fixtures")
    sp.add_argument("dir")
    sp.set_defaults(fn=cmd_record_fixtures)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

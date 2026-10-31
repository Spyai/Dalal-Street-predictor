# Dalal

A private, single-user research desk for Indian equities (NSE + BSE). Every trading day it
predicts before the open, grades itself after the close, writes down what it got wrong, and
only promotes a new model if it beats the old one out of sample.

It is **not** a trading bot. It never places orders: broker access is read-only, and a test
fails the build if the Kite adapter references an order endpoint.

> Personal research tool. Not investment advice.

## Status

| Phase | What | State |
|---|---|---|
| **1. Data foundation** | Docker Compose, TimescaleDB schema, provider adapters (Kite, yfinance, NSE), backfill, scheduler, holiday calendar, Today page (global cues, FII/DII), Stock page with chart, password + TOTP login, `/health` | **Done** |
| 2. Market call + ledger | Feature pipeline, walk-forward LightGBM, baselines, hash-chained ledger, 07:30 run, 15:45 grading, Scorecard, Telegram brief | Next |
| 3. Stock ideas + paper trading | Universe filters, swing/positional models, rule-based strategies, EV ranking after costs, Ideas, paper portfolio, intraday alerts | — |
| 4. Self-review loop | Nightly reflection, LLM post-mortems, journal, champion/challenger retrain with MLflow, kill switch | — |
| 5. Depth | Screener, sectors, F&O, events, news sentiment, backtester, broker portfolio import, alerts manager | — |

## Honesty rules, as implemented

- **No fabricated numbers.** A missing reading shows `—` plus the reason. An old one shows
  `stale` with its as-of time. A failed refresh never overwrites the last good reading; it
  marks it stale.
- **Every value carries its source** (`kite`, `nse_bhavcopy`, `yfinance`). When two sources
  supply the same day, the exchange bhavcopy wins, then Kite, then yfinance.
- **Nothing is skipped silently.** Each scheduled job writes a `job_run` row: success,
  partial, failed (after retries, with a Telegram alert) or `skipped: <reason>`
  (weekend, `NSE holiday: Diwali`, `Kite not logged in`).
- **Polite scraping only.** Honest user agent, robots.txt respected, ≤1 request/second per
  host (shared across processes). On a block (401/403/429/CAPTCHA page) Dalal stops,
  cools down for 6 hours, and says so on `/health`.
- **GIFT Nifty is shown as unavailable.** Neither Kite nor yfinance offers it, and Dalal
  won't scrape around that.

## Architecture

```
browser ──► web (Next.js :3000, only published port, BIND_ADDR = 127.0.0.1 or Tailscale IP)
              │  /api/* proxied (same-origin cookie)
              ▼
            backend (FastAPI)  ◄──►  db (TimescaleDB)   redis (sessions, rate limits, cooldowns)
            scheduler (APScheduler, Asia/Kolkata)  ──┘
              │
              ▼  provider adapters (app/providers)
            KiteProvider ── read-only Kite Connect (daily token, encrypted at rest)
            YFinanceProvider ── EOD fallback + global cues, batched
            NSEProvider ── FII/DII, holidays, index lists, bhavcopy + delivery %
```

Business logic depends only on the interfaces in `app/providers/base.py`
(`MarketDataProvider`, `ExchangeDataProvider`, `NewsProvider`, `FundamentalsProvider`).
Swapping brokers means writing one adapter.

## Quick start

Two ways to run it:

- **Private (recommended):** Docker Compose on your laptop or a VPS, reachable only over
  Tailscale. Full walkthrough, including every key and the daily Kite login:
  **[SETUP.md](SETUP.md)**.
- **Free public link:** Vercel (web + API) + Neon Postgres + GitHub Actions for jobs. Add one
  secret (`VERCEL_TOKEN`) and the "Deploy to Vercel" workflow sets everything up:
  **[DEPLOY_VERCEL.md](DEPLOY_VERCEL.md)**.

```bash
cp .env.example .env                 # fill in SECRET_KEY, POSTGRES_PASSWORD, Kite, Telegram
docker compose up -d --build
docker compose run --rm backend python -m app.cli create-user
open http://127.0.0.1:3000           # log in, then "Log in to Kite"
docker compose run --rm backend python -m app.cli sync-instruments
docker compose run --rm backend python -m app.cli backfill
```

## Repository layout

```
backend/
  app/providers/   adapters + polite HTTP client (robots.txt, rate limit, block cooldown)
  app/services/    calendar, instruments, market data, backfill, cues, flows, indicators, kite auth
  app/jobs/        schedule (definitions.py), runner (retries/skips/alerts), scheduler process
  app/api/         FastAPI routers: auth, today, stocks, kite, health
  app/security/    argon2 + TOTP auth, Fernet encryption for stored secrets
  alembic/         migrations (daily_bar and intraday_bar are hypertables)
  api/index.py      Vercel entrypoint (vercel.json, requirements.txt)
  tests/           170 tests (4 skip until real fixtures are recorded): adapters, lookahead
                   safety, calendar, auth, KV store (Redis vs Postgres), job runner, API,
                   deploy scripts
frontend/          Next.js App Router + Tailwind + lightweight-charts (dark by default)
.github/workflows/ vercel-deploy.yml (one-secret deploy), dalal-jobs.yml (job schedule on Vercel)
.github/scripts/   the deploy/job scripts those workflows run (tested with a fake Vercel CLI)
imports/           manual CSV drop-in (holidays, index lists, Screener exports); gitignored
data/              backups, fixtures, later model artifacts; gitignored
```

## Development

```bash
make test-up                 # throwaway TimescaleDB :55432 + Redis :56379
cd backend && uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pytest             # unit + DB tests
make lint                    # ruff, mypy, tsc
```

Adapter tests use **synthetic fixtures** shaped like each source's documented format (see
`backend/tests/fixtures/README.md`). Replace them with real payloads via
`python -m app.cli record-fixtures`.

## Known gaps after Phase 1

- **Live sources not yet exercised end to end.** The build environment's network blocked
  NSE, Yahoo, Kite and Telegram. Adapters are tested against synthetic fixtures. Your first
  run is the real integration test; `/health` shows exactly what worked.
- **Split/bonus adjustment:** yfinance closes are split-adjusted; Kite and bhavcopy are raw.
  Phase 2's feature pipeline adjusts for corporate actions before computing returns.
- **Not built yet:** F&O OI, participant OI and bulk/block deals in the evening pull (Phase 2);
  BSE archive fallback; intraday candles (table exists, filled in Phase 3); MLflow service
  (Phase 2); news and fundamentals adapters (interfaces only, Phase 5).
- **Frontend:** type-checked and verified with browser screenshots; no automated UI tests yet.

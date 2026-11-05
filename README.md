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
| **2. Market call + ledger** | 52-feature point-in-time pipeline with leakage tests, walk-forward LightGBM (Nifty 50 + Bank Nifty), four naive baselines, append-only hash-chained ledger with verify, 07:30 / 09:20 / 15:45 jobs, Saturday champion/challenger retrain, Scorecard page, Today market-call cards, Telegram morning brief | **Done** |
| **3. Stock ideas + paper trading** | Nifty 500 universe filter (turnover, history, price), four transparent candidate strategies, triple-barrier labels, walk-forward LightGBM per horizon (swing 2–10 days, positional 1–3 months), ranking by expected value after a configurable cost model, idea ledger (second hash chain), 09:20 fills, target/stop/expiry grading, paper portfolio with equity curve, Ideas and Paper pages, drivers top 8 + all factors | **Done** (intraday alerts need an always-on host) |
| 4. Self-review loop | Nightly reflection, LLM post-mortems, journal, champion/challenger retrain with MLflow, kill switch | Next |
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
  app/services/    calendar, instruments, market data, backfill, cues, flows, indicators, kite auth,
                   ledger (two hash chains + verify), grading, idea_grading, paper, scorecard,
                   ideas_scorecard, brief (Telegram text)
  app/ml/          core vocabulary, features (point-in-time), events, models (walk-forward
                   LightGBM + quantiles + drivers), baselines, metrics, registry (champion in
                   Postgres); ideas_features, strategies, ideas_models, costs, ranking
  app/jobs/        schedule (definitions.py), runner (retries/skips/alerts/--force), scheduler process
  app/api/         FastAPI routers: auth, today, stocks, kite, scorecard, ideas, paper, health
  app/security/    argon2 + TOTP auth, Fernet encryption for stored secrets
  alembic/         migrations (daily_bar and intraday_bar are hypertables)
  api/index.py      Vercel entrypoint (vercel.json, requirements.txt)
  tests/           ~520 tests (4 skip until real fixtures are recorded): adapters, lookahead
                   leakage (features, labels, walk-forward with embargo), both ledgers' hashing +
                   immutability, grading, paper math, scorecards, calendar, auth, KV store (Redis
                   vs Postgres), jobs, API, deploy scripts
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

## How the market call works (Phase 2)

Every trading day at 07:30 IST the `premarket_prediction` job builds one feature row per
index from data dated strictly before the session (index returns and indicators, India VIX,
US/Asia/crude/USD-INR/DXY/US 10Y daily bars, FII/DII flows, Nifty 500 breadth, event flags),
asks the champion LightGBM model for UP/FLAT/DOWN probabilities (±0.30 % close-to-close) and a
10–90 % range, adds a rule-based gap call from US futures, and writes an immutable ledger
record (SHA-256 chained to the previous one) before the 09:15 lock. Four naive baselines are
recorded alongside. At 09:20 the gap call is graded against the open; at 15:45 the call is
graded against the close (correct/incorrect, Brier, log loss). On Saturdays a challenger is
retrained with strict walk-forward validation and promoted only if its out-of-sample Brier
beats the champion's by a margin. The Scorecard page shows accuracy vs each baseline,
calibration, confidence buckets and the sample-size warning until 100 calls are graded, and
can re-verify the whole ledger chain.

```bash
python -m app.cli train                 # first model (walk-forward, promotes when no champion)
python -m app.cli predict --force       # record today's call (or next session's) now
python -m app.cli grade --force         # grade whatever can be graded
python -m app.cli verify-ledger         # recompute the hash chain; exit 1 if broken
python -m app.cli scorecard             # the verdict and metrics in the terminal
```

## How the stock ideas work (Phase 3)

After the market calls, the 07:30 job scores every Nifty 500 stock that passes the universe
filter (≥ ₹10 cr average daily turnover, ≥ 250 sessions of history, a bar in the last five
sessions, price ≥ ₹20). Four transparent strategies propose candidates: 20-day breakout with
volume, 12-month-minus-1-month momentum, RSI mean reversion in an uptrend, and a pullback to
the 20-DMA. One LightGBM model per horizon (swing: 10 sessions, target 2 × ATR, stop 1 × ATR;
positional: 60 sessions, 4 × / 2 × ATR) gives each candidate the probability that the target
is hit before the stop, trained walk-forward on triple-barrier labels with an embargo so no
training window overlaps a test signal. The label is exactly the event the ledger later
grades: levels set from the signal session's close, a fill only when the next open lands
inside the entry zone (otherwise the row is "would not have filled" and never trained on),
then the recorded target before the recorded stop. Ideas are ranked by expected value after
brokerage, STT, exchange charges, GST, stamp duty and slippage; the bar is EV ≥ 0.5 % after
costs with a 40 % probability floor (with 2:1 targets the breakeven probability is about a
third plus costs, so the floor only keeps lottery tickets out). At most three per horizon
make it, each with an entry zone, stop, two targets, the probability, risk-reward, a position
size for 1 % of paper capital (never more than the paper book's cash), 3–5 numbered reasons
and what invalidates it. Nothing clears the bar → "No edge today", with the nearest misses
in the job details. The Scorecard shows, per horizon, the out-of-sample hit rate by
probability decile and what "the best three ideas every session, bar ignored" would have
done against every candidate, so a model that only echoes the strategies' base rates is
visible as such. ASM/GSM status is never checked (NSE is unreachable from GitHub) and every idea
says so. At 09:20 an idea is filled at the open if the open is inside its zone, otherwise it
is invalidated; at 15:45 open ideas are graded (target / stop / expired, with costs) and the
paper book is marked to market.

```bash
python -m app.cli train-ideas               # both horizons, walk-forward, promotes when no champion
python -m app.cli ideas --force             # generate today's ideas now (also runs the market calls)
python -m app.cli ideas --dry-run           # score and rank now, print, record nothing
python -m app.cli paper --summary           # hit rate, expectancy, profit factor, drawdown, Sharpe
python -m app.cli verify-ledger             # both chains
```

## Known gaps after Phase 3

- **Every idea is unproven** until its setup/horizon has 100 graded outcomes; the UI and the
  Telegram brief say so beside every probability.
- **Survivorship bias in history.** Universe membership is today's Nifty 500 applied to five
  years of bars; stocks that left the index are missing from training. Membership history is
  not published free; it will be recorded going forward from the daily sync.
- **No intraday alerts** (brief §3 09:15–15:30) on the free hosting: the job stays unbuilt until
  Dalal runs on an always-on machine (SETUP.md).
- **ASM/GSM lists, delivery %, fundamentals and news** are not used (no reachable source yet;
  fundamentals and news are Phase 5).
- **Nifty 50 price index, not TRI**, on the paper equity curve.

## Known gaps after Phase 2

- **The model is unproven.** It has 0 graded live calls; everything shown is out-of-sample
  walk-forward on history, and the UI says so until 100 calls are graded.
- **FII/DII history is thin.** Phase 1 only refreshes the latest day, so the flow features are
  missing for most training rows (kept as missing, never filled in).
- **Timing guidance and error categories** are recorded but not graded (Phase 4).
- **No LLM text yet.** The morning brief is template text; Claude-written briefs and
  post-mortems come in Phase 4 once `ANTHROPIC_API_KEY` is set.

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

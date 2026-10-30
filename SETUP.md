# Setup

This is the private setup: Docker on your laptop or a VPS, over Tailscale. For a free public
link on Vercel instead, follow [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md); sections 1-5 below
(getting the keys) apply to both.

You need: Docker (with Compose), a Zerodha account, a Telegram account, and about 30 minutes.
Every secret goes into `.env`. Never paste a key into a chat, issue or commit.

```bash
cp .env.example .env
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64))"        # paste into .env
python3 -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(24))" # paste into .env
```

---

## 1. Zerodha Kite Connect (required)

**What it gives Dalal:** live quotes, intraday candles, 5 years of daily history, read-only
holdings. Dalal never calls an order endpoint (a test enforces this).

1. Go to <https://developers.kite.trade> and sign up / log in.
2. **Create an app.**
   - Plan: Dalal needs market data (quotes + historical candles). Per Zerodha's 2026
     pricing that means the paid **Connect** plan (recently reported around ₹500/month per
     app). The free **Personal** plan has no market data. Dalal still runs on it, but prices
     fall back to yfinance (EOD only) and Phase 3 intraday alerts won't work.
     **Check the current price on Zerodha's site yourself.** Dalal never subscribes to anything.
   - **Redirect URL:** `<APP_BASE_URL>/api/kite/callback`, exactly. Examples:
     - laptop only: `http://127.0.0.1:3000/api/kite/callback`
     - over Tailscale: `http://dalal-box.your-tailnet.ts.net:3000/api/kite/callback`
   - **Postback URL:** leave blank (Dalal places no orders).
3. Copy the **API key** and **API secret** into `.env`:
   ```
   KITE_API_KEY=...
   KITE_API_SECRET=...
   KITE_USER_ID=AB1234        # your Zerodha client ID. Strongly recommended, see below.
   ```

### The daily login (SEBI rule, unavoidable)

Kite access tokens die every morning at about 06:00 IST, and SEBI's retail-API framework
(mandatory since 1 April 2026) requires an interactive daily login. Dalal will not automate your
Zerodha password or TOTP. Instead:

- At **07:00 IST** on trading days (`KITE_LOGIN_REMINDER_TIME`), if no token exists for the
  day, Dalal sends you a Telegram message with a **Log in to Kite** link.
- Tap it → log in on kite.zerodha.com → Zerodha redirects back to Dalal → done. The token is
  stored encrypted (Fernet, keyed from `SECRET_KEY`) until ~05:55 the next morning.
- You can also use the **Log in to Kite** banner on the Today page.

`KITE_USER_ID` matters: with it set, the callback accepts a token only for *your* Zerodha
account, which is what lets the Telegram link work from your phone without logging in to Dalal
first. Without it, you must be logged in to Dalal in the same browser.

**Static IP:** SEBI's 2026 rules require a whitelisted static IP for *order placement* via API.
Dalal places no orders, so this shouldn't apply. If Zerodha asks for one anyway, register
your VPS's public IP in the developer console.

If you skip the morning login, the 07:30 run (Phase 2) still happens on yfinance + NSE data
and says so on the dashboard. A missed login never means a silently skipped day.

---

## 2. Anthropic API key (required from Phase 2)

Used for news sentiment, the daily brief and post-mortems. The LLM never makes the numeric
prediction.

1. <https://console.anthropic.com> → **API Keys** → **Create key**.
2. Set a monthly spend limit under **Billing → Limits**.
3. `.env`:
   ```
   ANTHROPIC_API_KEY=...
   ANTHROPIC_MODEL_WRITER=claude-opus-5-5     # brief, post-mortems, weekly report
   ANTHROPIC_MODEL_FAST=claude-haiku-5-5      # per-headline sentiment (high volume)
   ```

---

## 3. Telegram bot (required)

1. In Telegram, message **@BotFather** → `/newbot` → pick a name and a username ending in `bot`.
   BotFather replies with the **token**.
2. Send any message (e.g. "hi") to your new bot.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser. Find
   `"chat":{"id":123456789,...}`. That number is your **chat ID**.
4. `.env`:
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=123456789
   ```
5. Test: `docker compose run --rm backend python -m app.cli test-telegram`

Dalal only ever sends to this one chat ID.

---

## 4. FRED API key (optional)

US macro series. <https://fred.stlouisfed.org> → My Account → **API Keys** → request a key →
`FRED_API_KEY=...`. Used from Phase 2.

## 5. NewsAPI or similar (optional)

RSS feeds (ET Markets, Moneycontrol, Business Standard, Livemint, NSE/BSE announcements) cover
news by default. If you want a fallback, get a key from <https://newsapi.org> and set
`NEWSAPI_KEY=...`. Check its terms first: free tiers are typically delayed and development-only.
Dalal will ask before adding anything paid.

---

## 6. Private access with Tailscale (no public URL)

Dalal publishes exactly one port (the web UI), on `BIND_ADDR`. The API, Postgres and Redis are
never exposed outside Docker.

**Laptop only:** keep `BIND_ADDR=127.0.0.1`, `APP_BASE_URL=http://127.0.0.1:3000`.

**Phone + laptop/VPS over Tailscale:**
1. Install Tailscale on the server/laptop and on your phone; log in to the same tailnet.
2. On the server: `tailscale ip -4` → e.g. `100.101.102.103`.
3. `.env`:
   ```
   BIND_ADDR=100.101.102.103
   APP_BASE_URL=http://100.101.102.103:3000      # or the MagicDNS name
   ```
4. Update the Kite redirect URL to match `APP_BASE_URL`.

**Optional HTTPS:** `tailscale serve --bg 3000` serves `https://<machine>.<tailnet>.ts.net`
on your tailnet only. Then set `BIND_ADDR=127.0.0.1`,
`APP_BASE_URL=https://<machine>.<tailnet>.ts.net`, `COOKIE_SECURE=true`.

**VPS firewall:** deny all inbound except Tailscale, e.g.
`ufw default deny incoming && ufw allow in on tailscale0 && ufw enable`
(keep SSH reachable over Tailscale before you enable this).

---

## 7. First run

```bash
docker compose up -d --build                                  # db, redis, backend, scheduler, web
docker compose run --rm backend python -m app.cli create-user # password + TOTP QR in the terminal
```

Scan the QR with an authenticator app (Google Authenticator, 1Password, Aegis). Open
`APP_BASE_URL`, log in, then click **Log in to Kite** on the Today page.

```bash
docker compose run --rm backend python -m app.cli sync-instruments   # Kite tokens + Nifty 500 list
docker compose run --rm backend python -m app.cli backfill           # 5y daily history, resumable
```

The backfill takes a few minutes on Kite (one request per stock at ≤3 req/s). Without a Kite
session it uses yfinance in batches of 50. Re-running it only fetches what's missing.

### If NSE blocks Dalal

NSE's site sometimes refuses automated clients. Dalal identifies itself honestly, respects
robots.txt and ≤1 req/s, and **stops** when refused: the source shows as unavailable on
`/health` and it waits 6 hours before trying again. Nothing is scraped around the block.
Manual fallbacks (drop files into `imports/`):

| Data | File | Where to get it |
|---|---|---|
| Holiday calendar | `imports/nse_holidays_2026.csv` (`date,description`) | nseindia.com → Resources → Holidays |
| Nifty 500 list | `imports/ind_nifty500list.csv` | niftyindices.com → Indices → Nifty 500 → constituents CSV |

Then: `docker compose run --rm backend python -m app.cli import-holidays /imports/nse_holidays_2026.csv`

---

## Daily routine

| IST | What happens | You |
|---|---|---|
| 06:05 | Holiday calendar refresh | — |
| 07:00 | Kite login reminder (only if no token today) | Tap the Telegram link |
| 07:30 | Pre-market prediction (Phase 2) | Read the brief |
| 08:45 | Instrument + universe sync | — |
| every 10 min | Global cues refresh | — |
| 19:30 | Evening pull: FII/DII, bhavcopy + delivery %, gap-fill | — |

Failed jobs retry with backoff (30s, 60s, ...) and then alert you on Telegram. Holidays and
weekends are recorded as "skipped: reason" on `/health`.

## Useful commands

```bash
docker compose ps                                   # what's running
docker compose logs -f scheduler                    # job logs (JSON)
docker compose run --rm backend python -m app.cli jobs
docker compose run --rm backend python -m app.cli run-job evening_data_pull
docker compose run --rm backend python -m app.cli reset-user
make test                                           # backend tests (throwaway DB on :55432)
```

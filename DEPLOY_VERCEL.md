# Deploy on Vercel (free tier)

This gives you a link like `https://dalal-yourname.vercel.app` that works from any phone or
laptop, at ₹0/month. It takes about 20 minutes of clicking.

## How it fits together

| Piece | Runs on | Free tier (check current limits) |
|---|---|---|
| Web app (Next.js) | Vercel project **dalal** | Hobby plan |
| API (FastAPI) | Vercel project **dalal-api**, Mumbai region (`bom1`) | Hobby plan |
| Database | Neon Postgres, added from Vercel's Storage tab | 0.5 GB; 5 years × 500 stocks is ~100 MB |
| Cache, sessions | Upstash Redis, added from Vercel's Storage tab | ~500k commands/month |
| Scheduled jobs | GitHub Actions (`.github/workflows/dalal-jobs.yml`) | Free on public repos; 2,000 min/month private (Dalal uses ~1,200) |

Vercel can't run Dalal's always-on scheduler, so the scheduled jobs run in GitHub Actions
instead, against the same database.

## What you trade for free hosting

Read this before you start. The Docker + Tailscale setup in SETUP.md has none of these issues.

- **The URL is public.** Anyone can reach the login page. It's protected by password + TOTP,
  a lockout after 5 failed attempts, and a one-time setup token. Your brief asked for no
  public URL; this setup reverses that.
- **Make the repo private.** On a public repo, GitHub Actions logs (job results, errors) are
  readable by anyone. GitHub → repo → Settings → General → Danger Zone → Change visibility.
- **Jobs can start late.** GitHub's scheduled runs often start 5 to 30 minutes late, and
  global cues refresh every 30 minutes instead of 10.
- **NSE may refuse GitHub's US servers.** Holiday list, FII/DII and bhavcopy may then show as
  unavailable. The fallback: commit `imports/nse_holidays_2026.csv` and
  `imports/ind_nifty500list.csv`. They're public exchange files, and `.gitignore` allows these
  two names.
- **Phase 3 intraday alerts (every minute) won't fit here.** At that point you'll need a small
  always-on machine (e.g. a free Oracle Cloud VM running the Docker setup).
- **GitHub pauses schedules after 60 days without commits** on public repos. Re-enable
  from the Actions tab.

---

## 1. Generate two secrets

Use your password manager's generator (or `openssl rand -base64 48`):

- `SECRET_KEY`: 64+ random characters. It must be **identical** in Vercel and GitHub,
  because it encrypts your Kite token.
- `SETUP_TOKEN`: 24+ random characters. Used once to create your login.

## 2. Create the API project

1. Go to <https://vercel.com/signup> and sign up with GitHub (Hobby plan, free).
2. **Add New… → Project** → import **Spyai/Dalal-Street-predictor**.
3. Configure:
   - Project name: `dalal-api`
   - **Root Directory:** `backend`
   - Framework Preset: **Other**
   - Environment Variables:

     | Name | Value |
     |---|---|
     | `SECRET_KEY` | from step 1 |
     | `SETUP_TOKEN` | from step 1 |
     | `SCHEDULER_MODE` | `external` |
     | `COOKIE_SECURE` | `true` |
     | `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID` | from Kite (SETUP.md §1) |
     | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | from BotFather (SETUP.md §3) |
     | `APP_BASE_URL` | leave for now; set in step 5 |
4. **Deploy.** It will deploy but can't work yet; there's no database.
5. In `dalal-api` → **Storage** → **Create Database**:
   - **Neon** (Postgres) → Free plan → pick the region closest to India (Singapore, or Mumbai if
     listed) → connect it to `dalal-api`.
   - **Create Database** again → **Upstash for Redis** → Free → same region → connect it.
6. **Deployments** → the latest one → **⋯ → Redeploy**, so it picks up the new variables.
7. Copy the project's URL, e.g. `https://dalal-api-abc123.vercel.app`.

## 3. Create the database tables (GitHub Actions)

1. Find the two connection strings: Vercel → `dalal-api` → **Storage** → click each database
   → **.env.local** tab → **Show secret**.
   - From Neon, copy **`DATABASE_URL_UNPOOLED`** (the direct connection, needed for migrations).
   - From Upstash, copy **`REDIS_URL`** (or `KV_URL`; it starts with `rediss://`).
2. GitHub → your repo → **Settings → Secrets and variables → Actions**:
   - **Secrets** tab → New repository secret, one by one:
     `DATABASE_URL` (the unpooled Neon URL), `REDIS_URL`, `SECRET_KEY` (same as Vercel),
     `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
   - **Variables** tab → New repository variable: `DALAL_JOBS_ENABLED` = `true`.
3. **Actions** tab → **Dalal jobs** → **Run workflow** → command `migrate` → Run.
   It should go green in about a minute.
4. Check `https://dalal-api-abc123.vercel.app/healthz`. It should say `{"ok":true}`.

## 4. Create the web project (this is your link)

1. Vercel → **Add New… → Project** → import **the same repo** again.
2. Configure:
   - Project name: `dalal` (this becomes `dalal-….vercel.app`)
   - **Root Directory:** `frontend`
   - Framework Preset: **Next.js**
   - Environment Variable: `BACKEND_URL` = the API URL from step 2.7
3. **Deploy.** When it finishes, Vercel shows your link, e.g. `https://dalal-xyz.vercel.app`.

## 5. Point everything at the link

- Vercel → `dalal-api` → **Settings → Environment Variables** → add
  `APP_BASE_URL` = your web link → **Redeploy** the API.
- GitHub → **Settings → Secrets and variables → Actions → Variables** → add
  `APP_BASE_URL` = your web link.
- Kite developer console → your app → **Redirect URL** = `https://<your link>/api/kite/callback`.

## 6. Create your login

Open `https://<your link>/setup?token=<SETUP_TOKEN>` → choose a username and a 12+ character
password → scan the QR code with your authenticator app → **Log in**.

Setup switches itself off once a login exists. You can then delete `SETUP_TOKEN` from the
API project's variables.

## 7. Load data

GitHub → **Actions → Dalal jobs → Run workflow**:

1. Log in to Kite first (Today page banner), so instrument tokens and history come from Kite.
2. `sync-instruments`: Kite instrument tokens and the Nifty 500 list.
3. `backfill`: 5 years of daily history. Takes 5 to 15 minutes. Safe to re-run.

From then on the schedule runs by itself. `/health` shows the last run of every job.

## When something's wrong

| Symptom | Check |
|---|---|
| Web build fails: "Set BACKEND_URL" | Step 4.2: the variable is missing from the `dalal` project |
| Login page says "Can't reach the API" | Open `<API URL>/healthz`. If it errors, re-run `migrate` and redeploy the API |
| `/health` says Scheduler (GitHub Actions): down | Actions tab: is `DALAL_JOBS_ENABLED` = `true`? Did the last run fail? |
| Kite login loops back with "failed" | Redirect URL in Kite console must exactly match `<link>/api/kite/callback` |
| Kite says "token invalid" in jobs | `SECRET_KEY` differs between Vercel and GitHub |

Sources for the limits above: Vercel raised the Python function bundle limit to 500 MB in
2026 ([Vercel KB](https://vercel.com/kb/guide/troubleshooting-function-250mb-limit)). Neon,
Upstash and GitHub Actions free-tier numbers were correct when this was written; check their
pricing pages if anything looks off.

# Deploy on Vercel (free tier)

You get a link like `https://dalal-yourname.vercel.app` that works on any phone or laptop,
at ₹0/month. A GitHub Actions workflow does the setup; you add one token.

## The short way: one secret

1. **Make the repo private** (recommended). On a public repo, GitHub Actions logs are
   readable by anyone. GitHub → repo → Settings → General → Danger Zone → Change visibility.
2. **Create a Vercel token.** Sign up at <https://vercel.com/signup> (Hobby, free), then
   **Account Settings → Tokens → Create**. Scope: your account.
3. **Give it to GitHub.** Repo → **Settings → Secrets and variables → Actions → New
   repository secret**. Name: `VERCEL_TOKEN`. Value: the token.
4. **Run the deploy.** Repo → **Actions → Deploy to Vercel**, open the latest run →
   **Re-run all jobs** (or **Run workflow** once it's on `main`). It takes about 5 minutes,
   and the run's **Summary** shows your link.
5. **Create your login.** Vercel → **dalal-api → Settings → Environment Variables** → copy
   `SETUP_TOKEN`, then open `https://<your link>/setup?token=<SETUP_TOKEN>`. Choose a username
   and a 12+ character password, and scan the QR code with an authenticator app.
6. **Add your keys** (Kite, Telegram; see SETUP.md §1 and §3) in **dalal-api → Settings →
   Environment Variables**, set the Kite redirect URL to `https://<your link>/api/kite/callback`,
   and re-run the deploy so the API picks them up.
7. **Merge the PR into `main`.** GitHub only runs scheduled workflows from the default
   branch, so the daily jobs start then. Pushes to `main` redeploy automatically.

### What the workflow does (`.github/workflows/vercel-deploy.yml`)

| Step | First run | Later runs |
|---|---|---|
| Projects | creates `dalal-api` (from `backend/`) and `dalal` (from `frontend/`) | reuses them |
| Database | creates a free Neon Postgres (`dalal-db`) and connects it to `dalal-api` | reuses it |
| Secrets | generates `SECRET_KEY` and `SETUP_TOKEN`, stored only in Vercel | never rotates them |
| Migrations | runs `app.cli migrate` | same |
| Deploy | API in the Vercel region next to the database, then the web app pointed at it | same |
| Check | `GET /healthz` through the web link must answer | same |

There's no Redis on this setup. Sessions, login throttling and rate limits use a small
Postgres table instead (`app/kv.py`), so the database is the only extra service. The
scheduled jobs (`dalal-jobs.yml`) read their settings from the `dalal-api` project with
the same token, so every key lives in one place: Vercel's environment variables.

## What you trade for free hosting

The Docker + Tailscale setup in SETUP.md has none of these issues.

- **The URL is public.** Anyone can reach the login page. It's protected by password + TOTP,
  a lockout after 5 failed attempts, and a one-time setup token. Your brief asked for no
  public URL; this setup reverses that.
- **Jobs can start late.** GitHub's scheduled runs often start 5 to 30 minutes late, and
  global cues refresh every 30 minutes instead of 10.
- **NSE may refuse GitHub's US servers.** Holiday list, FII/DII and bhavcopy may then show as
  unavailable. The fallback: commit `imports/nse_holidays_2026.csv` and
  `imports/ind_nifty500list.csv`. They're public exchange files, and `.gitignore` allows these
  two names.
- **Phase 3 intraday alerts (every minute) won't fit here.** At that point you'll need a small
  always-on machine (e.g. a free Oracle Cloud VM running the Docker setup).
- **GitHub pauses schedules after 60 days without commits.** Re-enable from the Actions tab.

Free-tier limits (check the providers' current pages): Vercel Hobby, Neon about 0.5 GB
(5 years × 500 stocks of daily bars is ~100 MB), GitHub Actions free on public repos and
2,000 min/month on private ones (Dalal uses ~1,200).

## When something goes wrong

The workflow stops with a red run and a plain-English message in its Summary.

| Message | Fix |
|---|---|
| "VERCEL_TOKEN isn't set" | Step 3 above |
| "Couldn't create Vercel project" | The token expired or has the wrong scope. Make a new one |
| "Couldn't create or connect the Neon database" | Vercel → `dalal-api` → Storage → Create Database → Neon (Free) → Connect, then re-run |
| "SECRET_KEY … is marked Sensitive" | Someone marked it Sensitive in Vercel. The jobs need to read it |
| "/healthz didn't answer" | Vercel → `dalal-api` → Logs shows the error |
| Login page says "Can't reach the API" | Re-run the deploy; `BACKEND_URL` on `dalal` must point at `dalal-api` |

## The manual way (Vercel dashboard)

If you'd rather click than run the workflow:

1. Vercel → **Add New → Project** → import this repo → name `dalal-api`, **Root Directory**
   `backend`, Framework **Other**. Variables: `SECRET_KEY` (64 random chars), `SETUP_TOKEN`
   (24+ random chars), `SCHEDULER_MODE=external`, `COOKIE_SECURE=true`. Deploy.
2. `dalal-api` → **Storage → Create Database → Neon (Free)** → Connect → Redeploy.
3. Run the migrations once. GitHub → Settings → Secrets → add `VERCEL_TOKEN`, then Actions →
   **Dalal jobs** → Run workflow → `migrate`. Or put `DATABASE_URL` (the unpooled Neon URL) and
   `SECRET_KEY` there as secrets instead.
4. **Add New → Project** → same repo → name `dalal`, **Root Directory** `frontend`,
   Framework **Next.js**, variable `BACKEND_URL` = the `dalal-api` URL. Deploy. That's your link.
5. On `dalal-api`, add `APP_BASE_URL` = your link and redeploy. Then do steps 5 to 7 of the
   short way.

Vercel's Python functions may be up to 500 MB uncompressed
([Vercel KB](https://vercel.com/kb/guide/troubleshooting-function-250mb-limit)).
Dalal's API bundle is well under that.

#!/usr/bin/env bash
# Deploy Dalal to Vercel's free tier: an API project (backend/) and a web project
# (frontend/), backed by a free Neon Postgres database. Run by
# .github/workflows/vercel-deploy.yml; see DEPLOY_VERCEL.md.
#
# Idempotent. The first run creates both projects, the database, SECRET_KEY and
# SETUP_TOKEN; later runs migrate and redeploy. SECRET_KEY is never rotated: it
# encrypts the stored TOTP seed and Kite token.
#
# Needs VERCEL_TOKEN. Optional: VERCEL_SCOPE (team slug), DALAL_API_PROJECT,
# DALAL_WEB_PROJECT (existing projects are reused and re-pointed at backend/ and
# frontend/), DALAL_DB_NAME, DALAL_SKIP_HEALTHCHECK=1 (local tests only).
#
# Helpers set variables rather than echoing results through $(...): fail() inside a
# command substitution would swallow its own error message.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/.github/scripts"
API=${DALAL_API_PROJECT:-dalal-api}
WEB=${DALAL_WEB_PROJECT:-dalal}
# Variables are set for production and preview, so Vercel's own Git builds of pull
# requests get the same settings as production deploys.
TARGETS=production,preview
DB_NAME=${DALAL_DB_NAME:-dalal-db}
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}
OUTPUT=${GITHUB_OUTPUT:-/dev/null}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

: "${VERCEL_TOKEN:?VERCEL_TOKEN is not set}"

vc() { vercel --token "$VERCEL_TOKEN" --non-interactive ${VERCEL_SCOPE:+--scope "$VERCEL_SCOPE"} "$@"; }
py() { python3 "$HERE/vercel_env.py" "$@"; }
step() { echo; echo "== $*"; }
fail() {
  echo "::error::$*"
  { echo "### Deploy stopped"; echo; echo "$*"; } >> "$SUMMARY"
  exit 1
}

# Production settings of the API project in $WORK/api.env. Database URLs stored as
# Sensitive can't be pulled for production; the development environment (which the
# Neon integration also connects) is used for those instead.
pull_api_env() {
  vc env pull "$WORK/api.env" --environment production --project "$API" --yes >/dev/null \
    || fail "Couldn't read environment variables of Vercel project $API."
  if vc env pull "$WORK/api-dev.env" --environment development --project "$API" --yes \
    >/dev/null 2>&1; then
    py fill "$WORK/api.env" "$WORK/api-dev.env"
  fi
  py mask "$WORK/api.env"
}

# set_var NAME VALUE PROJECT [--force]. Stored as readable config (not Sensitive),
# because the scheduled jobs pull these values to run.
set_var() {
  local name=$1 value=$2 project=$3
  shift 3
  vc env add "$name" "$TARGETS" --value "$value" --project "$project" --no-sensitive --yes "$@" \
    >/dev/null || fail "Couldn't set $name on Vercel project $project."
}

ensure_project() {
  if vc project inspect "$1" >/dev/null 2>&1; then
    echo "project $1: exists"
  elif vc project create "$1" >/dev/null; then
    echo "project $1: created"
  elif vc project inspect "$1" >/dev/null 2>&1; then
    echo "project $1: exists"
  else
    fail "Couldn't create Vercel project '$1'. Check that VERCEL_TOKEN is valid and has access to your account."
  fi
}

# Both projects build from this monorepo: point each at its folder, so Vercel's Git
# integration (if connected) builds the same thing this script deploys.
configure_project() { # PROJECT ROOT_DIR FRAMEWORK
  vc project update "$1" --root-directory "$2" --framework "$3" --yes >/dev/null \
    || fail "Couldn't set Root Directory '$2' on Vercel project '$1'."
  echo "project $1: root directory $2, framework $3"
}

# deploy PROJECT [flags...]: sets DEPLOYED_URL to the production URL Vercel reports.
# Deploys from the repo root: Vercel applies the project's Root Directory itself.
deploy() {
  local project=$1 out="$WORK/deploy-$1.json"
  shift
  if ! vc deploy "$ROOT" --prod --yes --project "$project" "$@" >"$out" 2>"$out.err"; then
    cat "$out.err" >&2 || true
    fail "Vercel deploy of $project failed: $(py deploy "$out" 2>&1 >/dev/null || true)"
  fi
  if ! DEPLOYED_URL=$(py deploy "$out" 2>"$out.msg"); then
    DEPLOYED_URL=""
    echo "::warning::$project deployed but Vercel reported no production URL: $(cat "$out.msg")"
  fi
}

# prod_url PROJECT DEPLOY_REPORTED_URL: sets PROD_URL. Never falls back to a
# per-deployment URL, which sits behind Vercel Authentication.
prod_url() {
  local found
  if found=$(vc api "/v9/projects/$1/domains" --raw 2>/dev/null | py domain); then
    PROD_URL=$found
  elif [ -n "$2" ]; then
    PROD_URL=$2
  else
    fail "Couldn't find the production domain of $1. Open Vercel → $1 → Settings → Domains, then re-run."
  fi
}

step "Projects"
ensure_project "$API"
ensure_project "$WEB"
configure_project "$API" backend other
configure_project "$WEB" frontend nextjs

step "Database"
pull_api_env
if ! py get "$WORK/api.env" DATABASE_URL POSTGRES_URL >/dev/null; then
  if py has "$WORK/api.env" DATABASE_URL || py has "$WORK/api.env" POSTGRES_URL; then
    fail "$API has a database, but its URL is stored as Sensitive, so this workflow can't run migrations. In Vercel → $API → Settings → Environment Variables, make DATABASE_URL non-sensitive (or re-add it), then re-run."
  fi
  echo "No database on $API yet; creating a free Neon Postgres database ($DB_NAME)."
  vc integration accept-terms neon --yes >/dev/null 2>&1 || true
  vc integration add neon --name "$DB_NAME" --no-connect --no-env-pull \
    || echo "::warning::'integration add' failed; it may already exist. Trying to connect it."
  if ! vc integration resource connect "$DB_NAME" "$API" --yes; then
    vc integration add neon --help || true
    fail "Couldn't create or connect the Neon database automatically. In Vercel, open $API → Storage → Create Database → Neon (Free plan) → Connect, then re-run this workflow."
  fi
  pull_api_env
  py get "$WORK/api.env" DATABASE_URL POSTGRES_URL >/dev/null \
    || fail "Neon was connected but no readable DATABASE_URL reached $API. Check $API → Storage in Vercel, then re-run."
fi
echo "database: connected"

step "App settings"
NEW_SETUP_TOKEN=0
if ! py has "$WORK/api.env" SECRET_KEY; then
  value=$(openssl rand -hex 32)
  echo "::add-mask::$value"
  set_var SECRET_KEY "$value" "$API"
  echo "SECRET_KEY: created"
fi
if ! py has "$WORK/api.env" SETUP_TOKEN; then
  value=$(openssl rand -hex 16)
  echo "::add-mask::$value"
  set_var SETUP_TOKEN "$value" "$API"
  NEW_SETUP_TOKEN=1
  echo "SETUP_TOKEN: created"
fi
set_var SCHEDULER_MODE external "$API" --force
set_var COOKIE_SECURE true "$API" --force
pull_api_env
SECRET=$(py get "$WORK/api.env" SECRET_KEY) \
  || fail "SECRET_KEY on $API is stored as Sensitive, so scheduled jobs can't read it. In Vercel → $API → Settings → Environment Variables, change it to a normal variable with the same value, then re-run."

step "Migrate"
DB=$(py get "$WORK/api.env" DATABASE_URL_UNPOOLED POSTGRES_URL_NON_POOLING DATABASE_URL POSTGRES_URL)
(cd "$ROOT/backend" && DATABASE_URL="$DB" SECRET_KEY="$SECRET" python -m app.cli migrate) \
  || fail "Database migration failed; see the log above."

step "Deploy API"
REGION=$(py region "$WORK/api.env")
echo "functions region: $REGION (next to the database)"
# Same region for Vercel's own Git builds (best effort; the deploy flag is what counts).
vc api "/v9/projects/$API" -X PATCH -f "serverlessFunctionRegion=$REGION" >/dev/null 2>&1 \
  || echo "::warning::Couldn't set the default function region on $API; Git builds may use iad1."
deploy "$API" --regions "$REGION"
prod_url "$API" "$DEPLOYED_URL"
API_URL=$PROD_URL
echo "API: $API_URL"

step "Deploy web"
set_var BACKEND_URL "$API_URL" "$WEB" --force
deploy "$WEB" --build-env "BACKEND_URL=$API_URL"
prod_url "$WEB" "$DEPLOYED_URL"
WEB_URL=$PROD_URL
echo "web: $WEB_URL"

if [ "$(py get "$WORK/api.env" APP_BASE_URL || true)" != "$WEB_URL" ]; then
  step "Point the API at the web URL"
  set_var APP_BASE_URL "$WEB_URL" "$API" --force
  deploy "$API" --regions "$REGION"
fi

NEEDS_SETUP=unknown
if [ "${DALAL_SKIP_HEALTHCHECK:-}" != "1" ]; then
  step "Health check"
  ok=0
  for _ in $(seq 1 12); do
    if curl -fsS --max-time 20 "$WEB_URL/healthz" 2>/dev/null | grep -q '"ok":true'; then
      ok=1
      break
    fi
    sleep 5
  done
  [ "$ok" = 1 ] || fail "Deployed, but $WEB_URL/healthz didn't answer. Open Vercel → $API → Logs to see why."
  echo "healthz: ok"
  if curl -fsS --max-time 20 "$WEB_URL/api/auth/setup-status" 2>/dev/null | grep -q '"needs_setup":true'; then
    NEEDS_SETUP=yes
  else
    NEEDS_SETUP=no
  fi
fi

echo "url=$WEB_URL" >> "$OUTPUT"
{
  echo "## Dalal is live"
  echo
  echo "**Link:** $WEB_URL"
  echo
  if [ "$NEEDS_SETUP" != "no" ]; then
    echo "**Create your login (once):**"
    echo "1. In Vercel, open **$API → Settings → Environment Variables** and copy \`SETUP_TOKEN\`."
    echo "2. Open \`$WEB_URL/setup?token=<SETUP_TOKEN>\`, choose a username and password, scan the QR code."
    echo
  fi
  echo "**Next:**"
  echo "- Kite developer console → Redirect URL: \`$WEB_URL/api/kite/callback\`"
  echo "- Add \`KITE_API_KEY\`, \`KITE_API_SECRET\`, \`KITE_USER_ID\`, \`TELEGRAM_BOT_TOKEN\`, \`TELEGRAM_CHAT_ID\` in **$API → Settings → Environment Variables** (as normal, not Sensitive, so the scheduled jobs can read them), then re-run this workflow."
  echo "- Scheduled jobs start once this branch is merged into \`main\` (GitHub only runs schedules there)."
} >> "$SUMMARY"
if [ "$NEW_SETUP_TOKEN" = 1 ]; then
  echo "A new SETUP_TOKEN was created; it's in Vercel, not in this log."
fi
echo
echo "Dalal: $WEB_URL"

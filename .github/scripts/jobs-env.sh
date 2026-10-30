#!/usr/bin/env bash
# Prepare settings for a scheduled job run (.github/workflows/dalal-jobs.yml).
#
# With VERCEL_TOKEN set, settings come from the dalal-api project on Vercel, so they
# live in one place. Otherwise from GitHub secrets (GH_* variables). Writes KEY=value
# lines to $GITHUB_ENV and configured=true|false to $GITHUB_OUTPUT. No database yet
# means "not deployed": the job does nothing and stays green.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
API=${DALAL_API_PROJECT:-dalal-api}
ENV_OUT=${GITHUB_ENV:-/dev/stdout}
OUTPUT=${GITHUB_OUTPUT:-/dev/stdout}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

not_configured() {
  echo "::notice::$1"
  echo "configured=false" >> "$OUTPUT"
  exit 0
}

if [ -n "${VERCEL_TOKEN:-}" ]; then
  command -v vercel >/dev/null || npm install -g --silent --no-audit --no-fund "vercel@${VERCEL_CLI_VERSION:-64.5.0}" >/dev/null
  vc() { vercel --token "$VERCEL_TOKEN" --non-interactive ${VERCEL_SCOPE:+--scope "$VERCEL_SCOPE"} "$@"; }
  vc env pull "$WORK/api.env" --environment production --project "$API" --yes >/dev/null \
    || not_configured "Couldn't read settings from Vercel project $API. Run the 'Deploy to Vercel' workflow first."
  # Sensitive database URLs can't be pulled for production; use development's (same DB).
  if vc env pull "$WORK/api-dev.env" --environment development --project "$API" --yes \
    >/dev/null 2>&1; then
    python3 "$HERE/vercel_env.py" fill "$WORK/api.env" "$WORK/api-dev.env"
  fi
  python3 "$HERE/vercel_env.py" mask "$WORK/api.env"
  status=0
  python3 "$HERE/vercel_env.py" export "$WORK/api.env" > "$WORK/export" || status=$?
  case $status in
    0) cat "$WORK/export" >> "$ENV_OUT" ;;
    3) not_configured "Vercel project $API has no database yet. Run the 'Deploy to Vercel' workflow." ;;
    *) exit "$status" ;;  # settings exist but are unreadable: a real misconfiguration
  esac
else
  [ -n "${GH_DATABASE_URL:-}" ] \
    || not_configured "Dalal isn't deployed yet (no VERCEL_TOKEN or DATABASE_URL secret); nothing to run."
  for pair in DATABASE_URL:GH_DATABASE_URL REDIS_URL:GH_REDIS_URL SECRET_KEY:GH_SECRET_KEY \
    KITE_API_KEY:GH_KITE_API_KEY KITE_API_SECRET:GH_KITE_API_SECRET KITE_USER_ID:GH_KITE_USER_ID \
    TELEGRAM_BOT_TOKEN:GH_TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID:GH_TELEGRAM_CHAT_ID \
    ANTHROPIC_API_KEY:GH_ANTHROPIC_API_KEY APP_BASE_URL:GH_APP_BASE_URL WATCHLIST:GH_WATCHLIST; do
    name=${pair%%:*}
    src=${pair#*:}
    [ -n "${!src:-}" ] && echo "$name=${!src}" >> "$ENV_OUT"
  done
  [ -n "${GH_REDIS_URL:-}" ] || echo "CACHE_BACKEND=postgres" >> "$ENV_OUT"
fi
echo "configured=true" >> "$OUTPUT"

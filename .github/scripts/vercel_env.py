"""Helpers for the Vercel deploy and job workflows. Reads `vercel env pull` files
without evaluating them in a shell, so values containing $ or quotes are safe.

  vercel_env.py has FILE KEY              exit 0 if KEY is present (even if Sensitive)
  vercel_env.py get FILE KEY [KEY ...]    print the first readable value
  vercel_env.py fill FILE FALLBACK        copy readable database URLs from FALLBACK into
                                          FILE where FILE's are missing or Sensitive
  vercel_env.py mask FILE                 print ::add-mask:: for private values
  vercel_env.py export FILE               print app settings as $GITHUB_ENV lines;
                                          exit 3 if no database, 4 if it's unreadable
  vercel_env.py region FILE               Vercel region code closest to the database
  vercel_env.py domain < projects-domains.json
                                          print https://<production domain>
  vercel_env.py deploy FILE               read `vercel deploy` JSON output: print the
                                          production URL, or the error and exit 1
  vercel_env.py neon-product < integration.json
                                          print "<integration id> <product id> <add target>"
  vercel_env.py free-plan < plans.json    print the id of the free billing plan; exit 1
                                          if there is none (never pick a paid one)

`vercel env pull` writes the placeholder "[SENSITIVE]" for variables stored as
Sensitive; those are treated as present but unreadable, never as real values.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SECRETISH = re.compile(
    r"SECRET|TOKEN|PASSWORD|KEY|URL|PG|POSTGRES|DATABASE|KV_|REDIS|USER_ID|CHAT_ID|WATCHLIST",
    re.I,
)
UNREADABLE = "[SENSITIVE]"
# Public by design; masking them would hide your own link in the logs.
PUBLIC = {"APP_BASE_URL", "BACKEND_URL"}

# Settings the app reads (app/config.py), copied from the API project into job runs.
APP_KEYS = [
    "SECRET_KEY",
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_USER_ID",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL_WRITER",
    "ANTHROPIC_MODEL_FAST",
    "FRED_API_KEY",
    "NEWSAPI_KEY",
    "APP_BASE_URL",
    "WATCHLIST",
    "UNIVERSE",
    "REDIS_URL",
    "KV_URL",
]
# Direct (unpooled) connections first: migrations and long jobs shouldn't go through
# a transaction-mode pooler.
DB_KEYS = ["DATABASE_URL_UNPOOLED", "POSTGRES_URL_NON_POOLING", "DATABASE_URL", "POSTGRES_URL"]

# AWS region in a Neon hostname -> nearest Vercel Functions region.
AWS_TO_VERCEL = {
    "us-east-1": "iad1",
    "us-east-2": "cle1",
    "us-west-1": "sfo1",
    "us-west-2": "pdx1",
    "ca-central-1": "yul1",
    "sa-east-1": "gru1",
    "eu-central-1": "fra1",
    "eu-west-1": "dub1",
    "eu-west-2": "lhr1",
    "eu-west-3": "cdg1",
    "eu-north-1": "arn1",
    "ap-south-1": "bom1",
    "ap-southeast-1": "sin1",
    "ap-southeast-2": "syd1",
    "ap-northeast-1": "hnd1",
    "ap-northeast-2": "icn1",
    "ap-northeast-3": "kix1",
    "ap-east-1": "hkg1",
    "af-south-1": "cpt1",
    "me-central-1": "dxb1",
}
DEFAULT_REGION = "iad1"


def parse(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            quote, value = value[0], value[1:-1]
            if quote == '"':
                # The CLI escapes only line breaks (escapeValue in vercel's env pull);
                # quotes and backslashes are written as they are.
                value = value.replace("\\n", "\n").replace("\\r", "\r")
        out[key] = value
    return out


def quoted(value: str) -> str:
    """A value written the way `vercel env pull` writes it, so parse() reads it back."""
    return '"' + value.replace("\n", "\\n").replace("\r", "\\r") + '"'


def readable(env: dict[str, str], key: str) -> str | None:
    value = env.get(key)
    return value if value and value != UNREADABLE else None


def first(env: dict[str, str], keys: list[str]) -> str | None:
    for k in keys:
        if readable(env, k):
            return env[k]
    return None


def env_line(key: str, value: str) -> str:
    if "\n" in value:
        return f"{key}<<__DALAL_EOF__\n{value}\n__DALAL_EOF__"
    return f"{key}={value}"


def region_for(env: dict[str, str]) -> str:
    url = first(env, DB_KEYS)
    if not url:
        return DEFAULT_REGION
    host = urlsplit(url).hostname or ""
    m = re.search(r"\.([a-z]{2}-[a-z]+-\d)\.aws\.neon\.tech$", host)
    if m and m.group(1) in AWS_TO_VERCEL:
        return AWS_TO_VERCEL[m.group(1)]
    return DEFAULT_REGION


def production_domain(payload: dict[str, object]) -> str | None:
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return None
    usable = [
        d
        for d in domains
        if isinstance(d, dict)
        and d.get("name")
        and not d.get("redirect")
        and not d.get("gitBranch")
        and d.get("verified", True)
    ]
    # Prefer a custom domain if you add one later; otherwise the shortest *.vercel.app.
    custom = [d for d in usable if not str(d["name"]).endswith(".vercel.app")]
    pool = custom or usable
    if not pool:
        return None
    return "https://" + str(min(pool, key=lambda d: len(str(d["name"])))["name"])


def deploy_result(text: str) -> tuple[str | None, str]:
    """(production URL or None, message) from `vercel deploy --non-interactive` output."""
    try:
        start = text.index("{")
        payload = json.loads(text[start : text.rindex("}") + 1])
    except ValueError:
        return None, text.strip()[-500:] or "no output from vercel deploy"
    message = str(payload.get("message") or payload.get("reason") or "")
    if payload.get("status") != "ok":
        return None, message or "vercel deploy failed"
    deployment = payload.get("deployment") or {}
    return deployment.get("productionUrl") or None, message


def dicts(value: object) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def neon_product(integration: dict[str, Any]) -> tuple[str, str, str] | None:
    """(integration id, product id, `integration add` target) for Neon's Postgres product."""
    products = dicts(integration.get("products"))
    if not integration.get("id") or not products:
        return None
    postgres = [p for p in products if "postgres" in f"{p.get('slug')} {p.get('name')}".lower()]
    product = (postgres or products)[0]
    slug = str(integration.get("slug") or "neon")
    target = slug if len(products) == 1 else f"{slug}/{product.get('slug')}"
    return str(integration["id"]), str(product.get("id")), target


FREE = re.compile(r"\bfree\b", re.I)
# The whole cost must be zero: "$0", "$0.00/month", "Free". "$0.106/CU-hour" is not.
ZERO_COST = re.compile(r"^\s*(free|\$?\s*0+(\.0+)?)\s*(/\s*[a-z]+)?\s*$", re.I)


def free_plan(payload: dict[str, Any]) -> str | None:
    """The id of the free billing plan. Dalal never creates anything that costs money."""
    plans = [p for p in dicts(payload.get("plans")) if not p.get("disabled")]
    for plan in plans:
        cost = str(plan.get("cost") or "")
        if (
            FREE.search(str(plan.get("name") or ""))
            and (not cost or ZERO_COST.match(cost))
            and plan.get("paymentMethodRequired") is not True
        ):
            return str(plan["id"])
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "deploy":
        url, message = deploy_result(Path(argv[2]).read_text(encoding="utf-8"))
        if url is None:
            print(message, file=sys.stderr)
            return 1
        print(url)
        return 0
    if cmd == "neon-product":
        found_product = neon_product(json.load(sys.stdin))
        if not found_product:
            return 1
        print(" ".join(found_product))
        return 0
    if cmd == "free-plan":
        plan = free_plan(json.load(sys.stdin))
        if not plan:
            return 1
        print(plan)
        return 0
    if cmd == "domain":
        found = production_domain(json.load(sys.stdin))
        if not found:
            return 1
        print(found)
        return 0
    env = parse(argv[2])
    if cmd == "has":
        return 0 if argv[3] in env else 1
    if cmd == "get":
        value = first(env, argv[3:])
        if value is None:
            return 1
        print(value)
        return 0
    if cmd == "fill":
        fallback = parse(argv[3])
        filled = [k for k in DB_KEYS if not readable(env, k) and readable(fallback, k)]
        if filled:
            with Path(argv[2]).open("a", encoding="utf-8") as fh:
                for k in filled:
                    fh.write(f"{k}={quoted(fallback[k])}\n")
            print(f"read {', '.join(filled)} from the development environment", file=sys.stderr)
        return 0
    if cmd == "mask":
        for k, v in env.items():
            if k not in PUBLIC and v != UNREADABLE and SECRETISH.search(k) and len(v) >= 4:
                for part in v.splitlines():
                    if part:
                        print(f"::add-mask::{part}")
        return 0
    if cmd == "region":
        print(region_for(env))
        return 0
    if cmd == "export":
        db = first(env, DB_KEYS)
        if not db:
            if any(env.get(k) == UNREADABLE for k in DB_KEYS):
                print("::error::The database URL is stored as Sensitive in Vercel, so jobs "
                      "can't read it.", file=sys.stderr)
                return 4
            return 3
        hidden = [k for k in APP_KEYS if env.get(k) == UNREADABLE]
        if "SECRET_KEY" in hidden:
            print("::error::SECRET_KEY is stored as Sensitive in Vercel, so jobs can't read it.",
                  file=sys.stderr)
            return 4
        if hidden:
            print(f"::warning::Sensitive in Vercel, so not available to jobs: {', '.join(hidden)}",
                  file=sys.stderr)
        lines = [env_line("DATABASE_URL", db)]
        lines += [env_line(k, env[k]) for k in APP_KEYS if readable(env, k)]
        # Same choice Settings.kv_backend makes on Vercel, so jobs and the API share one
        # store (NSE throttle slots and block cooldowns live there).
        forced = (readable(env, "CACHE_BACKEND") or "").strip().lower()
        if forced not in {"redis", "postgres"}:
            has_redis = readable(env, "REDIS_URL") or readable(env, "KV_URL")
            forced = "redis" if has_redis else "postgres"
        lines.append(f"CACHE_BACKEND={forced}")
        print("\n".join(lines))
        return 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env python3
"""Stop the deploy early, with a plain-English fix, if VERCEL_TOKEN can't do the job.

The Vercel CLI's first call reads the token's user (GET /v2/user). Project-scoped and
team-scoped tokens can't, and the CLI then dies with "User not found". So this checks
the user, the team and both projects first. It logs only HTTP statuses and Vercel's
error codes: never the token, user names or emails (Actions logs on a public repo are
public).

Env: VERCEL_TOKEN, VERCEL_TEAM_ID (team_..., optional), DALAL_API_PROJECT,
DALAL_WEB_PROJECT. VERCEL_API_URL overrides the API origin (tests only).
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("VERCEL_API_URL", "https://api.vercel.com")
NEW_TOKEN = (
    "Create a new token at vercel.com → Account Settings → Tokens with **Scope: Full Account** "
    "(the Vercel CLI must be able to read your user), replace the `VERCEL_TOKEN` repository "
    "secret with it, delete the old token, and re-run this workflow."
)


def get(path: str, token: str) -> tuple[int, dict]:
    req = urllib.request.Request(
        API + path, headers={"Authorization": f"Bearer {token}", "User-Agent": "dalal-deploy"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return 0, {"error": {"code": "network", "message": str(e)[:120]}}


def error(body: dict) -> dict:
    err = body.get("error")
    return err if isinstance(err, dict) else {}


def describe(status: int, body: dict) -> str:
    if 200 <= status < 300:
        return f"{status} ok"
    err = error(body)
    return f"{status} {str(err.get('code', ''))[:40]} {str(err.get('message', ''))[:120]}".strip()


def diagnose() -> str | None:
    """Returns what's wrong with the token, or None if the deploy can use it."""
    raw = os.environ.get("VERCEL_TOKEN", "")
    token = raw.strip()
    if not token:
        return "VERCEL_TOKEN is empty. " + NEW_TOKEN
    if token != raw or any(c.isspace() for c in token):
        return (
            "VERCEL_TOKEN contains spaces or line breaks. Paste only the token into the "
            "`VERCEL_TOKEN` repository secret, then re-run this workflow."
        )
    print(f"token: {len(token)} chars")

    user_status, user_body = get("/v2/user", token)
    print(f"GET /v2/user: {describe(user_status, user_body)}")
    team = os.environ.get("VERCEL_TEAM_ID", "")
    team_status = 0
    if team:
        team_status, body = get(f"/v2/teams/{urllib.parse.quote(team)}", token)
        print(f"GET /v2/teams/{team}: {describe(team_status, body)}")
    visible = []
    for name in filter(
        None, (os.environ.get("DALAL_API_PROJECT"), os.environ.get("DALAL_WEB_PROJECT"))
    ):
        query = f"?teamId={urllib.parse.quote(team)}" if team else ""
        status, body = get(f"/v9/projects/{urllib.parse.quote(name)}{query}", token)
        print(f"GET /v9/projects/{name}: {describe(status, body)}")
        if status == 200:
            visible.append(name)

    if user_status == 0:
        return f"Couldn't reach {API} ({error(user_body).get('message')}). Re-run this workflow."
    if user_status in (401, 403) and error(user_body).get("invalidToken"):
        return "VERCEL_TOKEN is invalid, expired or revoked. " + NEW_TOKEN
    if user_status != 200:
        if visible and team_status != 200:
            scope = f"is limited to the project {', '.join(visible)}"
        elif team_status == 200:
            scope = "is limited to a team"
        else:
            scope = f"can't read your Vercel user (HTTP {user_status})"
        return f"VERCEL_TOKEN {scope}, so the Vercel CLI can't use it. " + NEW_TOKEN
    if team and team_status != 200:
        return (
            f"VERCEL_TOKEN works but can't reach the Vercel team {team} that owns the projects "
            f"(HTTP {team_status}). Create the token while signed in to the account that owns "
            "that team, or set the repository variables VERCEL_SCOPE and VERCEL_TEAM_ID to "
            "your own team, then re-run this workflow."
        )
    return None


def main() -> int:
    problem = diagnose()
    if problem is None:
        print("VERCEL_TOKEN: ok")
        return 0
    print(f"::error::{problem}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Deploy stopped: the Vercel token can't be used\n\n{problem}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

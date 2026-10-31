#!/usr/bin/env python3
"""Check what VERCEL_TOKEN can reach before the deploy uses it.

Prints one line per probe: HTTP status plus Vercel's error code and message. Never
prints the token, user names or emails (Actions logs on a public repo are public).
Writes user_ok, team_ok and projects_ok to $GITHUB_OUTPUT.

Env: VERCEL_TOKEN, VERCEL_TEAM_ID (team_...), DALAL_API_PROJECT, DALAL_WEB_PROJECT.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.vercel.com"


def request(url: str, token: str, data: bytes | None = None) -> tuple[int, dict]:
    headers = {"User-Agent": "dalal-deploy-check"}
    if data is None:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers)
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


def describe(status: int, body: dict) -> str:
    if 200 <= status < 300:
        return f"{status} ok"
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    code = str(err.get("code", ""))[:40]
    message = str(err.get("message", ""))[:120]
    flags = [k for k in ("invalidToken", "missingToken", "saml", "scope") if k in err]
    return f"{status} {code} {message}".strip() + (f" {flags}" if flags else "")


def main() -> int:
    raw = os.environ.get("VERCEL_TOKEN", "")
    token = raw.strip()
    team = os.environ.get("VERCEL_TEAM_ID", "")
    projects = [p for p in (os.environ.get("DALAL_API_PROJECT"), os.environ.get("DALAL_WEB_PROJECT")) if p]
    out = os.environ.get("GITHUB_OUTPUT")

    print(f"token: {len(token)} chars" + (", had surrounding whitespace" if raw != token else ""))
    if any(c.isspace() for c in token):
        print("::error::VERCEL_TOKEN contains spaces or line breaks. Paste only the token.")
        return 1

    status, body = request(f"{API}/v2/user", token)
    user_ok = status == 200
    print(f"GET /v2/user: {describe(status, body)}")

    team_ok = False
    if team:
        status, body = request(f"{API}/v2/teams/{team}", token)
        team_ok = status == 200
        print(f"GET /v2/teams/<team>: {describe(status, body)}")

    status, body = request(f"{API}/v2/teams?limit=20", token)
    teams = body.get("teams") if isinstance(body.get("teams"), list) else []
    print(f"GET /v2/teams: {describe(status, body)}, {len(teams)} team(s) visible")

    projects_ok = bool(projects)
    for name in projects:
        query = f"?teamId={urllib.parse.quote(team)}" if team else ""
        status, body = request(f"{API}/v9/projects/{urllib.parse.quote(name)}{query}", token)
        projects_ok &= status == 200
        print(f"GET /v9/projects/{name}: {describe(status, body)}")

    # The CLI's own fallback for tokens without a user (app tokens) relies on this.
    status, body = request("https://vercel.com/.well-known/openid-configuration", "")
    endpoint = body.get("introspection_endpoint") if status == 200 else None
    if endpoint:
        status, body = request(endpoint, "", urllib.parse.urlencode({"token": token}).encode())
        team_info = body.get("team") if isinstance(body.get("team"), dict) else {}
        print(
            f"token introspection: {describe(status, body)}, active={body.get('active')}, "
            f"app_token={bool(body.get('client_id'))}, team_matches={team_info.get('id') == team}, "
            f"fields={sorted(body)}"
        )
    else:
        print(f"token introspection: discovery {describe(status, body)}")

    if out:
        with open(out, "a") as f:
            f.write(f"user_ok={str(user_ok).lower()}\nteam_ok={str(team_ok).lower()}\n")
            f.write(f"projects_ok={str(projects_ok).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

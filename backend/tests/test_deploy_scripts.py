"""The GitHub Actions deploy/jobs scripts, run against a fake Vercel CLI.

The real Vercel API can't be reached from tests; tests/fixtures/fake_vercel.py mimics
the CLI commands the scripts call and records them, so the control flow (first run,
re-run, failure paths, secret handling) is exercised for real, including migrations.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FIXTURES, TEST_DB

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / ".github" / "scripts"


def _helper() -> Any:
    spec = importlib.util.spec_from_file_location("vercel_env", SCRIPTS / "vercel_env.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- vercel_env.py


def test_parse_env_pull_file_without_shell_evaluation(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        '# Created by Vercel CLI\nA="plain"\nB="has $(rm -rf /) and \\"quotes\\""\n'
        'C=unquoted\nD="multi\\nline"\nEMPTY=""\n'
    )
    env = _helper().parse(str(f))
    assert env["A"] == "plain"
    assert env["B"] == 'has $(rm -rf /) and "quotes"'
    assert env["C"] == "unquoted"
    assert env["D"] == "multi\nline"
    assert env["EMPTY"] == ""


@pytest.mark.parametrize(
    ("host", "region"),
    [
        ("ep-cool-1.ap-southeast-1.aws.neon.tech", "sin1"),
        ("ep-cool-1-pooler.us-east-1.aws.neon.tech", "iad1"),
        ("ep-x.ap-south-1.aws.neon.tech", "bom1"),
        ("ep-x.eu-central-1.aws.neon.tech", "fra1"),
        ("db.example.com", "iad1"),
    ],
)
def test_region_follows_database(host: str, region: str) -> None:
    env = {"DATABASE_URL": f"postgresql://u:p@{host}/neondb?sslmode=require"}
    assert _helper().region_for(env) == region


def test_production_domain_skips_branch_aliases_and_prefers_custom() -> None:
    h = _helper()
    payload = {
        "domains": [
            {"name": "dalal-git-main-me.vercel.app", "gitBranch": "main", "verified": True},
            {"name": "dalal-me.vercel.app", "verified": True},
            {"name": "www.dalal.example", "redirect": "dalal.example", "verified": True},
        ]
    }
    assert h.production_domain(payload) == "https://dalal-me.vercel.app"
    payload["domains"].append({"name": "dalal.example", "verified": True})
    assert h.production_domain(payload) == "https://dalal.example"
    assert h.production_domain({"error": {"code": "forbidden"}}) is None


def test_mask_hides_secrets_but_not_public_urls(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        'SECRET_KEY="abcdef123456"\nDATABASE_URL="postgresql://u:pw@h/db"\n'
        'APP_BASE_URL="https://dalal-me.vercel.app"\nSCHEDULER_MODE="external"\n'
    )
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_env.py"), "mask", str(f)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "::add-mask::abcdef123456" in out
    assert "::add-mask::postgresql://u:pw@h/db" in out
    assert "dalal-me.vercel.app" not in out
    assert "external" not in out


def test_export_prefers_unpooled_db_and_falls_back_to_postgres_kv(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        'DATABASE_URL="postgres://pooled"\nDATABASE_URL_UNPOOLED="postgres://direct"\n'
        'SECRET_KEY="s"\nPGPASSWORD="x"\nUNRELATED="y"\n'
    )
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_env.py"), "export", str(f)],
        capture_output=True,
        text=True,
    )
    lines = run.stdout.splitlines()
    assert run.returncode == 0
    assert lines[0] == "DATABASE_URL=postgres://direct"
    assert "SECRET_KEY=s" in lines and "CACHE_BACKEND=postgres" in lines
    assert not any(line.startswith(("PGPASSWORD", "UNRELATED")) for line in lines)
    f.write_text('SECRET_KEY="s"\n')
    no_db = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_env.py"), "export", str(f)], capture_output=True
    )
    assert no_db.returncode == 3


# --------------------------------------------------------- vercel-deploy.sh


@pytest.fixture()
def fake_cli(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "vercel"
    shutil.copy(FIXTURES / "fake_vercel.py", target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC)
    python_dir = Path(sys.executable).parent
    db_url = TEST_DB.replace("postgresql+psycopg://", "postgresql://")
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{python_dir}:{os.environ['PATH']}",
        "FAKE_STATE": str(tmp_path / "state"),
        "FAKE_DB_URL": db_url,
        "VERCEL_TOKEN": "fake-token",
        "DALAL_SKIP_HEALTHCHECK": "1",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_ENV": str(tmp_path / "github_env"),
    }


def _deploy(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / "vercel-deploy.sh")],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )


def _calls(env: dict[str, str]) -> list[str]:
    return (Path(env["FAKE_STATE"]) / "calls.log").read_text().splitlines()


def _var(env: dict[str, str], project: str, name: str) -> str:
    return (Path(env["FAKE_STATE"]) / "env" / project / name).read_text()


def test_first_deploy_creates_everything_then_redeploy_is_idempotent(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    first = _deploy(fake_cli)
    assert first.returncode == 0, first.stdout + first.stderr
    calls = _calls(fake_cli)
    assert any("project create dalal-api" in c for c in calls)
    # Each project is pointed at its folder, so Vercel's own Git builds work too.
    assert any(
        "project update dalal-api --root-directory backend --framework other" in c for c in calls
    )
    assert any(
        "project update dalal --root-directory frontend --framework nextjs" in c for c in calls
    )
    deploys = [c for c in calls if " deploy " in f" {c} "]
    assert deploys and all(f"deploy {REPO} --prod" in c for c in deploys), "deploy from repo root"
    assert any("integration add neon --name dalal-db" in c for c in calls)
    assert any("integration resource connect dalal-db dalal-api --yes" in c for c in calls)
    secret = _var(fake_cli, "dalal-api", "SECRET_KEY")
    assert len(secret) == 64
    assert _var(fake_cli, "dalal-api", "SCHEDULER_MODE") == "external"
    assert _var(fake_cli, "dalal-api", "APP_BASE_URL") == "https://dalal-me.vercel.app"
    assert _var(fake_cli, "dalal", "BACKEND_URL") == "https://dalal-api-me.vercel.app"
    # The web build gets the API's production domain, not a per-deployment URL.
    assert any(
        "deploy" in c and "--build-env BACKEND_URL=https://dalal-api-me.vercel.app" in c
        for c in calls
    )
    summary = Path(fake_cli["GITHUB_STEP_SUMMARY"]).read_text()
    assert "https://dalal-me.vercel.app" in summary
    # Never print a generated secret outside a mask command.
    for line in (first.stdout + first.stderr).splitlines():
        if secret in line:
            assert line.startswith("::add-mask::"), line
    assert secret not in summary

    Path(fake_cli["FAKE_STATE"], "calls.log").unlink()
    second = _deploy(fake_cli)
    assert second.returncode == 0, second.stdout + second.stderr
    calls = _calls(fake_cli)
    assert not any("project create" in c or "integration add" in c for c in calls)
    assert _var(fake_cli, "dalal-api", "SECRET_KEY") == secret, "SECRET_KEY must never rotate"
    api_deploys = [c for c in calls if " deploy " in f" {c} " and "--project dalal-api" in c]
    assert len(api_deploys) == 1, "no extra API redeploy when APP_BASE_URL is unchanged"


def test_deploy_stops_with_instructions_when_neon_cannot_be_created(
    fake_cli: dict[str, str],
) -> None:
    res = _deploy({**fake_cli, "FAKE_NEON_FAIL": "1"})
    assert res.returncode == 1
    assert "Storage → Create Database → Neon" in res.stdout
    assert not any(" deploy " in c for c in _calls(fake_cli)), "nothing deployed without a DB"


def test_deploy_requires_token(fake_cli: dict[str, str]) -> None:
    env = {k: v for k, v in fake_cli.items() if k != "VERCEL_TOKEN"}
    res = _deploy(env)
    assert res.returncode != 0 and "VERCEL_TOKEN" in res.stderr


# ---------------------------------------------------------------- jobs-env.sh


def _jobs_env(env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str, str]:
    for k in ("GITHUB_ENV", "GITHUB_OUTPUT"):
        Path(env[k]).write_text("")
    res = subprocess.run(
        ["bash", str(SCRIPTS / "jobs-env.sh")], env=env, capture_output=True, text=True
    )
    return res, Path(env["GITHUB_ENV"]).read_text(), Path(env["GITHUB_OUTPUT"]).read_text()


def test_jobs_skip_quietly_until_deployed(fake_cli: dict[str, str]) -> None:
    res, _, out = _jobs_env(fake_cli)  # token set, but no database on Vercel yet
    assert res.returncode == 0 and "configured=false" in out
    no_token = {k: v for k, v in fake_cli.items() if k != "VERCEL_TOKEN"}
    res, _, out = _jobs_env(no_token)
    assert res.returncode == 0 and "configured=false" in out


def test_jobs_load_settings_from_vercel(migrated_db: None, fake_cli: dict[str, str]) -> None:
    assert _deploy(fake_cli).returncode == 0
    res, github_env, out = _jobs_env(fake_cli)
    assert res.returncode == 0, res.stderr
    assert "configured=true" in out
    assert "DATABASE_URL=" in github_env and "SECRET_KEY=" in github_env
    assert "CACHE_BACKEND=postgres" in github_env


def test_jobs_github_secrets_path(fake_cli: dict[str, str]) -> None:
    env = {k: v for k, v in fake_cli.items() if k != "VERCEL_TOKEN"}
    env |= {"GH_DATABASE_URL": "postgres://x", "GH_SECRET_KEY": "s" * 40, "GH_KITE_USER_ID": ""}
    res, github_env, out = _jobs_env(env)
    assert "configured=true" in out
    assert "DATABASE_URL=postgres://x" in github_env
    assert "KITE_USER_ID" not in github_env, "empty secrets are not exported"
    assert "CACHE_BACKEND=postgres" in github_env


def test_workflows_reference_existing_scripts() -> None:
    import yaml

    for wf in ("vercel-deploy.yml", "dalal-jobs.yml"):
        doc = yaml.safe_load((REPO / ".github" / "workflows" / wf).read_text())
        runs = json.dumps(doc)
        for script in ("vercel-deploy.sh", "jobs-env.sh"):
            if script in runs:
                assert (SCRIPTS / script).exists()


def test_workflows_pass_actionlint() -> None:
    """Catches what GitHub would reject at load time (e.g. contexts not allowed in env)."""
    linter = shutil.which("actionlint") or str(Path(sys.executable).parent / "actionlint")
    checker = shutil.which("shellcheck") or str(Path(sys.executable).parent / "shellcheck")
    if not Path(linter).exists():
        pytest.skip("actionlint not installed (pip install actionlint-py)")
    args = [linter]
    if Path(checker).exists():
        args.append(f"-shellcheck={checker}")
    res = subprocess.run(
        [*args, *sorted(str(p) for p in (REPO / ".github" / "workflows").glob("*.yml"))],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr


# ------------------------------------------- regressions from the adversarial review


def test_deploy_json_output_is_parsed_not_tail_of_pretty_json(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    """`vercel deploy --non-interactive` prints pretty JSON; `tail -n 1` gave '}'."""
    res = _deploy({**fake_cli, "FAKE_DOMAINS_FAIL": "1"})  # force the deploy-URL path
    assert res.returncode == 0, res.stdout + res.stderr
    assert _var(fake_cli, "dalal", "BACKEND_URL") == "https://dalal-api-me.vercel.app"
    assert _var(fake_cli, "dalal-api", "APP_BASE_URL") == "https://dalal-me.vercel.app"


def test_failed_deploy_reports_vercels_error(migrated_db: None, fake_cli: dict[str, str]) -> None:
    res = _deploy({**fake_cli, "FAKE_DEPLOY_FAIL": "dalal"})
    assert res.returncode == 1
    assert "::error::Vercel deploy of dalal failed: Build failed for dalal" in res.stdout
    summary = Path(fake_cli["GITHUB_STEP_SUMMARY"]).read_text()
    assert "pip resolution error" in summary


def test_sensitive_database_url_falls_back_to_development(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    env = {**fake_cli, "FAKE_SENSITIVE": "DATABASE_URL,DATABASE_URL_UNPOOLED"}
    res = _deploy(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Database migrated" in res.stdout


def test_sensitive_secret_key_stops_without_regenerating(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    assert _deploy(fake_cli).returncode == 0
    secret = _var(fake_cli, "dalal-api", "SECRET_KEY")
    res = _deploy({**fake_cli, "FAKE_SENSITIVE": "SECRET_KEY"})
    assert res.returncode == 1 and "SECRET_KEY on dalal-api is stored as Sensitive" in res.stdout
    assert _var(fake_cli, "dalal-api", "SECRET_KEY") == secret, "must never rotate SECRET_KEY"
    # The placeholder is never handed to anything as if it were the key.
    assert "[SENSITIVE]" not in res.stdout.replace("stored as Sensitive", "")


def test_jobs_fail_loudly_when_secret_key_is_sensitive(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    assert _deploy(fake_cli).returncode == 0
    res, github_env, out = _jobs_env({**fake_cli, "FAKE_SENSITIVE": "SECRET_KEY"})
    assert res.returncode == 4
    assert "configured=true" not in out and "SECRET_KEY" not in github_env


def test_jobs_skip_sensitive_optional_keys_with_warning(
    migrated_db: None, fake_cli: dict[str, str]
) -> None:
    assert _deploy(fake_cli).returncode == 0
    (Path(fake_cli["FAKE_STATE"]) / "env" / "dalal-api" / "KITE_API_KEY").write_text("kk-123")
    res, github_env, out = _jobs_env({**fake_cli, "FAKE_SENSITIVE": "KITE_API_KEY"})
    assert res.returncode == 0 and "configured=true" in out
    assert "KITE_API_KEY" not in github_env
    assert "Sensitive in Vercel, so not available to jobs: KITE_API_KEY" in res.stderr


def test_private_identifiers_are_masked(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('KITE_USER_ID="AB1234"\nTELEGRAM_CHAT_ID="987654321"\nWATCHLIST="INFY,TCS"\n')
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_env.py"), "mask", str(f)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for value in ("AB1234", "987654321", "INFY,TCS"):
        assert f"::add-mask::{value}" in out


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ("", "CACHE_BACKEND=postgres"),
        ('KV_URL="rediss://up"\n', "CACHE_BACKEND=redis"),
        ('KV_URL="rediss://up"\nCACHE_BACKEND="postgres"\n', "CACHE_BACKEND=postgres"),
    ],
)
def test_jobs_use_the_same_cache_backend_as_the_api(
    tmp_path: Path, extra: str, expected: str
) -> None:
    f = tmp_path / ".env"
    f.write_text('DATABASE_URL="postgres://x"\nSECRET_KEY="s"\n' + extra)
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_env.py"), "export", str(f)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert expected in out
    assert sum(line.startswith("CACHE_BACKEND=") for line in out) == 1


# --------------------------------------------------------- vercel_token_check.py

TOKEN = "tok_abcdefghijklmnopqrstuvwxyz0123456789"
TEAM = "team_test"
OK: tuple[int, dict[str, Any]] = (200, {})


def _not_found(what: str) -> tuple[int, dict[str, Any]]:
    return 404, {"error": {"code": "not_found", "message": f"{what} not found."}}


FORBIDDEN = (403, {"error": {"code": "forbidden", "message": "You are not authorized"}})


@pytest.fixture()
def vercel_api() -> Any:
    """A local stand-in for api.vercel.com; tests set `routes` (path -> status, body)."""
    import http.server
    import threading

    routes: dict[str, tuple[int, dict[str, Any]]] = {}
    seen: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            seen.append(self.headers.get("Authorization", ""))
            status, body = routes.get(path, (404, {"error": {"code": "not_found"}}))
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_port}", "routes": routes, "seen": seen}
    server.shutdown()


def _token_check(api: dict[str, Any], tmp_path: Path, token: str = TOKEN) -> tuple[int, str, str]:
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "VERCEL_API_URL": api["url"],
        "VERCEL_TOKEN": token,
        "VERCEL_TEAM_ID": TEAM,
        "DALAL_API_PROJECT": "dalal-api",
        "DALAL_WEB_PROJECT": "dalal-web",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "vercel_token_check.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert TOKEN not in out  # never logged
    return proc.returncode, out, summary.read_text() if summary.exists() else ""


def _routes(user: Any, team: Any, api_project: Any, web_project: Any) -> dict[str, Any]:
    return {
        "/v2/user": user,
        f"/v2/teams/{TEAM}": team,
        "/v9/projects/dalal-api": api_project,
        "/v9/projects/dalal-web": web_project,
    }


def test_token_check_passes_a_full_account_token(vercel_api: Any, tmp_path: Path) -> None:
    vercel_api["routes"].update(_routes(OK, OK, OK, OK))
    code, out, summary = _token_check(vercel_api, tmp_path)
    assert code == 0, out
    assert "VERCEL_TOKEN: ok" in out
    assert summary == ""
    assert vercel_api["seen"] and all(h == f"Bearer {TOKEN}" for h in vercel_api["seen"])


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        # What a project-scoped token got on the real API (run 41171365713).
        (
            _routes(_not_found("User"), FORBIDDEN, _not_found("Project"), OK),
            "limited to the project dalal-web",
        ),
        (_routes(_not_found("User"), OK, OK, OK), "limited to a team"),
        (
            _routes(
                (403, {"error": {"code": "forbidden", "invalidToken": True}}),
                FORBIDDEN,
                FORBIDDEN,
                FORBIDDEN,
            ),
            "invalid, expired or revoked",
        ),
        (
            _routes(OK, FORBIDDEN, _not_found("Project"), _not_found("Project")),
            "can't reach the Vercel team",
        ),
    ],
)
def test_token_check_stops_with_the_fix(
    vercel_api: Any, tmp_path: Path, routes: dict[str, Any], expected: str
) -> None:
    vercel_api["routes"].update(routes)
    code, out, summary = _token_check(vercel_api, tmp_path)
    assert code == 1, out
    assert expected in out
    assert "::error::" in out
    assert expected in summary
    if "can't reach the Vercel team" not in expected:
        assert "Scope: Full Account" in summary


def test_token_check_rejects_whitespace_without_calling_vercel(
    vercel_api: Any, tmp_path: Path
) -> None:
    code, out, _ = _token_check(vercel_api, tmp_path, token=TOKEN + "\n")
    assert code == 1
    assert "spaces or line breaks" in out
    assert vercel_api["seen"] == []

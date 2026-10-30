#!/usr/bin/env python3
"""Fake Vercel CLI for tests of .github/scripts/*.sh. Records calls; state in $FAKE_STATE.

Mirrors the commands, flags and output shapes of Vercel CLI 64.5.0 that the scripts
use, including what it does with --non-interactive (deploy prints a JSON payload) and
how `env pull` writes Sensitive variables (the "[SENSITIVE]" placeholder).

Failure switches: FAKE_NEON_FAIL=1, FAKE_DEPLOY_FAIL=<project>, FAKE_DOMAINS_FAIL=1,
FAKE_SENSITIVE=KEY1,KEY2 (stored as Sensitive: unreadable in production pulls).
"""

import json
import os
import pathlib
import sys

state = pathlib.Path(os.environ["FAKE_STATE"])
state.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
with open(state / "calls.log", "a") as f:
    f.write(" ".join(args) + "\n")

# Drop global flags.
clean: list[str] = []
skip = False
for a in args:
    if skip:
        skip = False
        continue
    if a in ("--token", "--scope"):
        skip = True
        continue
    if a == "--non-interactive":
        continue
    clean.append(a)


def opt(name: str, default: str | None = None) -> str | None:
    return clean[clean.index(name) + 1] if name in clean else default


def envdir(project: str) -> pathlib.Path:
    return state / "env" / project


def project_file(name: str) -> pathlib.Path:
    return state / "projects" / name


def done(code: int = 0) -> None:
    sys.exit(code)


DB_KEYS = {"DATABASE_URL", "DATABASE_URL_UNPOOLED", "PGPASSWORD"}
cmd = clean[:2]

if cmd == ["project", "inspect"]:
    done(0 if project_file(clean[2]).exists() else 1)

if cmd == ["project", "create"]:
    project_file(clean[2]).parent.mkdir(exist_ok=True)
    project_file(clean[2]).touch()
    done()

if cmd == ["project", "update"]:
    if not project_file(clean[2]).exists():
        done(1)
    settings = {"root": opt("--root-directory"), "framework": opt("--framework")}
    project_file(clean[2]).write_text(json.dumps(settings))
    done()

if cmd == ["env", "pull"]:
    out = pathlib.Path(clean[2])
    d = envdir(opt("--project") or "")
    environment = opt("--environment", "development")
    sensitive = set(filter(None, os.environ.get("FAKE_SENSITIVE", "").split(",")))
    lines = []
    for item in sorted(d.glob("*")) if d.exists() else []:
        if environment == "development" and item.name not in DB_KEYS:
            continue  # only the database integration's variables exist in development
        value = item.read_text()
        if environment != "development" and item.name in sensitive:
            value = "[SENSITIVE]"
        lines.append(f'{item.name}="{value}"')
    out.write_text("# Created by Vercel CLI\n" + "\n".join(lines) + "\n")
    done()

if cmd == ["env", "add"]:
    name, targets = clean[2], clean[3]
    project, value = opt("--project") or "", opt("--value") or ""
    if targets != "production,preview":
        print(f"Error: unexpected targets {targets}", file=sys.stderr)
        done(2)
    d = envdir(project)
    d.mkdir(parents=True, exist_ok=True)
    if (d / name).exists() and "--force" not in clean:
        print(f"Error: {name} already exists", file=sys.stderr)
        done(1)
    (d / name).write_text(value)
    done()

if cmd == ["integration", "accept-terms"]:
    done()

if cmd == ["integration", "add"]:
    if os.environ.get("FAKE_NEON_FAIL"):
        print("Error: plan required", file=sys.stderr)
        done(1)
    (state / "resources").mkdir(exist_ok=True)
    (state / "resources" / (opt("--name") or "")).touch()
    done()

if cmd == ["integration", "resource"] and clean[2] == "connect":
    resource, project = clean[3], clean[4]
    if not (state / "resources" / resource).exists():
        done(1)
    d = envdir(project)
    d.mkdir(parents=True, exist_ok=True)
    url = os.environ["FAKE_DB_URL"]
    (d / "DATABASE_URL").write_text(url)
    (d / "DATABASE_URL_UNPOOLED").write_text(url)
    (d / "PGPASSWORD").write_text("devpass")
    done()

if clean and clean[0] == "deploy":
    project = opt("--project") or ""
    root = json.loads(project_file(project).read_text() or "{}").get("root")
    target = pathlib.Path(clean[1]) / (root or "")
    if not target.is_dir():
        print(f"Error: {target} does not exist", file=sys.stderr)
        done(1)
    if os.environ.get("FAKE_DEPLOY_FAIL") == project:
        error = {
            "status": "error",
            "reason": "build_failed",
            "message": f"Build failed for {project}: pip resolution error",
        }
        print(json.dumps(error, indent=2))
        done(1)
    n = len(list(state.glob("deploy-*")))
    (state / f"deploy-{n}").write_text(" ".join(clean))
    url = f"https://{project}-{n}abc-me.vercel.app"
    payload = {
        "status": "ok",
        "deployment": {
            "id": f"dpl_{n}",
            "url": url,
            "productionUrl": f"https://{project}-me.vercel.app",
            "readyState": "READY",
            "target": "production",
        },
        "message": f"Deployed {url}",
        "next": [{"command": f"vercel inspect {url}", "when": "Inspect deployment"}],
    }
    print(json.dumps(payload, indent=2))
    done()

if clean and clean[0] == "api":
    if "-X" in clean:
        done()  # PATCH project settings
    if os.environ.get("FAKE_DOMAINS_FAIL"):
        print("Error: request failed", file=sys.stderr)
        done(1)
    p = clean[1].split("/")[3]
    domains = [
        {"name": f"{p}-git-main-me.vercel.app", "gitBranch": "main", "verified": True},
        {"name": f"{p}-me.vercel.app", "verified": True},
    ]
    print(json.dumps({"domains": domains}))
    done()

print("fake vercel: unhandled " + " ".join(clean), file=sys.stderr)
done(2)

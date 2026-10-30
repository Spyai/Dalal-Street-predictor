#!/usr/bin/env python3
"""Fake Vercel CLI for tests of .github/scripts/*.sh. Records calls; state in $FAKE_STATE.

Mirrors the commands and flags of Vercel CLI 64.5.0 that the scripts use.
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
# strip global flags
clean, skip = [], False
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


def opt(name, default=None):
    return clean[clean.index(name) + 1] if name in clean else default


def envdir(project: str) -> pathlib.Path:
    return state / "env" / project


cmd = clean[:2]
if cmd == ["project", "inspect"]:
    sys.exit(0 if (state / "projects" / clean[2]).exists() else 1)
if cmd == ["project", "create"]:
    (state / "projects").mkdir(exist_ok=True)
    (state / "projects" / clean[2]).touch()
    sys.exit(0)
if cmd == ["project", "update"]:
    project = clean[2]
    if not (state / "projects" / project).exists():
        sys.exit(1)
    settings = {"root": opt("--root-directory"), "framework": opt("--framework")}
    (state / "projects" / project).write_text(json.dumps(settings))
    sys.exit(0)
if cmd == ["env", "pull"]:
    out = pathlib.Path(clean[2])
    d = envdir(opt("--project"))
    lines = [f'{p.name}="{p.read_text()}"' for p in sorted(d.glob("*"))] if d.exists() else []
    out.write_text("# Created by Vercel CLI\n" + "\n".join(lines) + "\n")
    sys.exit(0)
if cmd == ["env", "add"]:
    name, targets, project, value = clean[2], clean[3], opt("--project"), opt("--value")
    if targets != "production,preview":
        print(f"Error: unexpected targets {targets}", file=sys.stderr)
        sys.exit(2)
    d = envdir(project)
    d.mkdir(parents=True, exist_ok=True)
    if (d / name).exists() and "--force" not in clean:
        print(f"Error: {name} already exists", file=sys.stderr)
        sys.exit(1)
    (d / name).write_text(value)
    sys.exit(0)
if cmd == ["integration", "accept-terms"]:
    sys.exit(0)
if cmd == ["integration", "add"]:
    if os.environ.get("FAKE_NEON_FAIL"):
        print("Error: plan required", file=sys.stderr)
        sys.exit(1)
    (state / "resources").mkdir(exist_ok=True)
    (state / "resources" / opt("--name")).touch()
    sys.exit(0)
if cmd == ["integration", "resource"] and clean[2] == "connect":
    res, project = clean[3], clean[4]
    if not (state / "resources" / res).exists():
        sys.exit(1)
    d = envdir(project)
    d.mkdir(parents=True, exist_ok=True)
    url = os.environ["FAKE_DB_URL"]
    (d / "DATABASE_URL").write_text(url)
    (d / "DATABASE_URL_UNPOOLED").write_text(url)
    (d / "PGPASSWORD").write_text("devpass")
    sys.exit(0)
if clean and clean[0] == "deploy":
    project = opt("--project")
    root = json.loads((state / "projects" / project).read_text() or "{}").get("root")
    target = pathlib.Path(clean[1]) / (root or "")
    if not target.is_dir():
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(1)
    n = len(list(state.glob("deploy-*")))
    (state / f"deploy-{project}-{n}").write_text(" ".join(clean))
    print(f"Inspect: https://vercel.com/x/{project}/{n}", file=sys.stderr)
    print(f"https://{project}-{n}abc-me.vercel.app")
    sys.exit(0)
if clean and clean[0] == "api" and "-X" in clean:
    sys.exit(0)  # PATCH project settings
if clean and clean[0] == "api":
    p = clean[1].split("/")[3]
    print(
        json.dumps(
            {
                "domains": [
                    {"name": f"{p}-git-main-me.vercel.app", "gitBranch": "main", "verified": True},
                    {"name": f"{p}-me.vercel.app", "verified": True},
                ]
            }
        )
    )
    sys.exit(0)
print("fake vercel: unhandled " + " ".join(clean), file=sys.stderr)
sys.exit(2)

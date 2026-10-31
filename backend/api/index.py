"""Vercel entrypoint: serves the FastAPI app as a Python function (see vercel.json)."""

from __future__ import annotations

import sys
from pathlib import Path

# The function lives in api/; the application package is one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

__all__ = ["app"]

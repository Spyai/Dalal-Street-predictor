"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from app.api import auth, health, kite, scorecard, stocks, today
from app.config import get_settings
from app.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Dalal",
    description="Private NSE/BSE research desk. Personal research tool. Not investment advice.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

for module in (auth, today, scorecard, stocks, kite, health):
    app.include_router(module.router)

"""A polite HTTP client for public data sources (NSE/BSE archives, RSS).

Rules it enforces:
- honest, fixed User-Agent (configurable); no identity rotation
- robots.txt checked per host before the first request, and respected
- <= 1 request/second per host (shared across processes through Redis)
- on 401/403/429 or a bot-protection page: stop, put the host on cooldown, raise
  SourceUnavailable. No retries around blocks.
"""

from __future__ import annotations

import contextlib
import threading
import time
import urllib.robotparser
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from app.providers.base import SourceUnavailable

BLOCK_STATUSES = {401, 403, 429}
BLOCK_COOLDOWN_SECONDS = 6 * 3600


class PoliteClient:
    def __init__(
        self,
        source: str,
        user_agent: str,
        min_interval_seconds: float = 1.0,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
        slot_waiter: Callable[[str, float], None] | None = None,
        cooldown_store: CooldownStore | None = None,
    ) -> None:
        self.source = source
        self.user_agent = user_agent
        self.min_interval = min_interval_seconds
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "*/*"},
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()
        self._slot_waiter = slot_waiter
        self._cooldown = cooldown_store or CooldownStore()

    # ------------------------------------------------------------------ public

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, expect_html: bool = False
    ) -> httpx.Response:
        host = urlsplit(url).netloc
        if self._cooldown.active(host):
            raise SourceUnavailable(
                self.source,
                f"{host} blocked an earlier request; on cooldown, not retrying",
            )
        if not self._allowed(url):
            raise SourceUnavailable(self.source, f"robots.txt disallows {urlsplit(url).path}")
        self._throttle(host)
        try:
            resp = self._client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(
                self.source, f"request failed: {exc.__class__.__name__}"
            ) from exc
        if resp.status_code in BLOCK_STATUSES or (not expect_html and _looks_like_bot_wall(resp)):
            self._cooldown.set(host, BLOCK_COOLDOWN_SECONDS)
            raise SourceUnavailable(
                self.source, f"{host} refused the request (HTTP {resp.status_code}); stopped"
            )
        if resp.status_code >= 400:
            raise SourceUnavailable(self.source, f"HTTP {resp.status_code} for {url}")
        return resp

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------------- helpers

    def _throttle(self, host: str) -> None:
        if self._slot_waiter is not None:
            self._slot_waiter(f"host:{host}", self.min_interval)
        with self._lock:
            last = self._last_request.get(host, 0.0)
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            self._robots[base] = self._load_robots(base, parts.netloc)
        return self._robots[base].can_fetch(self.user_agent, url)

    def _load_robots(self, base: str, host: str) -> urllib.robotparser.RobotFileParser:
        try:
            self._throttle(host)
            r = self._client.get(f"{base}/robots.txt")
        except httpx.HTTPError:
            return _ALLOW_ALL  # unreachable robots.txt (network) => treat as no robots.txt
        if r.status_code in BLOCK_STATUSES:
            return _DENY_ALL  # RFC 9309: robots.txt refused => assume full disallow
        if r.status_code >= 400:
            return _ALLOW_ALL  # no robots.txt => everything allowed
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(r.text.splitlines())
        return parser


class _Fixed(urllib.robotparser.RobotFileParser):
    def __init__(self, allow: bool) -> None:
        super().__init__()
        self._allow = allow

    def can_fetch(self, useragent: str, url: str) -> bool:  # noqa: ARG002
        return self._allow


_ALLOW_ALL = _Fixed(True)
_DENY_ALL = _Fixed(False)


class CooldownStore:
    """In-process cooldown registry; the Redis-backed variant shares it across processes."""

    def __init__(self) -> None:
        self._until: dict[str, float] = {}

    def active(self, host: str) -> bool:
        return self._until.get(host, 0.0) > time.time()

    def set(self, host: str, seconds: float) -> None:
        self._until[host] = time.time() + seconds


class RedisCooldownStore(CooldownStore):
    def __init__(self, r: object) -> None:
        super().__init__()
        self._r = r

    def active(self, host: str) -> bool:
        try:
            return bool(self._r.exists(f"cooldown:{host}"))  # type: ignore[attr-defined]
        except Exception:
            return super().active(host)

    def set(self, host: str, seconds: float) -> None:
        super().set(host, seconds)
        with contextlib.suppress(Exception):
            self._r.set(f"cooldown:{host}", "1", ex=int(seconds))  # type: ignore[attr-defined]


def _looks_like_bot_wall(resp: httpx.Response) -> bool:
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype:
        return False
    head = resp.text[:2000].lower()
    return any(marker in head for marker in ("captcha", "access denied", "are you a robot"))

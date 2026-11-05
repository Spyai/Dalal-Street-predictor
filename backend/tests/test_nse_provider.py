"""NSE adapter: parsers + polite client behaviour (robots.txt, rate limit, block handling)."""

from __future__ import annotations

import json
import time
from datetime import date

import httpx
import pytest

from app.providers.base import SourceUnavailable
from app.providers.http import PoliteClient
from app.providers.nse import (
    NSEProvider,
    parse_bhavcopy,
    parse_constituents,
    parse_fii_dii,
    parse_holidays,
)
from tests.conftest import FIXTURES


def _csv_rows(name: str) -> list[dict[str, str]]:
    import csv
    import io

    text = (FIXTURES / name).read_text()
    return [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]


def test_parse_fii_dii() -> None:
    rows = parse_fii_dii(json.loads((FIXTURES / "nse_fiidii.json").read_text()))
    by_cat = {r.category: r for r in rows}
    assert by_cat["FII"].net_value_cr == pytest.approx(-2054.52)
    assert by_cat["DII"].buy_value_cr == pytest.approx(14523.18)
    assert by_cat["FII"].trade_date == date(2026, 10, 29)
    assert all(r.provisional for r in rows)


def test_parse_fii_dii_rejects_garbage() -> None:
    with pytest.raises(SourceUnavailable):
        parse_fii_dii({"error": "blocked"})
    with pytest.raises(SourceUnavailable):
        parse_fii_dii([{"category": "OTHER", "date": "29-Oct-2026"}])


def test_parse_holidays_uses_equity_segment_only() -> None:
    rows = parse_holidays(json.loads((FIXTURES / "nse_holidays.json").read_text()))
    assert [r.holiday_date for r in rows] == [
        date(2026, 1, 26),
        date(2026, 3, 3),
        date(2026, 11, 10),
    ]
    assert rows[0].description == "Republic Day"
    with pytest.raises(SourceUnavailable):
        parse_holidays({"FO": []})


def test_parse_constituents() -> None:
    rows = parse_constituents(_csv_rows("nse_nifty50list.csv"))
    assert [r.symbol for r in rows] == ["RELIANCE", "HDFCBANK", "INFY"]
    assert rows[0].isin == "INE002A01018"
    assert rows[1].industry == "Financial Services"


def test_parse_bhavcopy_handles_spaces_and_dashes() -> None:
    rows = parse_bhavcopy(_csv_rows("nse_bhavcopy_full.csv"))
    rel = next(r for r in rows if r.symbol == "RELIANCE")
    assert rel.close == pytest.approx(1411.25)
    assert rel.delivery_pct == pytest.approx(50.0)
    assert rel.volume == pytest.approx(8123456)
    n1 = next(r for r in rows if r.series == "N1")
    assert n1.delivery_pct is None and n1.delivery_qty is None


# ----------------------------------------------------------------- polite client


class Recorder:
    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, float, dict[str, str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((str(request.url), time.monotonic(), dict(request.headers)))
        for prefix, resp in self.routes.items():
            if str(request.url).startswith(prefix):
                return resp
        return httpx.Response(404)


def _client(rec: Recorder, interval: float = 0.0) -> PoliteClient:
    return PoliteClient(
        source="nse",
        user_agent="Dalal-test/1.0",
        min_interval_seconds=interval,
        transport=httpx.MockTransport(rec),
    )


ROBOTS = httpx.Response(200, text=(FIXTURES / "robots.txt").read_text())


def test_provider_fetches_fii_dii_with_session_warmup() -> None:
    rec = Recorder(
        {
            "https://www.nseindia.com/robots.txt": ROBOTS,
            "https://www.nseindia.com/api/fiidiiTradeReact": httpx.Response(
                200, json=json.loads((FIXTURES / "nse_fiidii.json").read_text())
            ),
            "https://www.nseindia.com/": httpx.Response(
                200, text="<html>home</html>", headers={"content-type": "text/html"}
            ),
        }
    )
    rows = NSEProvider(_client(rec)).fii_dii()
    assert len(rows) == 2
    urls = [c[0] for c in rec.calls]
    assert urls[0].endswith("/robots.txt")
    assert urls.index("https://www.nseindia.com/") < urls.index(
        "https://www.nseindia.com/api/fiidiiTradeReact"
    )
    # Honest, fixed user agent on every request.
    assert all(c[2]["user-agent"] == "Dalal-test/1.0" for c in rec.calls)


def test_robots_disallow_is_respected() -> None:
    rec = Recorder({"https://www.nseindia.com/robots.txt": ROBOTS})
    client = _client(rec)
    with pytest.raises(SourceUnavailable, match="robots.txt"):
        client.get("https://www.nseindia.com/private/data.json")
    assert all("/private/" not in c[0] for c in rec.calls)


def test_robots_refused_means_disallow_all() -> None:
    rec = Recorder({"https://www.nseindia.com/robots.txt": httpx.Response(403)})
    with pytest.raises(SourceUnavailable, match="robots.txt"):
        _client(rec).get("https://www.nseindia.com/api/anything")


def test_block_stops_and_cools_down() -> None:
    rec = Recorder(
        {
            "https://nsearchives.nseindia.com/robots.txt": httpx.Response(404),
            "https://nsearchives.nseindia.com/": httpx.Response(403),
        }
    )
    client = _client(rec)
    with pytest.raises(SourceUnavailable, match="refused"):
        client.get("https://nsearchives.nseindia.com/content/a.csv")
    n_calls = len(rec.calls)
    with pytest.raises(SourceUnavailable, match="cooldown"):
        client.get("https://nsearchives.nseindia.com/content/b.csv")
    assert len(rec.calls) == n_calls, "no request may be sent while on cooldown"


def test_bot_wall_html_counts_as_block() -> None:
    rec = Recorder(
        {
            "https://nsearchives.nseindia.com/robots.txt": httpx.Response(404),
            "https://nsearchives.nseindia.com/": httpx.Response(
                200,
                text="<html>Please complete the CAPTCHA</html>",
                headers={"content-type": "text/html"},
            ),
        }
    )
    with pytest.raises(SourceUnavailable, match="refused"):
        _client(rec).get("https://nsearchives.nseindia.com/content/a.csv")


def test_rate_limit_spacing() -> None:
    rec = Recorder(
        {
            "https://nsearchives.nseindia.com/robots.txt": httpx.Response(404),
            "https://nsearchives.nseindia.com/": httpx.Response(200, text="a,b\n1,2\n"),
        }
    )
    client = _client(rec, interval=0.2)
    client.get("https://nsearchives.nseindia.com/x.csv")
    client.get("https://nsearchives.nseindia.com/y.csv")
    times = [c[1] for c in rec.calls]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(g >= 0.19 for g in gaps), gaps


def test_bhavcopy_url_and_parse() -> None:
    url = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_29102026.csv"
    body = (FIXTURES / "nse_bhavcopy_full.csv").read_text()
    rec = Recorder(
        {
            "https://nsearchives.nseindia.com/robots.txt": httpx.Response(404),
            url: httpx.Response(200, text=body),
        }
    )
    rows = NSEProvider(_client(rec)).bhavcopy(date(2026, 10, 29))
    assert {r.symbol for r in rows} >= {"RELIANCE", "INFY", "NEWCO"}

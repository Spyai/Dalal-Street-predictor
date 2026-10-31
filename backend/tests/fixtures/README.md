# Test fixtures

**These payloads are synthetic.** They were hand-built to match each source's documented
response shape (NSE JSON/CSV formats, yfinance DataFrame layouts, Kite Connect responses),
because the build environment could not reach NSE, Yahoo or Kite. The numbers are made up
and exist only to exercise parsers. They are never loaded into a real database.

To replace them with real recorded payloads, run this on a machine with network access:

    docker compose run --rm backend python -m app.cli record-fixtures /data/fixtures
    cp -r data/fixtures backend/tests/fixtures/recorded

`test_recorded_fixtures.py` runs the same parsers against `fixtures/recorded/` when it
exists, and is skipped otherwise.

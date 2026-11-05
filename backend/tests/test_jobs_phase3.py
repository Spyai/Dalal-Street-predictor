"""Phase 3 jobs end to end: the 07:30 run generates and records stock ideas after the
market call and lists them in the brief, the 09:20 run fills or invalidates them at the
open and opens the paper positions, the 15:45 run grades them, closes the paper trades
and snapshots the paper equity; plus the CLI commands, the Telegram texts, the workflow
allowlist (evaluated through real bash) and the Vercel env keys.

No network: the market-data chain and NSE are faked and Telegram is captured. The world is
the Phase 2 synthetic year (indices, VIX, global cues, a verified 2026 calendar) plus a
small NSE universe engineered so the breakout and RSI rules fire on the signal session;
the ideas models are real LightGBM champions trained walk-forward on the synthetic
candidate frame of tests/test_ideas_models.py. The ranking bar is lowered through
Settings where a test needs ideas on the ledger; the production defaults are untouched.
"""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app import cli
from app.config import IST, Settings
from app.jobs import runner, tasks
from app.jobs.definitions import get_spec, job_specs
from app.jobs.runner import run_job
from app.ml import ideas_models, strategies
from app.ml.core import TrainResult, horizons
from app.models import (
    DailyBar,
    Idea,
    IdeaOutcome,
    ModelVersion,
    PaperEquity,
    PaperTrade,
)
from app.providers.base import Bar
from app.services import brief, idea_grading, instruments, ledger, paper
from app.services.ideas_scorecard import IdeaTrack
from app.services.market_data import upsert_daily_bars
from tests.test_idea_grading import bars as grading_bars
from tests.test_idea_ledger import AS_OF, SESSION, make_idea, make_instrument
from tests.test_ideas_models import (
    LOOSE_PARAMS,
    synthetic_candidates,
    universe_frame,
)
from tests.test_ideas_models import (
    settings as training_settings,
)
from tests.test_jobs_phase2 import (
    FIRST,
    FRIDAY,
    FRIDAY_0730,
    FRIDAY_0920,
    FRIDAY_1545,
    LAST,
    TARGET,
    BarProvider,
    ExchangeStub,
    _friday_bar,
    _last_run,
    _weekdays,
    champion,  # noqa: F401  (pytest fixture: the market-call champion)
    world,  # noqa: F401  (pytest fixture: the Phase 2 synthetic market)
)

REPO = Path(__file__).resolve().parents[2]
HOLIDAY = date(2026, 1, 26)  # the one holiday of the Phase 2 world
PLAIN = [f"PLAIN{i}" for i in range(6)]
BREAKOUT, OVERSOLD = "BRKOUT", "OVRSLD"
ALLOWLIST = (
    "^(migrate|sync-instruments|backfill|test-telegram|train|train-ideas( --horizon [a-z]+)?|"
    "predict( --force)?|ideas( --force)?|verify-ledger|run-job [a-z_]+( --force)?)$"
)
# The market model's small walk-forward window (as in Phase 2) and a ranking bar every
# priced candidate clears, so the ideas reach the ledger: the tests are about the loop,
# not about whether a model trained on synthetic noise beats 55 % / 0.5 %.
SETTINGS = Settings(
    prediction_targets=TARGET,
    wf_min_train_days=150,
    wf_step_days=60,
    ideas_min_p_target=0.0,
    ideas_min_ev_pct=-100.0,
)
STRICT = SETTINGS.model_copy(update={"ideas_min_p_target": 1.0})  # nothing can clear p >= 1


# ------------------------------------------------------------------ the universe


def _trend_bars(
    rng: np.random.Generator, days: list[date], start: float, drift: float, vol: float
) -> list[Bar]:
    close = start * np.cumprod(1 + drift + rng.normal(0.0, vol, len(days)))
    out: list[Bar] = []
    for i, (d, c) in enumerate(zip(days, close, strict=True)):
        prev = close[i - 1] if i else c
        o = prev * (1 + rng.normal(0, 0.002))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
        out.append(Bar(d, float(o), float(hi), float(lo), float(c), float(rng.uniform(2e6, 3e6))))
    return out


def _breakout_last_bar(bars: list[Bar]) -> Bar:
    """Rewrite the last bar as a 20-day breakout on six times the usual volume."""
    prior_high = max(b.high for b in bars[-21:-1])
    prev_close = bars[-2].close
    close = prior_high * 1.02
    open_ = prev_close * 1.003
    return Bar(bars[-1].trade_date, open_, close * 1.004, open_ * 0.997, close, 1.5e7)


def _oversold_tail(bars: list[Bar]) -> list[Bar]:
    """Three consecutive 3 % down closes at the end of a long uptrend: RSI(14) collapses
    while the close stays well above the 200-DMA."""
    out = list(bars[:-3])
    c = out[-1].close
    for b in bars[-3:]:
        o = c * 0.995
        c = c * 0.97
        out.append(Bar(b.trade_date, o, o * 1.002, c * 0.995, c, b.volume))
    return out


@pytest.fixture()
def stocks(db: Session, world: dict[str, Bar]) -> dict[str, list[Bar]]:  # noqa: F811
    """Eight in_universe NSE stocks with a year of bars (turnover far above 10 cr): six
    random walks, one engineered 20-day breakout and one oversold uptrend on the signal
    session LAST."""
    rng = np.random.default_rng(11)
    days = [d for d in _weekdays(FIRST, LAST) if d != HOLIDAY]
    out: dict[str, list[Bar]] = {}
    for i, sym in enumerate(PLAIN):
        out[sym] = _trend_bars(rng, days, 200.0 + 300.0 * i, 0.0, 0.012)
    out[BREAKOUT] = _trend_bars(rng, days, 500.0, 0.0012, 0.003)
    out[BREAKOUT][-1] = _breakout_last_bar(out[BREAKOUT])
    out[OVERSOLD] = _oversold_tail(_trend_bars(rng, days, 800.0, 0.0015, 0.002))
    for sym, bars in out.items():
        inst = instruments.get_or_create_equity(db, sym, "NSE")
        inst.in_universe = True
        upsert_daily_bars(db, inst.id, bars, "yfinance")
    db.commit()
    return out


_TRAINED: dict[str, TrainResult] = {}


def _trained(horizon: str) -> TrainResult:
    """Walk-forward trained ideas model on the synthetic candidate frame (cached: the
    training is deterministic and takes a few seconds)."""
    if horizon not in _TRAINED:
        _TRAINED[horizon] = ideas_models.train_walk_forward(
            synthetic_candidates(),
            horizon,
            params=LOOSE_PARAMS,
            settings=training_settings(),
            universe=universe_frame() if horizon == "swing" else None,
        )
    return _TRAINED[horizon]


@pytest.fixture()
def ideas_champions(db: Session) -> dict[str, ModelVersion]:
    out = {
        h: ideas_models.save_ideas_model(
            db, _trained(h), promote=True, note="test", settings=SETTINGS
        )
        for h in horizons(SETTINGS)
    }
    db.commit()
    return out


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sent: list[str] = []
    provider = BarProvider()
    nse = ExchangeStub()
    monkeypatch.setattr(tasks.registry, "market_data_chain", lambda db, **kw: [provider])
    monkeypatch.setattr(tasks.registry, "nse_provider", lambda: nse)
    monkeypatch.setattr(runner.telegram, "send", lambda text, **kw: sent.append(text) or True)
    return {"sent": sent, "provider": provider, "nse": nse}


def _run(name: str, now: datetime, settings: Settings = SETTINGS, **kw: Any) -> str:
    return run_job(get_spec(name), now=now, sleep=lambda s: None, settings=settings, **kw)


def _ideas(db: Session) -> list[Idea]:
    db.expire_all()
    return list(db.scalars(select(Idea).order_by(Idea.seq)).all())


def _outcome(db: Session, idea: Idea) -> IdeaOutcome | None:
    db.expire_all()
    return idea_grading.outcome_of(db, idea.id)


def _trades(db: Session) -> list[PaperTrade]:
    db.expire_all()
    return list(db.scalars(select(PaperTrade).order_by(PaperTrade.id)).all())


def _by(ideas: list[Idea], symbol: str, horizon: str) -> Idea:
    (found,) = [i for i in ideas if i.symbol == symbol and i.horizon == horizon]
    return found


def _stock_bar(
    close_d: float, *, open_pct: float, high_pct: float, low_pct: float, close_pct: float
) -> Bar:
    return Bar(
        FRIDAY,
        close_d * (1 + open_pct / 100),
        close_d * (1 + high_pct / 100),
        close_d * (1 + low_pct / 100),
        close_d * (1 + close_pct / 100),
        2.5e6,
    )


# ------------------------------------------------------------------ 07:30 ideas


def test_premarket_records_ideas_and_the_brief_lists_them(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    ideas = _ideas(db)
    assert {i.symbol for i in ideas} >= {BREAKOUT, OVERSOLD}
    assert {i.horizon for i in ideas} == {"swing", "positional"}
    assert len({(i.symbol, i.horizon) for i in ideas}) == len(ideas), "one idea per symbol/horizon"
    brk = _by(ideas, BREAKOUT, "swing")
    assert (brk.session_date, brk.exchange, brk.direction) == (FRIDAY, "NSE", "long")
    assert brk.setup in {"breakout", "momentum"} and brk.strategy in strategies.STRATEGY_NAMES
    assert brk.model_version_id == ideas_champions["swing"].id
    assert _by(ideas, BREAKOUT, "positional").model_version_id == ideas_champions["positional"].id
    assert _by(ideas, OVERSOLD, "swing").setup in {"mean_reversion", "momentum"}
    assert brk.horizon_days == 10 and _by(ideas, BREAKOUT, "positional").horizon_days == 60
    assert not brk.is_late and brk.surveillance_unknown is True
    close_d = stocks[BREAKOUT][-1].close
    assert brk.entry_low < close_d < brk.entry_high
    assert brk.stop < brk.entry_low and brk.target_1 > brk.entry_high
    assert brk.target_2 is not None and brk.target_2 > brk.target_1
    assert 0.0 <= brk.p_target <= 1.0 and brk.position_qty > 0
    assert brk.position_value == pytest.approx(brk.position_qty * close_d, rel=1e-4)
    assert brk.position_value <= SETTINGS.paper_capital * SETTINGS.paper_max_position_pct / 100
    assert brk.risk_amount <= SETTINGS.paper_capital * SETTINGS.risk_per_trade_pct / 100
    assert 3 <= len(brk.reasons) <= 5 and brk.invalidation
    assert brk.drivers and all(d.get("text") for d in brk.drivers)
    # The snapshot is the model's input: every trained feature plus the strategy one-hots.
    trained = set(ideas_champions["swing"].feature_names)
    assert trained <= set(brk.feature_snapshot)
    assert brk.feature_snapshot[f"strategy_{brk.strategy}"] == 1.0
    ranks = sorted(i.rank for i in ideas if i.horizon == "swing")
    assert ranks == list(range(1, len(ranks) + 1))
    assert ledger.verify_ledger(db, "idea").ok and ledger.verify_ledger(db).ok
    # Universe bars were fresh through the signal session: nothing was fetched for them.
    assert not [r for r in fakes["provider"].requests if r[0] in stocks]

    run = _last_run(db, "premarket_prediction")
    d = run.details["ideas"]
    assert d["session"] == FRIDAY.isoformat() and d["signal_session"] == LAST.isoformat()
    assert d["universe"]["instruments"] == 8 and d["universe"]["with_bar"] == 8
    assert d["candidates"]["n"] >= 2 and d["candidates"]["by_strategy"]["breakout_20d"] >= 1
    assert d["candidates"]["by_strategy"]["mean_reversion_rsi"] >= 1
    for h in ("swing", "positional"):
        assert d[h]["status"] == "recorded"
        assert d[h]["model_version"] == ideas_champions[h].version
        assert all(row["status"] == "recorded" for row in d[h]["ideas"])
    assert d["ranking"]["n_cleared"] >= 4 and d["ranking"]["n_unpriced"] == 0
    assert "ideas: swing" in (run.message or "") and "gaps" not in (run.message or "")
    (msg,) = fakes["sent"]  # one brief, no partial alert
    assert TARGET in msg and "<b>Stock ideas</b>" in msg
    line = next(ln for ln in msg.splitlines() if ln.startswith(f"• <b>{BREAKOUT}</b> swing/"))
    assert f"entry {brk.entry_low:,.2f}–{brk.entry_high:,.2f}" in line
    assert f"stop {brk.stop:,.2f}, target {brk.target_1:,.2f}" in line
    assert f"P(target before stop) {brk.p_target * 100:.0f}%" in line
    assert "EV " in line and "after costs" in line and f"{brk.position_qty} shares" in line
    assert "so far: unproven (no graded ideas yet)" in line
    assert "No edge today" not in msg
    assert "ASM/GSM surveillance status: unknown" in msg
    head = max(ideas, key=lambda i: i.seq)
    assert f"Idea ledger head: seq {head.seq}, {head.hash[:12]}" in msg
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")

    # Re-run: idempotent, nothing re-sent.
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    assert len(_ideas(db)) == len(ideas)
    again = _last_run(db, "premarket_prediction")
    assert again.details["ideas"]["swing"]["status"] == "already recorded"
    assert again.details["ideas"]["positional"]["status"] == "already recorded"
    assert again.details["brief"].startswith("not sent")
    assert len(fakes["sent"]) == 1


def test_overlapping_premarket_runs_record_each_idea_once(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both runs pass the cheap pre-check; the idea ledger lock makes the second one see
    the first's rows (LedgerConflict) and report them as already recorded."""
    monkeypatch.setattr(tasks.ledger, "latest_call", lambda db, target, session: None)
    monkeypatch.setattr(tasks.ledger, "ideas_for_session", lambda db, session, **kw: [])
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    n = len(_ideas(db))
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    assert len(_ideas(db)) == n
    run = _last_run(db, "premarket_prediction")
    rows = run.details["ideas"]["swing"]["ideas"]
    assert rows and all(r["status"] == "already recorded" for r in rows)
    assert len(fakes["sent"]) == 1 and ledger.verify_ledger(db, "idea").ok


def test_premarket_without_an_ideas_model_is_partial_with_the_train_ideas_hint(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "partial"
    assert _ideas(db) == []
    run = _last_run(db, "premarket_prediction")
    assert run.details[TARGET]["status"] == "recorded", "the market call is unaffected"
    d = run.details["ideas"]
    assert d["swing"]["status"] == tasks.TRAIN_IDEAS_HINT
    assert d["positional"]["status"] == tasks.TRAIN_IDEAS_HINT
    assert "universe" not in d, "no features are computed without a model"
    assert "run `python -m app.cli train-ideas`" in (run.message or "")
    briefs = [m for m in fakes["sent"] if "Not investment advice" in m and "gaps" not in m]
    (msg,) = briefs
    assert "<b>Stock ideas</b>" in msg and msg.count("no ideas model") == 2
    assert "No edge today" not in msg, "no model is not the same as no edge"
    (alert,) = [m for m in fakes["sent"] if "finished with gaps" in m]
    assert "train-ideas" in alert


def test_premarket_says_no_edge_today_when_nothing_clears_the_bar(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730, settings=STRICT) == "success"
    assert _ideas(db) == []
    run = _last_run(db, "premarket_prediction")
    d = run.details["ideas"]
    for h in ("swing", "positional"):
        assert d[h]["status"].startswith(f"No edge today ({h})")
        assert "P(target) >= 100%" in d[h]["status"]
    assert d["ranking"]["n_candidates"] >= 2 and d["ranking"]["n_cleared"] == 0
    assert d["ranking"]["best_p_target"] is not None
    assert "swing No edge today" in (run.message or "")
    (msg,) = fakes["sent"]
    assert "<b>No edge today</b> (swing)" in msg and "<b>No edge today</b> (positional)" in msg
    assert "P(target) &gt;= 100%" in msg and "Idea ledger head" not in msg


def test_premarket_without_a_universe_reports_it(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "partial"
    run = _last_run(db, "premarket_prediction")
    assert run.details["ideas"]["swing"]["status"] == f"not generated: {tasks.UNIVERSE_HINT}"
    assert "sync-instruments" in (run.message or "")
    assert _ideas(db) == []


def test_premarket_refuses_stale_universe_bars(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    """Most of the universe has no bar for the signal session and the provider is down:
    no idea is generated from stale data, and the run says so."""
    ids = [instruments.get_or_create_equity(db, s, "NSE").id for s in stocks]
    db.execute(
        text("DELETE FROM daily_bar WHERE instrument_id = ANY(:ids) AND trade_date = :d"),
        {"ids": ids, "d": LAST},
    )
    db.commit()
    fakes["provider"].fail = True
    assert _run("premarket_prediction", FRIDAY_0730) == "partial"
    run = _last_run(db, "premarket_prediction")
    d = run.details["ideas"]
    assert d["universe"]["with_bar"] == 0 and d["universe"]["errors"] == 8
    assert d["swing"]["status"].startswith("not generated: only 0 of 8 universe stocks have a bar")
    assert "candidates" not in d and _ideas(db) == []


def test_ideas_wait_for_the_calls_recording_guard(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    """Before 18:30 on the reference session the previous close may be provisional: no
    call and no idea is recorded against it."""
    too_early = datetime(2026, 10, 29, 16, 0, tzinfo=IST)
    assert _run("premarket_prediction", too_early) == "partial"
    assert _ideas(db) == []
    run = _last_run(db, "premarket_prediction")
    assert "provisional" in run.details["ideas"]["swing"]["status"]
    assert run.details["brief"].startswith("not sent")


# ------------------------------------------------------------------ 09:20 fills


def test_open_check_fills_inside_the_zone_and_invalidates_a_gap(
    db: Session,
    world: dict[str, Bar],  # noqa: F811
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    ideas = _ideas(db)
    fakes["sent"].clear()
    fakes["provider"].requests.clear()

    # No bar for the session yet: nothing is invented, the run is partial.
    assert _run("open_check", FRIDAY_0920) == "partial"
    run = _last_run(db, "open_check")
    assert "not available yet" in (run.message or "")
    assert all(_outcome(db, i) is None for i in ideas) and _trades(db) == []

    brk_close = stocks[BREAKOUT][-1].close
    ovr_close = stocks[OVERSOLD][-1].close
    fakes["provider"].bars[TARGET] = [_friday_bar(world, open_pct=0.2, close_pct=0.3)]
    fakes["provider"].bars[BREAKOUT] = [
        _stock_bar(brk_close, open_pct=0.0, high_pct=0.6, low_pct=-0.3, close_pct=0.4)
    ]
    fakes["provider"].bars[OVERSOLD] = [
        _stock_bar(ovr_close, open_pct=5.0, high_pct=5.5, low_pct=4.5, close_pct=5.0)
    ]
    assert _run("open_check", FRIDAY_0920) == "success"
    run = _last_run(db, "open_check")
    assert "1 gap call(s) graded" in (run.message or "")
    d = run.details["ideas"]
    for h in ("swing", "positional"):
        brk = _by(ideas, BREAKOUT, h)
        o = _outcome(db, brk)
        assert o is not None and o.status == "open"
        assert o.entry_price == pytest.approx(brk_close) and o.entry_date == FRIDAY
        assert o.source == "yfinance"
        entry = d[f"seq {brk.seq} {BREAKOUT} {h}"]
        assert entry["status"] == "open" and entry["paper_trade_id"] is not None
        assert entry["qty"] == brk.position_qty
        ovr = _by(ideas, OVERSOLD, h)
        o2 = _outcome(db, ovr)
        assert o2 is not None and o2.status == "invalidated" and o2.entry_price is None
        assert o2.notes and "gapped above the entry zone" in o2.notes
        assert d[f"seq {ovr.seq} {OVERSOLD} {h}"]["status"] == "invalidated"
    trades = _trades(db)
    assert sorted(t.symbol for t in trades) == [BREAKOUT, BREAKOUT]
    assert all(t.status == "open" and t.entry_price == pytest.approx(brk_close) for t in trades)
    assert {t.idea_id for t in trades} == {
        _by(ideas, BREAKOUT, h).id for h in ("swing", "positional")
    }
    assert all(
        t.qty == _by(ideas, BREAKOUT, "swing").position_qty
        for t in trades
        if t.idea_id == _by(ideas, BREAKOUT, "swing").id
    )
    assert "2 idea(s) filled, 2 invalidated, 0 still pending" in (run.message or "")
    # The partial session bar is never stored as the day's bar.
    inst = instruments.get_instrument(db, BREAKOUT, "NSE")
    assert inst is not None and db.get(DailyBar, (inst.id, FRIDAY)) is None
    (note,) = [m for m in fakes["sent"] if "open check" in m]
    assert f"{BREAKOUT} (swing" in note and "<b>filled</b>" in note
    assert f"{OVERSOLD} (swing" in note and "<b>invalidated</b>" in note
    assert d["summary"] == "sent"

    # Idempotent: already decided ideas are reported as they are, no new trades.
    fakes["sent"].clear()
    assert _run("open_check", FRIDAY_0920) == "success"
    assert len(_trades(db)) == 2
    d = _last_run(db, "open_check").details["ideas"]
    assert (
        d[f"seq {_by(ideas, BREAKOUT, 'swing').seq} {BREAKOUT} swing"]["status"] == "already open"
    )
    assert d[f"seq {_by(ideas, OVERSOLD, 'swing').seq} {OVERSOLD} swing"]["status"] == (
        "already invalidated"
    )
    assert not [m for m in fakes["sent"] if "open check" in m]


def test_open_check_runs_for_ideas_even_without_a_market_call(
    db: Session, fakes: dict[str, Any]
) -> None:
    inst = make_instrument(db, "RELIANCE")
    instruments.ensure_reference_instruments(db)
    idea = ledger.record_idea(db, make_idea(inst.id, session_date=FRIDAY, as_of=FRIDAY_0730))
    db.commit()
    fakes["provider"].bars["RELIANCE"] = [Bar(FRIDAY, 1000.0, 1010.0, 995.0, 1005.0, 1e6)]
    assert _run("open_check", FRIDAY_0920) == "success"
    o = _outcome(db, idea)
    assert o is not None and o.status == "open" and o.entry_price == 1000.0
    assert len(_trades(db)) == 1
    assert "0 gap call(s) graded" in (_last_run(db, "open_check").message or "")


# ------------------------------------------------------------------ 15:45 grading


def _friday_bars_for(
    market: dict[str, Bar], stocks: dict[str, list[Bar]], ideas: list[Idea], fakes: dict[str, Any]
) -> None:
    """Friday bars: BRKOUT touches the swing target (not the positional one) and stays
    above both stops; OVRSLD gaps above its zone; the Nifty 50 bar closes +0.5 %."""
    swing, positional = _by(ideas, BREAKOUT, "swing"), _by(ideas, BREAKOUT, "positional")
    close_d = stocks[BREAKOUT][-1].close
    high = swing.target_1 * 1.002
    assert high < positional.target_1
    fakes["provider"].bars[BREAKOUT] = [
        Bar(FRIDAY, close_d, high, max(swing.stop, positional.stop) * 1.002, close_d * 1.015, 3e6)
    ]
    ovr_close = stocks[OVERSOLD][-1].close
    fakes["provider"].bars[OVERSOLD] = [
        _stock_bar(ovr_close, open_pct=5.0, high_pct=5.5, low_pct=4.5, close_pct=5.0)
    ]
    fakes["provider"].bars[TARGET] = [_friday_bar(market, open_pct=0.1, close_pct=0.5)]


def _nifty_close(db: Session, d: date) -> float:
    inst = instruments.get_instrument(db, TARGET, "INDEX")
    assert inst is not None
    bar = db.get(DailyBar, (inst.id, d))
    assert bar is not None
    return float(bar.close)


def test_close_grading_grades_ideas_closes_trades_and_snapshots_equity(
    db: Session,
    world: dict[str, Bar],  # noqa: F811
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    ideas = _ideas(db)
    brk_close = stocks[BREAKOUT][-1].close
    fakes["provider"].bars[BREAKOUT] = [
        _stock_bar(brk_close, open_pct=0.0, high_pct=0.6, low_pct=-0.3, close_pct=0.4)
    ]
    assert _run("open_check", FRIDAY_0920) == "partial"  # BRKOUT filled; the rest waits
    fakes["provider"].bars.clear()
    fakes["sent"].clear()

    # The session's bars are not published yet: nothing is graded or marked, the equity
    # snapshot is refused rather than written at a guess, and the run is partial.
    assert _run("close_grading", FRIDAY_1545) == "partial"
    run = _last_run(db, "close_grading")
    d = run.details["ideas"]
    assert d["paper_equity"].startswith("not snapshotted: no close for 2026-10-30")
    assert BREAKOUT in d["paper_equity"]
    # The filled BRKOUT ideas wait for the session's bar, the pending OVRSLD ones for
    # their entry session's bar (the same session, never filled: no 09:20 open for them).
    assert len(d["waiting"]) == len(ideas) and all("2026-10-30 yet" in w for w in d["waiting"])
    assert sum("entry session" in w for w in d["waiting"]) == 2
    assert db.scalar(select(PaperEquity)) is None
    assert "waiting for a bar" in (run.message or "") and "not snapshotted" in (run.message or "")

    _friday_bars_for(world, stocks, ideas, fakes)
    assert _run("close_grading", FRIDAY_1545) == "success"
    run = _last_run(db, "close_grading")
    d = run.details["ideas"]
    swing, positional = _by(ideas, BREAKOUT, "swing"), _by(ideas, BREAKOUT, "positional")
    o_swing, o_pos = _outcome(db, swing), _outcome(db, positional)
    assert o_swing is not None and o_swing.status == "target"
    assert o_swing.exit_date == FRIDAY and o_swing.exit_price == pytest.approx(swing.target_1)
    assert o_swing.days_held == 1 and o_swing.net_return_pct is not None
    assert o_swing.net_return_pct == pytest.approx(o_swing.gross_return_pct - o_swing.costs_pct)
    assert o_pos is not None and o_pos.status == "open" and o_pos.max_favorable_pct is not None
    for h in ("swing", "positional"):
        o = _outcome(db, _by(ideas, OVERSOLD, h))
        assert o is not None and o.status == "invalidated"
    trades = {t.idea_id: t for t in _trades(db)}
    closed, still_open = trades[swing.id], trades[positional.id]
    assert closed.status == "closed" and closed.exit_price == pytest.approx(swing.target_1)
    assert closed.net_pnl is not None and closed.gross_pnl is not None
    assert closed.gross_pnl == pytest.approx(closed.qty * (swing.target_1 - brk_close), rel=1e-6)
    assert closed.net_pnl < closed.gross_pnl  # costs charged
    assert still_open.status == "open"
    # Equity: capital + realised net P&L - cost basis of the open position + its mark.
    eq = db.scalar(select(PaperEquity).where(PaperEquity.trade_date == FRIDAY))
    assert eq is not None
    friday_close = fakes["provider"].bars[BREAKOUT][0].close
    cash = SETTINGS.paper_capital + closed.net_pnl - still_open.qty * still_open.entry_price
    assert eq.cash == pytest.approx(cash, rel=1e-9)
    assert eq.invested == pytest.approx(still_open.qty * friday_close, rel=1e-9)
    assert eq.equity == pytest.approx(eq.cash + eq.invested, rel=1e-9)
    assert eq.open_positions == 1 and eq.nifty_close == pytest.approx(_nifty_close(db, FRIDAY))
    assert d["paper_equity"]["equity"] == pytest.approx(eq.equity)
    assert d[f"seq {swing.seq} {BREAKOUT} swing"]["status"] == "target"
    assert d[f"seq {swing.seq} {BREAKOUT} swing"]["paper_trade_status"] == "closed"
    assert d[f"seq {positional.seq} {BREAKOUT} positional"]["status"] == "open"
    assert "1 idea(s) settled, 1 open, 2 invalidated" in (run.message or "")
    (msg,) = [m for m in fakes["sent"] if "close grading" in m]
    assert "<b>Ideas</b>" in msg and f"{BREAKOUT} (swing" in msg and "<b>target</b>" in msg
    assert "net after costs" in msg and f"{BREAKOUT} (positional" in msg and "open since" in msg
    assert f"Paper equity {FRIDAY.isoformat()}: ₹" in msg and "Nifty 50" in msg
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")

    # Re-run: the same outcomes, the same single equity row, nothing new to report.
    fakes["sent"].clear()
    assert _run("close_grading", FRIDAY_1545) == "success"
    assert db.scalar(select(PaperEquity.trade_date)) == FRIDAY
    assert len(_trades(db)) == 2 and _outcome(db, swing).status == "target"  # type: ignore[union-attr]


def test_close_grading_fills_a_missed_idea_from_its_entry_bar(
    db: Session,
    world: dict[str, Bar],  # noqa: F811
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    """The 09:20 run never happened: grading fills the idea from the entry session's open
    (same rule), walks the bar and opens + closes the paper trade in one go."""
    assert _run("premarket_prediction", FRIDAY_0730) == "success"
    ideas = _ideas(db)
    _friday_bars_for(world, stocks, ideas, fakes)
    assert _run("close_grading", FRIDAY_1545) == "success"
    swing = _by(ideas, BREAKOUT, "swing")
    o = _outcome(db, swing)
    assert (
        o is not None
        and o.status == "target"
        and o.entry_price == pytest.approx(stocks[BREAKOUT][-1].close)
    )
    trade = paper.trade_for_idea(db, swing)
    assert trade is not None and trade.status == "closed" and trade.net_pnl is not None
    o2 = _outcome(db, _by(ideas, OVERSOLD, "swing"))
    assert (
        o2 is not None
        and o2.status == "invalidated"
        and paper.trade_for_idea(db, o2.idea_id) is None
    )
    eq = db.scalar(select(PaperEquity))
    assert eq is not None and eq.open_positions == 1  # the positional BRKOUT idea


def test_close_grading_skips_without_calls_ideas_or_book(
    db: Session, fakes: dict[str, Any]
) -> None:
    instruments.ensure_reference_instruments(db)
    db.commit()
    assert _run("close_grading", FRIDAY_1545) == "skipped"
    assert "no ungraded market calls, open ideas or paper positions" in (
        _last_run(db, "close_grading").message or ""
    )


# ------------------------------------------------------------------ Telegram texts


def test_morning_brief_ideas_section(db: Session) -> None:
    inst = make_instrument(db, "M&M")
    idea = ledger.record_idea(db, make_idea(inst.id, symbol="M&M"))
    db.commit()
    unproven = IdeaTrack("swing", "breakout", 0, None, None, None, True)
    msg = brief.morning_brief([], SESSION, ideas=[(idea, unproven)], idea_head=ledger.head_idea(db))
    assert "No market call was recorded" in msg and "<b>Stock ideas</b>" in msg
    line = next(ln for ln in msg.splitlines() if ln.startswith("• <b>M&amp;M</b>"))
    assert "M&M" not in line
    assert "swing/breakout: entry 990.00–1,010.00, stop 980.00, target 1,040.00" in line
    assert "(+4.0% to +6.0%)" in line and "P(target before stop) 60%" in line
    assert "EV +" in line and "after costs" in line and "R:R 2.0" in line
    assert "50 shares ≈ ₹50,000" in line
    assert "breakout/swing so far: unproven (no graded ideas yet)" in line
    assert "<b>No edge today</b> (positional): no candidate clears P(target) &gt;= 55%" in msg
    assert "ASM/GSM surveillance status: unknown" in msg and "paper positions" in msg
    assert f"Idea ledger head: seq {idea.seq}, {idea.hash[:12]}" in msg
    assert "Ledger head:" not in msg.replace("Idea ledger head:", "")  # no prediction anchor
    assert msg.rstrip().endswith("<i>Personal research tool. Not investment advice.</i>")

    proven = IdeaTrack("swing", "breakout", 12, 7 / 12, 0.5, 0.8, True)
    msg = brief.morning_brief([], SESSION, ideas=[(idea, proven)])
    assert "breakout/swing so far: 58% hit target of 12 graded, +0.80% net per idea" in msg
    assert "unproven (&lt; 100 graded)" in msg
    assert f"Idea ledger head: seq {idea.seq}" in msg, "the highest-seq idea stands in"

    # The engine ran and said why a horizon has nothing; and without the engine, no section.
    notes = {"swing": tasks.TRAIN_IDEAS_HINT, "positional": tasks.TRAIN_IDEAS_HINT}
    msg = brief.morning_brief([], SESSION, idea_notes=notes)
    assert msg.count("no ideas model: run `python -m app.cli train-ideas`") == 2
    assert "No edge today" not in msg and "Idea ledger head" not in msg
    assert "Stock ideas" not in brief.morning_brief([], SESSION)


def test_fill_and_close_summaries(db: Session) -> None:
    inst = make_instrument(db)
    idea = ledger.record_idea(db, make_idea(inst.id))
    other = ledger.record_idea(db, make_idea(inst.id, horizon="positional"))
    db.commit()
    now = datetime(2026, 11, 2, 3, 50, tzinfo=UTC)
    filled = idea_grading.open_check_fill(db, idea, 1000.0, now=now, source="kite")
    gapped = idea_grading.open_check_fill(db, other, 1020.0, now=now, source="kite")
    msg = brief.fill_summary([(idea, filled), (other, gapped)], SESSION)
    assert "<b>Dalal open check — Mon 02 Nov 2026</b>" in msg
    assert (
        "RELIANCE (swing, breakout): <b>filled</b> at 1,000.00 (zone 990.00–1,010.00), "
        "50 shares paper"
    ) in msg
    assert "RELIANCE (positional, breakout): <b>invalidated</b>: open 1020.00 gapped above" in msg
    assert "No idea was waiting" in brief.fill_summary([], SESSION)

    later = datetime(2026, 11, 3, 10, 16, tzinfo=UTC)
    outcome = idea_grading.grade_idea(
        db,
        idea,
        grading_bars((0, 1000.0, 1006.0, 996.0, 1002.0), (1, 1003.0, 1045.0, 1000.0, 1030.0)),
        now=later,
        source="yfinance",
    )
    assert outcome.status == "target"
    equity = PaperEquity(
        trade_date=date(2026, 11, 3),
        equity=1_001_234.5,
        cash=951_234.5,
        invested=50_000.0,
        open_positions=1,
        nifty_close=25_432.1,
        source="test",
    )
    msg = brief.close_summary([], ideas=[(idea, outcome)], equity=equity)
    assert "<b>Ideas</b>" in msg and "Nothing was graded" not in msg
    assert (
        "RELIANCE (swing, breakout): <b>target</b> on 2026-11-03 at 1,040.00 — +4.00% gross" in msg
    )
    assert "net after costs" in msg and "R, 2 sessions" in msg
    assert (
        "Paper equity 2026-11-03: ₹1,001,234 (cash ₹951,234, invested ₹50,000, "
        "1 open position); Nifty 50 25,432.10 (price index)"
    ) in msg
    assert brief.close_summary([]).count("Nothing was graded") == 1


# ------------------------------------------------------------------ CLI


def test_cli_train_ideas_trains_promotes_then_rejects(
    db: Session,
    world: dict[str, Bar],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instruments.ensure_reference_instruments(db)
    db.commit()
    monkeypatch.setattr(cli, "get_settings", lambda: training_settings())
    # No universe at all: the honest failure path.
    assert cli.main(["train-ideas", "--horizon", "swing"]) == 1
    assert "no universe instruments" in capsys.readouterr().err

    def fake_training_data(db: Session, horizon: str, **kw: Any) -> Any:
        return ideas_models.TrainingData(
            horizon=horizon,
            through=date(2025, 2, 24),
            frame=synthetic_candidates(),
            universe=universe_frame(),
            n_stocks=40,
        )

    monkeypatch.setattr(ideas_models, "training_data", fake_training_data)
    monkeypatch.setattr(
        ideas_models, "DEFAULT_PARAMS", {**ideas_models.DEFAULT_PARAMS, **LOOSE_PARAMS}
    )
    assert cli.main(["train-ideas", "--horizon", "swing"]) == 0
    out = capsys.readouterr().out
    assert "== swing (10 sessions, target 2 x ATR, stop 1 x ATR)" in out
    assert "40 universe stocks" in out and "3000 strategy candidates" in out
    assert "hit rate of ideas over the bar=" in out and "baseline all_candidates" in out
    assert "strategy breakout_20d" in out and "expectancy after costs" in out
    assert "PROMOTED" in out and "no champion yet" in out
    champ = ideas_models.ideas_champion(db, "swing", training_settings())
    assert champ is not None and champ.name == "ideas_swing"
    assert ideas_models.ideas_champion(db, "positional", training_settings()) is None

    assert cli.main(["train-ideas", "--horizon", "swing"]) == 0
    out = capsys.readouterr().out
    assert "rejected" in out and "not more training rows" in out
    db.expire_all()
    versions = list(db.scalars(select(ModelVersion).where(ModelVersion.name == "ideas_swing")))
    assert len(versions) == 2 and sum(v.is_champion for v in versions) == 1

    assert cli.main(["train-ideas", "--horizon", "swing", "--promote-always"]) == 0
    assert "--promote-always" in capsys.readouterr().out
    db.expire_all()
    promoted = ideas_models.ideas_champion(db, "swing", training_settings())
    assert promoted is not None and promoted.id != champ.id


def test_cli_ideas_passes_force_through_and_voids(
    db: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        runner, "run_job", lambda spec, **kw: calls.append((spec.name, kw["force"])) or "success"
    )
    assert cli.main(["ideas", "--force"]) == 0
    assert cli.main(["ideas"]) == 0
    assert calls == [("premarket_prediction", True), ("premarket_prediction", False)]

    assert cli.main(["ideas", "--void", "1"]) == 2  # no reason
    assert "--reason" in capsys.readouterr().err
    assert cli.main(["ideas", "--reason", "x"]) == 2
    capsys.readouterr()
    assert cli.main(["ideas", "--void", "999", "--reason", "x"]) == 1
    assert "no idea with seq 999" in capsys.readouterr().err
    inst = make_instrument(db)
    idea = ledger.record_idea(db, make_idea(inst.id))
    db.commit()
    args = ["ideas", "--void", str(idea.seq), "--reason", "bar never published", "--category"]
    assert cli.main([*args, "data_error"]) == 0
    assert f"voided idea seq {idea.seq}" in capsys.readouterr().out
    o = _outcome(db, idea)
    assert o is not None and o.status == "void" and o.error_category == "data_error"
    assert ledger.open_ideas(db) == []


def test_cli_paper_manual_trades_and_summary(
    db: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["paper"]) == 0
    out = capsys.readouterr().out
    assert "== paper portfolio (capital 1,000,000)" in out and "closed trades: 0  open: 0" in out
    assert "hit rate: —" in out and "sample-size warning" in out and "Paper portfolio" in out

    assert cli.main(["paper", "--trade", "RELIANCE", "10"]) == 2  # a price is never guessed
    assert "--price" in capsys.readouterr().err
    assert cli.main(["paper", "--trade", "RELIANCE", "ten", "--price", "2500"]) == 2
    capsys.readouterr()
    assert (
        cli.main(["paper", "--trade", "RELIANCE", "10", "--price", "2500", "--stop", "2600"]) == 1
    )
    assert "not entered" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "paper",
                "--trade",
                "reliance",
                "10",
                "--price",
                "2500",
                "--stop",
                "2400",
                "--target",
                "2700",
                "--date",
                "2026-10-30",
                "--note",
                "hand entry",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "buy 10 RELIANCE at 2500.00 on 2026-10-30 (stop 2400.00, target 2700.00)" in out
    (trade,) = _trades(db)
    assert trade.idea_id is None and trade.note == "hand entry" and trade.status == "open"
    assert cli.main(["paper"]) == 0
    out = capsys.readouterr().out
    assert (
        "open: 1" in out
        and f"open  #{trade.id} RELIANCE x10 at 2500.00" in out
        and "(manual)" in out
    )

    assert cli.main(["paper", "--close", str(trade.id)]) == 2
    capsys.readouterr()
    assert cli.main(["paper", "--close", "999", "--price", "1"]) == 1
    capsys.readouterr()
    assert (
        cli.main(["paper", "--close", str(trade.id), "--price", "2600", "--date", "2026-11-02"])
        == 0
    )
    out = capsys.readouterr().out
    assert f"paper trade {trade.id}: RELIANCE closed at 2600.00 on 2026-11-02" in out
    assert "gross 1,000" in out and "net " in out and "R 0." in out
    (trade,) = _trades(db)
    assert trade.status == "closed" and trade.net_pnl is not None and 0 < trade.net_pnl < 1000
    assert cli.main(["paper"]) == 0
    out = capsys.readouterr().out
    assert "closed trades: 1  open: 0  wins/losses: 1/0  hit rate: 100%" in out
    assert "profit factor: —" in out and f"closed #{trade.id} RELIANCE x10" in out


def test_cli_verify_ledger_reports_both_chains(
    db: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["verify-ledger"]) == 0
    out = capsys.readouterr().out
    assert "prediction: OK: ledger is empty" in out and "idea: OK: ledger is empty" in out
    inst = make_instrument(db)
    ledger.record_idea(db, make_idea(inst.id))
    ledger.record_idea(db, make_idea(inst.id, horizon="positional"))
    db.commit()
    assert cli.main(["verify-ledger"]) == 0
    out = capsys.readouterr().out
    assert "prediction: OK: ledger is empty" in out and "idea: OK: 2 records verified" in out

    db.execute(text("ALTER TABLE idea DISABLE TRIGGER idea_immutable"))
    db.execute(text("UPDATE idea SET note = 'edited' WHERE seq = 2"))
    db.execute(text("ALTER TABLE idea ENABLE TRIGGER idea_immutable"))
    db.commit()
    assert cli.main(["verify-ledger"]) == 1
    out = capsys.readouterr().out
    assert "prediction: OK" in out and "idea: BROKEN" in out
    assert "seq 2" in out and "first bad seq: 2 (1 records verified before it)" in out


# ------------------------------------------------------------------ schedule, workflow, env


def test_intraday_monitor_stays_off_with_the_always_on_note() -> None:
    spec = next(s for s in job_specs() if s.name == "intraday_monitor")
    assert spec.fn is None and spec.phase == 3
    assert "needs an always-on host" in spec.schedule and "SETUP.md" in spec.schedule
    with pytest.raises(RuntimeError, match="not built yet"):
        run_job(spec, now=FRIDAY_0920)


def _workflow() -> dict[str, Any]:
    doc = yaml.safe_load((REPO / ".github" / "workflows" / "dalal-jobs.yml").read_text())
    return {"on": doc[True] if True in doc else doc["on"], "jobs": doc["jobs"]}


def _bash_matches(cmd: str) -> bool:
    res = subprocess.run(
        ["bash", "-c", '[[ "$1" =~ $2 ]]', "_", cmd, ALLOWLIST], capture_output=True
    )
    return res.returncode == 0


def test_jobs_workflow_offers_the_phase3_commands_and_the_allowlist_holds() -> None:
    wf = _workflow()
    options = wf["on"]["workflow_dispatch"]["inputs"]["command"]["options"]
    assert {"train-ideas", "ideas --force"} <= set(options)
    steps = {s.get("id") or s.get("name"): s for s in wf["jobs"]["run"]["steps"]}
    pick, run = steps["pick"]["run"], steps["Run"]["run"]
    assert f"ALLOW='{ALLOWLIST}'" in pick and '[[ "$CMD" =~ $ALLOW ]]' in pick
    assert 'read -ra ARGS <<< "$CMD"' in run
    # Evaluated by bash exactly as the workflow does: every option, the horizon form...
    for opt in options:
        assert _bash_matches(opt), opt
    for ok in ("train-ideas --horizon swing", "train-ideas --horizon positional", "ideas"):
        assert _bash_matches(ok), ok
    # ...and nothing else: no injection, no extra flags, no other commands.
    for bad in (
        "train-ideas --horizon swing; echo pwned",
        "ideas --force && cat .env",
        "ideas --force --now",
        "ideas --void 1 --reason x",
        "train-ideas --horizon Swing",
        "train-ideas --horizon swing1",
        "train-ideas --horizon",
        "train-ideas --promote-always",
        "train-ideas  --horizon swing",
        "paper --trade X 1 --price 1",
        " ideas",
        "ideas ",
        "",
    ):
        assert not _bash_matches(bad), bad
    # The schedule is unchanged: the ideas ride on the three Phase 2 jobs.
    assert '"57 1 * * 1-5")    CMD="run-job premarket_prediction"' in pick
    assert "run-job open_check" in pick and "run-job close_grading" in pick


def test_vercel_env_exports_the_phase3_tunables(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "vercel_env", REPO / ".github" / "scripts" / "vercel_env.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wanted = (
        "IDEAS_MAX_PER_HORIZON",
        "IDEAS_MIN_P_TARGET",
        "IDEAS_MIN_EV_PCT",
        "IDEAS_UNIVERSE_MIN_BARS",
        "DRIVERS_TOP_N",
        "SWING_HORIZON_DAYS",
        "SWING_TARGET_ATR",
        "SWING_STOP_ATR",
        "POSITIONAL_HORIZON_DAYS",
        "POSITIONAL_TARGET_ATR",
        "POSITIONAL_STOP_ATR",
        "BROKERAGE_PCT",
        "STT_DELIVERY_PCT",
        "EXCHANGE_TXN_PCT",
        "GST_PCT",
        "STAMP_DUTY_PCT",
        "SEBI_FEE_PCT",
        "SLIPPAGE_PCT",
        "PAPER_MAX_POSITION_PCT",
        "PAPER_CAPITAL",
        "RISK_PER_TRADE_PCT",
        "MIN_AVG_TURNOVER_CR",
    )
    for key in wanted:
        assert key in mod.APP_KEYS, key
        assert key.lower() in Settings.model_fields, key  # every key is a real setting
        assert not mod.SECRETISH.search(key), f"{key} would be masked as a secret"
    f = tmp_path / ".env"
    f.write_text('DATABASE_URL="postgres://x"\nSECRET_KEY="s"\nIDEAS_MIN_P_TARGET="0.6"\n')
    env = mod.parse(str(f))
    lines = [mod.env_line(k, env[k]) for k in mod.APP_KEYS if mod.readable(env, k)]
    assert "IDEAS_MIN_P_TARGET=0.6" in lines


def test_premarket_marks_a_late_idea(
    db: Session,
    champion: ModelVersion,  # noqa: F811
    stocks: dict[str, list[Bar]],
    ideas_champions: dict[str, ModelVersion],
    fakes: dict[str, Any],
) -> None:
    late = datetime(2026, 10, 30, 9, 40, tzinfo=IST)
    assert _run("premarket_prediction", late) == "success"
    ideas = _ideas(db)
    assert ideas and all(i.is_late for i in ideas)
    assert all(i.as_of == late for i in ideas) and all(i.as_of.date() == FRIDAY for i in ideas)
    (msg,) = fakes["sent"]
    assert "[recorded late]" in msg
    d = _last_run(db, "premarket_prediction").details["ideas"]["swing"]["ideas"][0]
    assert d["is_late"] is True and d["surveillance_unknown"] is True
    assert SESSION and AS_OF  # the shared ledger constants are importable (owner C's helpers)

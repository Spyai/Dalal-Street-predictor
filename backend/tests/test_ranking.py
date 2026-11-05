"""Cost arithmetic, price levels, the p / EV bar, EV ranking with caps, position sizing.

Numbers are computed by hand from the Settings defaults so a silent change in the cost
model or the horizon multiples fails here first. No database."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from app.config import Settings, get_settings
from app.ml import costs, ranking
from app.ml import strategies as st
from app.ml.core import CostModel, Driver, HorizonSpec, IdeaOutput, horizons
from app.ml.strategies import Candidate

D = date(2026, 10, 30)
# Round trip with the default Settings (per side: brokerage 0.03 + exchange 0.00345 + SEBI
# 0.0001, GST 18 % on brokerage + exchange, STT 0.1, slippage 0.1; both sides; stamp duty
# 0.015 once).
PER_SIDE = 0.03 + 0.00345 + 0.0001 + (0.03 + 0.00345) * 0.18 + 0.1 + 0.1
ROUND_TRIP = 2.0 * PER_SIDE + 0.015  # 0.494142


def settings(**over: object) -> Settings:
    return get_settings().model_copy(update=over)


def candidate(
    symbol: str = "AAA",
    strategy: str = "breakout_20d",
    close: float | None = 100.0,
    atr: float | None = 2.0,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        session_date=D,
        strategy=strategy,
        setup="breakout",
        reasons=["Close Rs 100.00 is +1.0% above the prior 20-session high (Rs 99.01)"],
        invalidation="A close back below the breakout level (Rs 99.01): the breakout failed",
        close=close,
        atr14_pct=atr,
    )


def scored(
    symbol: str,
    p: float,
    *,
    horizon: str = "swing",
    strategy: str = "breakout_20d",
    close: float | None = 100.0,
    atr: float | None = 2.0,
) -> ranking.ScoredCandidate:
    return ranking.ScoredCandidate(
        candidate=candidate(symbol, strategy, close, atr),
        p_target=p,
        horizon=horizons()[horizon],
    )


# ------------------------------------------------------------------ costs


def test_round_trip_cost_with_the_default_settings() -> None:
    assert pytest.approx(0.494142, abs=1e-9) == ROUND_TRIP
    assert costs.round_trip_cost_pct() == pytest.approx(ROUND_TRIP)
    cm = costs.cost_model()
    assert isinstance(cm, CostModel) and cm.round_trip_pct() == pytest.approx(ROUND_TRIP)
    assert costs.round_trip_cost_pct(cm) == costs.round_trip_cost_pct(get_settings())
    assert (cm.brokerage_pct, cm.stt_delivery_pct, cm.slippage_pct) == (0.03, 0.1, 0.1)
    # The model is configurable: halve slippage and the round trip drops by 0.1 points.
    cheaper = costs.round_trip_cost_pct(settings(slippage_pct=0.05))
    assert cheaper == pytest.approx(ROUND_TRIP - 0.1)
    assert costs.cost_amount(100_000.0) == pytest.approx(494.142)


def test_expected_value_and_breakeven() -> None:
    # p 0.6, target +4 %, stop -2 %, cost 0.5 %: 2.4 - 0.8 - 0.5
    assert costs.expected_value_pct(0.6, 4.0, 2.0, 0.5) == pytest.approx(1.1)
    assert costs.expected_value_pct(0.0, 4.0, 2.0, 0.0) == pytest.approx(-2.0)
    assert costs.expected_value_pct(1.0, 4.0, 2.0, 0.0) == pytest.approx(4.0)
    be = costs.breakeven_p_target(4.0, 2.0, 0.5)
    assert be == pytest.approx(2.5 / 6.0)
    assert costs.expected_value_pct(be, 4.0, 2.0, 0.5) == pytest.approx(0.0)
    # With the default costs a 2:1 setup needs p above ~0.42 just to break even.
    assert costs.breakeven_p_target(4.0, 2.0, ROUND_TRIP) == pytest.approx((2 + ROUND_TRIP) / 6)


def test_distances_returns_and_r_multiples() -> None:
    assert costs.pct_distance(100.0, 104.0) == pytest.approx(4.0)
    assert costs.gross_return_pct(100.0, 98.0) == pytest.approx(-2.0)
    assert costs.net_return_pct(4.0, 0.5) == pytest.approx(3.5)
    assert costs.risk_reward(4.0, 2.0) == 2.0
    assert costs.r_multiple(3.0, 2.0) == 1.5
    assert costs.r_multiple(-2.5, 2.0) == -1.25
    assert costs.EXPIRY_RETURN_PCT == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: costs.expected_value_pct(float("nan"), 4.0, 2.0, 0.5),
        lambda: costs.expected_value_pct(0.6, None, 2.0, 0.5),  # type: ignore[arg-type]
        lambda: costs.expected_value_pct(1.2, 4.0, 2.0, 0.5),
        lambda: costs.expected_value_pct(0.6, 4.0, 0.0, 0.5),
        lambda: costs.breakeven_p_target(4.0, -2.0, 0.5),
        lambda: costs.risk_reward(4.0, 0.0),
        lambda: costs.r_multiple(1.0, float("inf")),
        lambda: costs.pct_distance(0.0, 5.0),
        lambda: costs.net_return_pct("abc", 0.5),  # type: ignore[arg-type]
        lambda: costs.cost_amount(-1.0),
    ],
)
def test_missing_or_invalid_numbers_raise(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


# ------------------------------------------------------------------ price levels


def test_price_levels_for_both_horizons() -> None:
    swing = horizons()["swing"]
    assert (swing.days, swing.target_atr, swing.stop_atr) == (10, 2.0, 1.0)
    lv = ranking.price_levels(100.0, 2.0, swing)  # ATR = Rs 2
    assert (lv.entry_low, lv.entry_mid, lv.entry_high) == (99.0, 100.0, 101.0)
    assert (lv.stop, lv.target_1, lv.target_2) == (98.0, 104.0, 106.0)
    assert (lv.target_pct, lv.stop_pct, lv.risk_reward) == (4.0, 2.0, 2.0)
    assert (lv.expected_return_low_pct, lv.expected_return_high_pct) == (4.0, 6.0)
    positional = horizons()["positional"]
    assert (positional.days, positional.target_atr, positional.stop_atr) == (60, 4.0, 2.0)
    lv2 = ranking.price_levels(100.0, 2.0, positional)
    assert (lv2.stop, lv2.target_1, lv2.target_2) == (96.0, 108.0, 112.0)
    assert (lv2.target_pct, lv2.stop_pct, lv2.risk_reward) == (8.0, 4.0, 2.0)
    # Prices are rounded to the paisa; the % distances are not.
    lv3 = ranking.price_levels(1234.567, 1.37, swing)
    assert lv3.entry_mid == 1234.57 and lv3.stop == round(1234.567 * (1 - 0.0137), 2)
    assert lv3.target_pct == pytest.approx(2.74)


@pytest.mark.parametrize(
    ("close", "atr"), [(None, 2.0), (100.0, None), (0.0, 2.0), (100.0, 0.0), (float("nan"), 2.0)]
)
def test_price_levels_never_guess(close: float | None, atr: float | None) -> None:
    with pytest.raises(ValueError):
        ranking.price_levels(close, atr, horizons()["swing"])


def test_price_levels_reject_a_stop_that_is_not_a_price() -> None:
    with pytest.raises(ValueError, match="not a price"):
        ranking.price_levels(100.0, 60.0, HorizonSpec("wild", 10, 2.0, 2.0))


# ------------------------------------------------------------------ output + bar


def test_build_output_carries_levels_ev_and_text() -> None:
    sc = scored("AAA", 0.6)
    sc = ranking.ScoredCandidate(
        sc.candidate, sc.p_target, sc.horizon, [Driver("rsi14", 27.0, 0.3, "RSI (14) at 27")]
    )
    out = ranking.build_output(sc)
    assert (out.symbol, out.exchange, out.horizon, out.horizon_days) == ("AAA", "NSE", "swing", 10)
    assert (out.setup, out.strategy) == ("breakout", "breakout_20d")
    assert (out.entry_low, out.entry_high, out.stop, out.target_1, out.target_2) == (
        99.0,
        101.0,
        98.0,
        104.0,
        106.0,
    )
    assert out.p_target == 0.6 and out.risk_reward == 2.0
    assert (out.expected_return_low_pct, out.expected_return_high_pct) == (4.0, 6.0)
    assert out.ev_pct_after_costs == pytest.approx(0.6 * 4 - 0.4 * 2 - ROUND_TRIP)
    assert out.reasons == sc.candidate.reasons and out.invalidation == sc.candidate.invalidation
    assert out.drivers[0].feature == "rsi14"
    assert ranking.entry_mid(out) == 100.0
    assert ranking.clears_bar(out)
    with pytest.raises(ValueError, match="probability"):
        ranking.build_output(scored("AAA", 1.2))
    with pytest.raises(ValueError, match="atr14_pct"):
        ranking.build_output(scored("AAA", 0.6, atr=None))


def test_the_bar_needs_both_probability_and_ev() -> None:
    s = get_settings()
    assert (s.ideas_min_p_target, s.ideas_min_ev_pct, s.ideas_max_per_horizon) == (0.40, 0.5, 3)
    # EV for a 2:1 setup = S (3p - 1) - cost. p 0.56, ATR 1 % -> S 1: 0.68 - 0.494 < 0.5.
    low_ev = ranking.build_output(scored("AAA", 0.56, atr=1.0))
    assert low_ev.p_target >= s.ideas_min_p_target and low_ev.ev_pct_after_costs < 0.5
    assert not ranking.clears_bar(low_ev)
    # p 0.39 on a very volatile stock (ATR 8 %) has EV +0.87 % and still fails the floor.
    low_p = ranking.build_output(scored("BBB", 0.39, atr=8.0))
    assert low_p.ev_pct_after_costs > 0.5 and not ranking.clears_bar(low_p)
    assert ranking.clears_bar(ranking.build_output(scored("CCC", 0.56, atr=2.0)))
    # p 0.46 on a 3 % ATR stock: EV = 3 (1.38 - 1) - 0.494 = +0.65 % -> over the bar.
    assert ranking.clears_bar(ranking.build_output(scored("DDD", 0.46, atr=3.0)))
    assert ranking.rank_ideas([scored("AAA", 0.56, atr=1.0), scored("BBB", 0.39, atr=8.0)]) == []
    stricter = settings(ideas_min_p_target=0.6)
    assert not ranking.clears_bar(ranking.build_output(scored("CCC", 0.56, atr=2.0)), stricter)


# ------------------------------------------------------------------ ranking


def test_rank_sorts_by_ev_caps_and_keeps_one_per_symbol() -> None:
    items = [
        scored("AAA", 0.60),  # EV 2*(1.8-1)... = 0.6*4 - 0.4*2 - c = 1.6 - c
        scored("BBB", 0.70, atr=3.0),  # 0.7*6 - 0.3*3 - c = 3.3 - c (best)
        scored("CCC", 0.58),  # 0.58*4 - 0.42*2 - c = 1.48 - c
        scored("DDD", 0.65),  # 0.65*4 - 0.35*2 - c = 1.9 - c
        scored("AAA", 0.75, strategy="momentum_12_1"),  # 0.75*4 - 0.25*2 - c = 2.5 - c
        scored("EEE", 0.38),  # 0.38*4 - 0.62*2 - c < 0: fails the bar (p and EV)
    ]
    report = ranking.rank_report(items)
    ideas = report.ideas
    assert [(o.symbol, o.strategy) for o in ideas] == [
        ("BBB", "breakout_20d"),
        ("AAA", "momentum_12_1"),  # the better of AAA's two setups
        ("DDD", "breakout_20d"),
    ]
    evs = [o.ev_pct_after_costs for o in ideas]
    assert evs == sorted(evs, reverse=True)
    assert evs[0] == pytest.approx(3.3 - ROUND_TRIP)
    assert len({o.symbol for o in ideas}) == 3
    assert report.n_candidates == 6 and report.n_cleared == 5 and report.n_unsizable == 0
    assert report.n_unpriced == 0 and report.no_edge == {}
    assert report.best_p_target == 0.75 and report.best_ev_pct == pytest.approx(3.3 - ROUND_TRIP)
    d = report.as_dict()
    assert d["n_ideas"] == 3 and d["n_candidates"] == 6
    assert ranking.rank_ideas(items) == ideas
    # A looser cap lets the fourth one through; a cap of 1 keeps the best only.
    assert [o.symbol for o in ranking.rank_ideas(items, settings(ideas_max_per_horizon=5))] == [
        "BBB",
        "AAA",
        "DDD",
        "CCC",
    ]
    assert [o.symbol for o in ranking.rank_ideas(items, settings(ideas_max_per_horizon=1))] == [
        "BBB"
    ]


def test_ties_are_broken_deterministically() -> None:
    items = [scored("ZZZ", 0.6), scored("AAA", 0.6), scored("MMM", 0.6)]
    assert [o.symbol for o in ranking.rank_ideas(items)] == ["AAA", "MMM", "ZZZ"]
    assert [o.symbol for o in ranking.rank_ideas(reversed(items))] == ["AAA", "MMM", "ZZZ"]
    # Same EV, different strategy on the same symbol: the earlier strategy in order wins.
    same = [scored("AAA", 0.6, strategy="pullback_20dma"), scored("AAA", 0.6)]
    assert [(o.symbol, o.strategy) for o in ranking.rank_ideas(same)] == [("AAA", "breakout_20d")]


def test_per_horizon_caps_are_independent_and_grouped() -> None:
    items = [scored(f"S{i}", 0.6 + i / 100, horizon="positional") for i in range(4)]
    items += [scored(f"P{i}", 0.6 + i / 100) for i in range(4)]
    ideas = ranking.rank_ideas(items)
    assert [o.horizon for o in ideas] == ["swing"] * 3 + ["positional"] * 3
    assert [o.symbol for o in ideas] == ["P3", "P2", "P1", "S3", "S2", "S1"]
    assert ideas[3].horizon_days == 60 and ideas[3].stop == 96.0 and ideas[3].target_1 == 108.0
    # One per symbol is per horizon: the same stock may be a swing and a positional idea.
    both = [scored("AAA", 0.7), scored("AAA", 0.7, horizon="positional")]
    assert [(o.symbol, o.horizon) for o in ranking.rank_ideas(both)] == [
        ("AAA", "swing"),
        ("AAA", "positional"),
    ]


def test_no_edge_today_when_nothing_clears() -> None:
    report = ranking.rank_report([scored("AAA", 0.38), scored("BBB", 0.39, horizon="positional")])
    assert report.ideas == [] and report.n_cleared == 0
    assert set(report.closest) == {"swing", "positional"}
    (swing_closest,) = report.closest["swing"]
    assert swing_closest["symbol"] == "AAA" and swing_closest["cleared"] is False
    assert swing_closest["p_target"] == 0.38 and swing_closest["ev_pct_after_costs"] < 0.5
    assert report.as_dict()["closest"]["positional"][0]["cleared"] is False
    assert set(report.no_edge) == {"swing", "positional"}
    for text in report.no_edge.values():
        assert text.startswith("No edge today") and "40%" in text and "0.5%" in text
    assert ranking.no_edge_text() == (
        "No edge today: no candidate clears P(target) >= 40% and EV >= 0.5% after costs"
    )
    assert ranking.no_edge_text("swing").startswith("No edge today (swing):")
    assert ranking.NO_EDGE_TEXT == "No edge today"
    # One horizon with ideas, the other without: only the empty one gets the line.
    mixed = ranking.rank_report([scored("AAA", 0.7), scored("BBB", 0.39, horizon="positional")])
    assert [o.symbol for o in mixed.ideas] == ["AAA"] and set(mixed.no_edge) == {"positional"}
    assert mixed.closest["swing"][0]["cleared"] is True
    assert mixed.closest["positional"][0] == {
        "symbol": "BBB",
        "strategy": "breakout_20d",
        "p_target": 0.39,
        "ev_pct_after_costs": mixed.closest["positional"][0]["ev_pct_after_costs"],
        "cleared": False,
    }
    assert mixed.closest["positional"][0]["ev_pct_after_costs"] < 0.5
    assert ranking.rank_report([]).no_edge == {} and ranking.rank_ideas([]) == []


def test_unpriced_and_unsizable_candidates_are_dropped_not_guessed() -> None:
    report = ranking.rank_report(
        [
            scored("NOATR", 0.9, atr=None),
            scored("NOPX", 0.9, close=None),
            scored("RICH", 0.9, close=150_000.0),  # one share breaks the 10 % position cap
            scored("OK", 0.6),
        ]
    )
    assert [o.symbol for o in report.ideas] == ["OK"]
    assert report.n_unpriced == 2 and report.n_unsizable == 1 and report.n_cleared == 2
    assert report.best_p_target == 0.9  # the unsizable one was still priced and scored


def test_tuple_input_form() -> None:
    drivers = [Driver("rsi14", 27.0, 0.3, "RSI (14) at 27")]
    items = [
        (candidate("AAA"), 0.65, {"horizon": "swing", "drivers": drivers, "exchange": "BSE"}),
        (candidate("BBB"), 0.65, {"horizon": horizons()["positional"]}),
    ]
    ideas = ranking.rank_ideas(items)
    assert [(o.symbol, o.horizon, o.exchange) for o in ideas] == [
        ("AAA", "swing", "BSE"),
        ("BBB", "positional", "NSE"),
    ]
    assert ideas[0].drivers == drivers and ideas[1].drivers == []
    with pytest.raises(ValueError, match="horizon"):
        ranking.rank_ideas([(candidate("AAA"), 0.65, {})])
    with pytest.raises(KeyError):
        ranking.rank_ideas([(candidate("AAA"), 0.65, {"horizon": "intraday"})])


# ------------------------------------------------------------------ sizing


def test_size_position_risk_budget_and_position_cap() -> None:
    s = get_settings()
    assert (s.paper_capital, s.risk_per_trade_pct, s.paper_max_position_pct) == (1e6, 1.0, 10.0)
    # Risk budget Rs 10,000. Entry 100, stop 90: 1,000 shares = Rs 1 lakh, exactly the cap.
    assert ranking.size_position(100.0, 90.0) == (1000, 100_000.0, 10_000.0)
    # Entry 100, stop 98: the budget allows 5,000 shares (Rs 5 lakh) but the cap cuts to 1,000.
    assert ranking.size_position(100.0, 98.0) == (1000, 100_000.0, 2_000.0)
    # Entry 1,000, stop 990: 1,000 shares by risk, 100 by the cap.
    assert ranking.size_position(1000.0, 990.0) == (100, 100_000.0, 1_000.0)
    # Wide stop: the risk budget binds, the position is small.
    assert ranking.size_position(100.0, 75.0) == (400, 40_000.0, 10_000.0)
    # Rounding down: 10,000 / 3 = 3,333.33 -> 3,333 shares, risk just under the budget.
    qty, value, risk = ranking.size_position(20.0, 17.0)
    assert (qty, value) == (3333, 66_660.0) and risk == pytest.approx(9_999.0)
    # Too expensive for one share within the cap.
    assert ranking.size_position(150_000.0, 147_000.0) == (0, 0.0, 0.0)
    custom = settings(paper_capital=500_000.0, risk_per_trade_pct=2.0, paper_max_position_pct=50.0)
    assert ranking.size_position(100.0, 98.0, custom) == (2500, 250_000.0, 5_000.0)


@pytest.mark.parametrize(
    ("entry", "stop"), [(100.0, 100.0), (100.0, 101.0), (None, 98.0), (100.0, None), (0.0, -1.0)]
)
def test_size_position_rejects_bad_inputs(entry: float | None, stop: float | None) -> None:
    with pytest.raises(ValueError):
        ranking.size_position(entry, stop)  # type: ignore[arg-type]


def test_strategies_to_ranking_end_to_end() -> None:
    """A feature-pipeline-shaped cross-section -> candidates -> scored -> ranked ideas: the
    words, the levels and the sizing travel together, one idea per symbol per horizon."""
    rows = pd.DataFrame(
        [
            {
                "symbol": "BRK",
                "session_date": D,
                "instrument_id": 11,
                "close": 500.0,
                "eligible": True,
                "atr14_pct": 2.0,
                "vol_z20": 1.5,
                "rsi14": 61.0,
                "dist_sma20_pct": 3.0,
                "dist_sma50_pct": 6.0,
                "dist_sma200_pct": 8.0,
                "down_streak": 0.0,
                "brk_20d_high_pct": 1.2,
                "rs_12_1": 5.0,
                "ret_20d": 6.0,
            },
            {
                "symbol": "OUT",
                "session_date": D,
                "instrument_id": 12,
                "close": 500.0,
                "eligible": False,  # outside the universe at D: never an idea
                "atr14_pct": 2.0,
                "vol_z20": 2.5,
                "rsi14": 61.0,
                "dist_sma20_pct": 3.0,
                "dist_sma50_pct": 6.0,
                "dist_sma200_pct": 8.0,
                "down_streak": 0.0,
                "brk_20d_high_pct": 2.2,
                "rs_12_1": 50.0,
                "ret_20d": 9.0,
            },
        ]
    ).set_index(["symbol", "session_date"])
    cands = st.candidates(rows, session=D)
    assert [(c.symbol, c.strategy, c.instrument_id) for c in cands] == [("BRK", "breakout_20d", 11)]
    swing, positional = horizons()["swing"], horizons()["positional"]
    scored_items = [
        ranking.ScoredCandidate(cands[0], 0.62, swing),
        ranking.ScoredCandidate(cands[0], 0.58, positional),
        ranking.ScoredCandidate(cands[0], 0.5, swing),  # a second swing score for the same symbol
    ]
    ideas = ranking.rank_ideas(scored_items)
    assert [(o.symbol, o.horizon, o.p_target) for o in ideas] == [
        ("BRK", "swing", 0.62),
        ("BRK", "positional", 0.58),
    ]
    out: IdeaOutput = ideas[0]
    assert out.reasons == cands[0].reasons and out.invalidation == cands[0].invalidation
    assert out.reasons[0] == "Close Rs 500.00 is +1.2% above the prior 20-session high (Rs 494.07)"
    assert (out.entry_low, out.entry_high, out.stop, out.target_1) == (495.0, 505.0, 490.0, 520.0)
    qty, value, risk = ranking.size_position(ranking.entry_mid(out), out.stop)
    assert (qty, value, risk) == (200, 100_000.0, 2_000.0)  # the 10 % position cap binds


def test_sized_follows_the_ideas() -> None:
    ideas = ranking.rank_ideas([scored("AAA", 0.7), scored("BBB", 0.6, atr=4.0)])
    sizes = ranking.sized(ideas)
    assert len(sizes) == 2
    assert sizes[0] == ranking.size_position(100.0, ideas[0].stop)
    assert all(q > 0 and math.isfinite(v) and r <= 10_000.0 + 1e-9 for q, v, r in sizes)

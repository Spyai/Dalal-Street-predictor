"""Rule-based candidate generators: each rule fires on the row built for it and on no
other, the words quote the numbers, missing inputs never fire, ineligible rows never fire,
and the momentum decile is ranked on the eligible cross-section. The rows use the column
names `app.ml.ideas_features` writes (brk_20d_high_pct, down_streak, eligible, ...); the
aliases are covered once. Pure pandas, no database."""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.ml import ideas_features as idf
from app.ml import strategies as st

D = date(2026, 10, 30)
D_PREV = date(2026, 10, 29)
HAS_DIGIT = re.compile(r"\d")


def base_row(symbol: str = "AAA", session: date = D, **over: object) -> dict[str, object]:
    """A row on which no rule fires: below its 20-session high, RSI 50, 20-DMA under the
    50-DMA, flat relative strength (a lone row cannot be ranked anyway)."""
    row: dict[str, object] = {
        "symbol": symbol,
        "session_date": session,
        "instrument_id": 7,
        "close": 500.0,
        "eligible": True,
        "atr14_pct": 2.0,
        "vol_z20": 0.0,
        "rsi14": 50.0,
        "dist_sma20_pct": 0.0,
        "dist_sma50_pct": -1.0,
        "dist_sma200_pct": -5.0,
        "down_streak": 0.0,
        "brk_20d_high_pct": -2.0,
        "rs_12_1": 0.0,
        "ret_20d": 1.0,
    }
    row.update(over)
    return row


BREAKOUT = {"brk_20d_high_pct": 1.2, "vol_z20": 1.5, "dist_sma200_pct": 8.0, "ret_20d": 6.0}
MEAN_REVERSION = {"rsi14": 27.0, "dist_sma200_pct": 4.0, "down_streak": 3.0}
PULLBACK = {"dist_sma20_pct": -1.0, "dist_sma50_pct": 2.0, "dist_sma200_pct": 9.0, "ret_20d": 7.5}


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def momentum_frame(n: int = 20) -> pd.DataFrame:
    """n symbols with relative strength 0..n-1, all above their 50-DMA."""
    return frame(*(base_row(f"M{i:02d}", rs_12_1=float(i), dist_sma50_pct=1.0) for i in range(n)))


# ------------------------------------------------------------------ which rule fires


def test_each_rule_fires_only_on_its_row() -> None:
    df = frame(
        base_row("NONE"),
        base_row("BRK", **BREAKOUT),
        base_row("MRV", **MEAN_REVERSION),
        base_row("PLB", **PULLBACK),
    )
    sig = st.signals(df)
    assert list(sig.columns) == list(st.STRATEGY_NAMES)
    assert sig.dtypes.eq(bool).all()
    assert sig.sum().to_dict() == {
        "breakout_20d": 1,
        "momentum_12_1": 0,  # four rows cannot make a decile without a rank column
        "mean_reversion_rsi": 1,
        "pullback_20dma": 1,
    }
    cands = st.candidates(df)
    assert [(c.symbol, c.strategy, c.setup) for c in cands] == [
        ("BRK", "breakout_20d", "breakout"),
        ("MRV", "mean_reversion_rsi", "mean_reversion"),
        ("PLB", "pullback_20dma", "pullback"),
    ]
    assert all(c.session_date == D for c in cands)
    assert all(c.close == 500.0 and c.atr14_pct == 2.0 and c.instrument_id == 7 for c in cands)
    assert set(st.SETUPS.values()) == {"breakout", "momentum", "mean_reversion", "pullback"}
    assert all(len(setup) <= 32 for setup in st.SETUPS.values())  # Idea.setup is String(32)


def test_every_concept_resolves_on_the_feature_pipelines_frame() -> None:
    """The rules read the columns `ideas_features.stock_features` actually writes."""
    frame = pd.DataFrame(columns=[*idf.META_COLUMNS, *idf.FEATURE_NAMES])
    found = st.resolve_columns(frame)
    # Every number a rule reads has a column; the two spellings of the streak and of the
    # breakout distance are interchangeable, and the momentum rank is computed here.
    either = {"down_streak", "up_streak"}
    assert any(found[c] for c in either), "no streak column"
    for concept, col in found.items():
        if concept in either or concept == "rs_12_1_rank":
            continue
        assert col is not None, concept
        assert col in idf.META_COLUMNS or col in idf.FEATURE_NAMES
    assert found["rs_12_1_rank"] is None  # the rank is this module's column (RANK_COLUMN)
    assert st.RANK_COLUMN not in idf.FEATURE_NAMES
    assert set(st.IDEA_FEATURE_LABELS).isdisjoint(idf.FEATURE_LABELS)
    # The down_streak helper reads whichever streak the pipeline writes.
    if found["up_streak"] is not None:
        rows = pd.DataFrame({found["up_streak"]: [-3.0, 2.0, 0.0, np.nan]})
    else:
        rows = pd.DataFrame({found["down_streak"]: [3.0, 0.0, 0.0, np.nan]})
    assert st.down_streak(rows).tolist()[:3] == [3.0, 0.0, 0.0]
    assert np.isnan(st.down_streak(rows).tolist()[3])


def test_reasons_quote_numbers_and_invalidation_names_a_level() -> None:
    df = frame(
        base_row("BRK", **BREAKOUT),
        base_row("MRV", **MEAN_REVERSION),
        base_row("PLB", **PULLBACK),
        base_row("MOM", rs_12_1=30.0, rs_12_1_rank=0.96, dist_sma50_pct=3.0),
    )
    cands = st.candidates(df)
    assert {c.strategy for c in cands} == set(st.STRATEGY_NAMES)
    for c in cands:
        assert 3 <= len(c.reasons) <= 5, c
        assert all(HAS_DIGIT.search(r) for r in c.reasons), c.reasons
        assert "Rs " in c.invalidation and HAS_DIGIT.search(c.invalidation), c.invalidation


def test_breakout_words_recover_the_breakout_level() -> None:
    (c,) = st.candidates(frame(base_row("BRK", **BREAKOUT)))
    high20 = 500.0 / 1.012  # close / (1 + dist/100)
    assert f"Rs {high20:,.2f}" == "Rs 494.07"
    assert c.reasons[0] == "Close Rs 500.00 is +1.2% above the prior 20-session high (Rs 494.07)"
    assert "Volume +1.5 sd" in c.reasons[1] and ">= +1.0" in c.reasons[1]
    assert c.reasons[2].startswith("+8.0% above the 200-DMA")
    assert "+6.0% over the prior 20 sessions" in c.reasons[3]
    assert "ATR(14) is 2.0% of price" in c.reasons[4]
    assert (
        c.invalidation == "A close back below the breakout level (Rs 494.07): the breakout failed"
    )


def test_mean_reversion_and_pullback_words() -> None:
    (mrv,) = st.candidates(frame(base_row("MRV", **MEAN_REVERSION)))
    assert mrv.reasons[0] == "RSI(14) at 27: oversold (below 30)"
    assert mrv.reasons[1] == "3 consecutive down closes"
    assert mrv.reasons[2].startswith("Still +4.0% above the 200-DMA (Rs 480.77)")
    assert mrv.invalidation.startswith("A close below the 200-DMA (Rs 480.77)")

    (plb,) = st.candidates(frame(base_row("PLB", **PULLBACK)))
    assert plb.reasons[0] == "Uptrend: 20-DMA Rs 505.05 > 50-DMA Rs 490.20 > 200-DMA Rs 458.72"
    assert plb.reasons[1] == "Close -1.0% from the 20-DMA (within +/-1.5%)"
    assert plb.reasons[2] == "+7.5% run over the prior 20 sessions"
    assert plb.invalidation == "A close below the 50-DMA (Rs 490.20) ends the pullback setup"


def test_rule_boundaries() -> None:
    # Breakout needs ALL of: above the prior high, volume z >= 1, above the 200-DMA.
    assert not st.candidates(frame(base_row(**{**BREAKOUT, "brk_20d_high_pct": 0.0})))
    assert not st.candidates(frame(base_row(**{**BREAKOUT, "vol_z20": 0.99})))
    assert st.candidates(frame(base_row(**{**BREAKOUT, "vol_z20": 1.0})))  # inclusive
    assert not st.candidates(frame(base_row(**{**BREAKOUT, "dist_sma200_pct": -0.1})))
    # Mean reversion: RSI strictly below 30, at least three down closes, above the 200-DMA.
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "rsi14": 30.0})))
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "down_streak": 2.0})))
    assert st.candidates(frame(base_row(**{**MEAN_REVERSION, "down_streak": 4.0})))
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "dist_sma200_pct": 0.0})))
    # Pullback: 20 > 50 > 200 DMA, within 1.5 % of the 20-DMA, a >= 5 % run.
    assert not st.candidates(frame(base_row(**{**PULLBACK, "dist_sma20_pct": -1.6})))
    assert st.candidates(frame(base_row(**{**PULLBACK, "dist_sma20_pct": 1.5})))
    assert not st.candidates(frame(base_row(**{**PULLBACK, "ret_20d": 4.9})))
    # 20-DMA below the 50-DMA (close further above the 20-DMA than the 50-DMA): no uptrend.
    assert not st.candidates(frame(base_row(**{**PULLBACK, "dist_sma50_pct": -1.2})))


def test_ineligible_rows_never_fire_and_never_enter_the_decile() -> None:
    """`eligible` is the point-in-time universe verdict: a stock outside the universe at D
    is not an idea and does not take a decile slot from one that is."""
    df = frame(
        base_row("BRK", eligible=False, **BREAKOUT),
        base_row("MRV", eligible=None, **MEAN_REVERSION),
        base_row("PLB", **PULLBACK),
    )
    assert [c.symbol for c in st.candidates(df)] == ["PLB"]
    assert st.signals(df)["breakout_20d"].tolist() == [False, False, False]
    assert st.candidate_frame(df)["symbol"].tolist() == ["PLB"]
    # Twenty stocks, the ten strongest ineligible: the decile is ranked among the ten left
    # (percentiles 0.1 .. 1.0, so the top two clear 0.9), never the ten outside.
    mom = momentum_frame(20)
    mom.loc[mom["rs_12_1"] >= 10, "eligible"] = False
    assert [c.symbol for c in st.candidates(mom)] == ["M08", "M09"]
    assert st.rs_rank(mom).tolist()[:10] == pytest.approx([0.1 * (i + 1) for i in range(10)])
    assert st.rs_rank(mom).isna().tolist()[10:] == [True] * 10
    # Without the column every row counts as eligible (the caller filtered the universe).
    assert [c.symbol for c in st.candidates(mom.drop(columns=["eligible"]))] == [
        "M17",
        "M18",
        "M19",
    ]


def test_missing_inputs_never_fire_and_missing_numbers_leave_no_reason() -> None:
    assert not st.candidates(frame(base_row(**{**BREAKOUT, "vol_z20": np.nan})))
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "rsi14": None})))
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "down_streak": np.nan})))
    assert not st.candidates(frame(base_row(**{**PULLBACK, "ret_20d": np.nan})))
    # A column the frame lacks entirely counts as missing (volume is NaN-prone in life).
    df = frame(base_row(**BREAKOUT)).drop(columns=["vol_z20"])
    assert not st.candidates(df)
    # Optional numbers only add words when present.
    (c,) = st.candidates(frame(base_row(**{**BREAKOUT, "ret_20d": np.nan, "atr14_pct": np.nan})))
    assert len(c.reasons) == 3 and c.atr14_pct is None
    assert all("ATR" not in r and "20 sessions" not in r for r in c.reasons)


# ------------------------------------------------------------------ momentum ranking


def test_momentum_is_ranked_on_the_whole_cross_section() -> None:
    df = momentum_frame(20)
    cands = st.candidates(df)
    assert [c.symbol for c in cands] == ["M17", "M18", "M19"]  # percentile >= 0.9 of 20
    assert all(c.strategy == "momentum_12_1" and c.setup == "momentum" for c in cands)
    top = cands[-1]
    assert top.reasons[0] == (
        "12-1 month relative strength vs NIFTY 50 +19.0 pts: top 1% of the universe"
    )
    assert cands[0].reasons[0].endswith("top 10% of the universe")
    assert top.reasons[1] == "+1.0% above the 50-DMA (Rs 495.05)"
    assert top.invalidation == "A close below the 50-DMA (Rs 495.05) breaks the trend"
    # The strongest stock below its 50-DMA is not a momentum candidate.
    weak = df.copy()
    weak.loc[weak["symbol"] == "M19", "dist_sma50_pct"] = -0.5
    assert [c.symbol for c in st.candidates(weak)] == ["M17", "M18"]
    # The rank is computed per session: two sessions, two independent deciles.
    two = pd.concat([df, df.assign(session_date=D_PREV, rs_12_1=-df["rs_12_1"])], ignore_index=True)
    by_session = {}
    for c in st.candidates(two):
        by_session.setdefault(c.session_date, []).append(c.symbol)
    assert by_session == {D_PREV: ["M00", "M01", "M02"], D: ["M17", "M18", "M19"]}


def test_explicit_rank_column_is_used_and_a_lone_row_cannot_be_ranked() -> None:
    lone = base_row("ONE", rs_12_1=40.0, dist_sma50_pct=2.0)
    assert not st.candidates(frame(lone))  # no cross-section, no rank column: no decile
    (c,) = st.candidates(frame({**lone, "rs_12_1_rank": 0.95}))
    assert c.strategy == "momentum_12_1" and "top 5% of the universe" in c.reasons[0]
    assert not st.candidates(frame({**lone, "rs_12_1_rank": 0.89}))
    # When the frame carries a rank it wins over ranking the rows given.
    ranked = momentum_frame(20).assign(rs_12_1_rank=0.5)
    assert not st.candidates(ranked)


def test_rs_rank_values() -> None:
    df = momentum_frame(4)
    assert st.rs_rank(df).tolist() == [0.25, 0.5, 0.75, 1.0]
    assert st.rs_rank(frame(base_row())).isna().all()  # a single row has no percentile
    no_rs = frame(base_row(), base_row("BBB")).drop(columns=["rs_12_1"])
    assert st.rs_rank(no_rs).isna().all()


# ------------------------------------------------------------------ frame shapes


def test_series_multiindex_and_session_filter_inputs() -> None:
    row = base_row("BRK", **BREAKOUT)
    assert [c.symbol for c in st.candidates(pd.Series(row))] == ["BRK"]
    flat = frame(row, base_row("OLD", session=D_PREV, **MEAN_REVERSION))
    # The feature pipeline's shape: a (symbol, session_date) MultiIndex, META then features.
    multi = flat.set_index(list(idf.INDEX_NAMES)).sort_index()
    assert [(c.symbol, c.strategy) for c in st.candidates(multi)] == [
        ("OLD", "mean_reversion_rsi"),
        ("BRK", "breakout_20d"),
    ]
    assert [
        (c.symbol, c.strategy) for c in st.candidates(flat.set_index(["session_date", "symbol"]))
    ] == [
        ("OLD", "mean_reversion_rsi"),
        ("BRK", "breakout_20d"),
    ]
    assert [c.symbol for c in st.candidates(flat, session=D)] == ["BRK"]
    assert st.candidates(flat, session=date(2026, 1, 1)) == []
    dated = flat.set_index("session_date")
    assert [c.symbol for c in st.candidates(dated, session=D_PREV)] == ["OLD"]
    with pytest.raises(ValueError, match="symbol"):
        st.candidates(flat.drop(columns=["symbol"]))


def test_candidate_frame_has_one_row_per_rule_hit_in_a_fixed_order() -> None:
    both = base_row("BOTH", **{**BREAKOUT, **MEAN_REVERSION, "vol_z20": 2.0})
    df = frame(base_row("NONE"), both, base_row("PLB", **PULLBACK))
    cf = st.candidate_frame(df)
    assert list(zip(cf["symbol"], cf["strategy"], cf["setup"], strict=True)) == [
        ("BOTH", "breakout_20d", "breakout"),
        ("BOTH", "mean_reversion_rsi", "mean_reversion"),
        ("PLB", "pullback_20dma", "pullback"),
    ]
    assert "rs_12_1_rank" in cf.columns  # the cross-sectional rank travels with the row
    assert set(df.columns) <= set(cf.columns)  # every stock feature is kept for training
    assert list(cf.index) == [0, 1, 2]
    empty = st.candidate_frame(frame(base_row("NONE")))
    assert empty.empty and {"strategy", "setup"} <= set(empty.columns)
    assert st.candidate_frame(df, session=date(2026, 1, 1)).empty


def test_one_hots() -> None:
    names = st.one_hot_names()
    assert names == [
        "strategy_breakout_20d",
        "strategy_momentum_12_1",
        "strategy_mean_reversion_rsi",
        "strategy_pullback_20dma",
    ]
    assert st.one_hot_row("mean_reversion_rsi") == {
        "strategy_breakout_20d": 0.0,
        "strategy_momentum_12_1": 0.0,
        "strategy_mean_reversion_rsi": 1.0,
        "strategy_pullback_20dma": 0.0,
    }
    with pytest.raises(ValueError, match="unknown strategy"):
        st.one_hot_row("results_play")
    hots = st.one_hots(pd.Series(["breakout_20d", None, "pullback_20dma"], index=[7, 8, 9]))
    assert list(hots.columns) == names and list(hots.index) == [7, 8, 9]
    assert hots.to_numpy().tolist() == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    assert all(name in st.IDEA_FEATURE_LABELS for name in names)


def test_reasons_and_invalidation_reject_unknown_strategies() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        st.reasons_for("results_play", {})
    with pytest.raises(ValueError, match="unknown strategy"):
        st.invalidation_for("results_play", {})
    # Without any number the invalidation still names the observation, never a made-up level.
    assert st.invalidation_for("breakout_20d", {}) == (
        "A close back below the breakout level: the breakout failed"
    )
    assert st.reasons_for("pullback_20dma", {}) == []


def test_column_aliases_resolve() -> None:
    df = frame(base_row(**BREAKOUT)).rename(
        columns={"brk_20d_high_pct": "dist_high20_pct", "vol_z20": "volume_z20", "rsi14": "rsi_14"}
    )
    found = st.resolve_columns(df)
    assert found["brk_20d_high_pct"] == "dist_high20_pct"
    assert found["vol_z20"] == "volume_z20" and found["rsi14"] == "rsi_14"
    assert found["rs_12_1_rank"] is None
    (c,) = st.candidates(df)
    assert c.strategy == "breakout_20d" and c.reasons[0].endswith("(Rs 494.07)")


def test_signed_up_streak_is_accepted_for_the_down_streak() -> None:
    """A frame with the signed streak (+n up closes, -n down closes) instead of the
    pipeline's `down_streak`: the rule and the words read the same number."""
    rows = frame(
        base_row("MRV", **{**MEAN_REVERSION, "down_streak": None}),
        base_row("UP", **{**MEAN_REVERSION, "down_streak": None}),
    ).drop(columns=["down_streak"])
    rows["up_streak"] = [-3.0, 2.0]
    assert st.down_streak(rows).tolist() == [3.0, 0.0]
    cands = st.candidates(rows)
    assert [c.symbol for c in cands] == ["MRV"]
    assert cands[0].reasons[1] == "3 consecutive down closes"
    cf = st.candidate_frame(rows)
    assert cf["down_streak"].tolist() == [3.0]  # the derived column travels with the row
    assert not st.candidates(frame(base_row(**{**MEAN_REVERSION, "down_streak": None})))

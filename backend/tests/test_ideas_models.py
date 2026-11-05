"""Ideas model: session-level walk-forward with an embargo, binary metrics, shrinkage
towards the base rate, drivers, registry names, API isolation.

The synthetic candidate frame has a weak but real signal (RSI, distance from the 200-DMA,
volume z, the breakout setup) so there is something to learn, and every assertion is
deterministic (fixed seeds). A pure-noise frame proves the out-of-sample shrinkage keeps
the model honest when there is nothing to learn. The registry tests need the database."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.ml import ideas_features as idf
from app.ml import ideas_models as im
from app.ml import registry
from app.ml.core import Driver, HorizonSpec, TrainResult, WalkForwardFold, horizons
from app.ml.strategies import STRATEGY_NAMES, one_hot_names
from app.providers.base import Bar
from app.services import instruments
from app.services.market_data import upsert_daily_bars

N_SESSIONS = 300
MIN_TRAIN = 120
STEP = 21
EMBARGO = 10  # swing horizon days
PER_SESSION = 10
N_SYMBOLS = 40
ROUND_TRIP = 0.494142
STOCK_FEATURES = [
    "atr14_pct",
    "rsi14",
    "dist_sma20_pct",
    "dist_sma50_pct",
    "dist_sma200_pct",
    "vol_z20",
    "rs_12_1",
    "mkt_ret_5d",
    "mkt_vix_level",
]


def settings(**over: object) -> Settings:
    return get_settings().model_copy(
        update={"wf_min_train_days": MIN_TRAIN, "wf_step_days": STEP, **over}
    )


def synthetic_candidates(
    n_sessions: int = N_SESSIONS, seed: int = 0, signal: float = 1.0
) -> pd.DataFrame:
    """PER_SESSION candidates a session from N_SYMBOLS stocks, each under one strategy; the
    swing label leans on RSI, the 200-DMA distance, volume and the breakout setup."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-01", periods=n_sessions).date
    n = n_sessions * PER_SESSION
    picks = np.concatenate(
        [rng.choice(N_SYMBOLS, size=PER_SESSION, replace=False) for _ in sessions]
    )
    strategy = rng.choice(STRATEGY_NAMES, size=n)
    df = pd.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in picks],
            "session_date": np.repeat(sessions, PER_SESSION),
            "exchange": "NSE",
            "strategy": strategy,
            "setup": "x",
            "close": rng.uniform(50, 3000, n),
            "atr14_pct": rng.uniform(1.0, 4.0, n),
            "rsi14": rng.uniform(20, 80, n),
            "dist_sma20_pct": rng.normal(0, 3, n),
            "dist_sma50_pct": rng.normal(1, 5, n),
            "dist_sma200_pct": rng.normal(3, 10, n),
            "vol_z20": rng.normal(0, 1, n),
            "rs_12_1": rng.normal(0, 15, n),
            "mkt_ret_5d": rng.normal(0, 1.5, n),
            "mkt_vix_level": rng.uniform(10, 25, n),
        }
    )
    df.loc[df.index[::17], "vol_z20"] = np.nan  # volume is missing now and then, as in life
    logit = -0.3 + signal * (
        0.04 * df["dist_sma200_pct"]
        - 0.03 * (df["rsi14"] - 50)
        + 0.4 * df["vol_z20"].fillna(0.0)
        + 0.5 * (df["strategy"] == "breakout_20d")
    )
    hit = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))
    expired = ~hit & (rng.uniform(size=n) < 0.3)
    df["label_swing"] = hit.astype(float)
    df["expired_swing"] = expired
    df["realised_swing_pct"] = np.where(expired, rng.normal(0, 2, n), np.nan)
    df["label_positional"] = (rng.uniform(size=n) < 0.4).astype(float)  # another horizon
    return df


def universe_frame(n_sessions: int = N_SESSIONS, seed: int = 1, base: float = 0.35) -> pd.DataFrame:
    """Every stock, every session, labelled with a lower hit rate than the candidates."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-01", periods=n_sessions).date
    n = n_sessions * N_SYMBOLS
    return pd.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in np.tile(np.arange(N_SYMBOLS), n_sessions)],
            "session_date": np.repeat(sessions, N_SYMBOLS),
            "label_swing": (rng.uniform(size=n) < base).astype(float),
        }
    )


LOOSE_PARAMS: dict[str, object] = {
    "num_leaves": 31,
    "min_data_in_leaf": 3,
    "lambda_l2": 0.0,
    "feature_fraction": 1.0,
    "learning_rate": 0.2,
    "n_estimators": 200,
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_candidates()


@pytest.fixture(scope="module")
def universe() -> pd.DataFrame:
    return universe_frame()


@pytest.fixture(scope="module")
def trained(frame: pd.DataFrame, universe: pd.DataFrame) -> TrainResult:
    return im.train_walk_forward(frame, "swing", settings=settings(), universe=universe)


def live_row(frame: pd.DataFrame, i: int = -1, strategy: str = "breakout_20d") -> dict:
    return im.live_features(frame.iloc[i].to_dict(), None, strategy)


def session_list(frame: pd.DataFrame) -> list[date]:
    return sorted(set(frame["session_date"]))


# ------------------------------------------------------------------ frame preparation


def test_default_params_are_a_binary_objective_with_deterministic_fits() -> None:
    p = im.DEFAULT_PARAMS
    assert p["objective"] == "binary" and "num_class" not in p
    assert p["n_estimators"] == im.MAX_ROUNDS == 400 and im.ROUND_STEP == 25
    assert (p["deterministic"], p["force_row_wise"], p["num_threads"], p["seed"]) == (
        True,
        True,
        1,
        42,
    )
    assert im.model_name("swing") == "ideas_swing" and im.model_name("positional") == (
        "ideas_positional"
    )
    assert im.model_target() == "NIFTY500"
    assert im.model_target(settings(universe="nifty50")) == "NIFTY50"
    assert (im.label_column("swing"), im.expired_column("swing"), im.realised_column("swing")) == (
        "label_swing",
        "expired_swing",
        "realised_swing_pct",
    )


def test_feature_columns_exclude_identifiers_labels_and_raw_prices(frame: pd.DataFrame) -> None:
    prepared = im.prepare_frame(frame, "swing")
    names = im.feature_columns(prepared)
    assert names == STOCK_FEATURES + one_hot_names()
    assert not any(im.is_label_column(n) for n in names)
    for banned in ("close", "symbol", "session_date", "strategy", "setup", "exchange"):
        assert banned not in names
    assert "label_positional" not in names and "realised_swing_pct" not in names
    assert len(prepared) == len(frame) and prepared["label_swing"].isin([0.0, 1.0]).all()
    assert (prepared[one_hot_names()].sum(axis=1) == 1.0).all()
    # Unlabelled rows are dropped; the positional label uses its own column.
    partial = frame.copy()
    partial.loc[partial.index[:50], "label_swing"] = np.nan
    assert len(im.prepare_frame(partial, "swing")) == len(frame) - 50
    assert len(im.prepare_frame(partial, "positional")) == len(frame)


def test_prepare_frame_rejects_missing_columns_and_unknown_horizons(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="strategy"):
        im.prepare_frame(frame.drop(columns=["strategy"]), "swing")
    with pytest.raises(ValueError, match="label_swing"):
        im.prepare_frame(frame.drop(columns=["label_swing"]), "swing")
    with pytest.raises(ValueError, match="unknown strategies"):
        im.prepare_frame(frame.assign(strategy="results_play"), "swing")
    with pytest.raises(ValueError, match="unknown horizon"):
        im.train_walk_forward(frame, "intraday", settings=settings())
    with pytest.raises(ValueError, match="labelled sessions"):
        im.train_walk_forward(synthetic_candidates(n_sessions=150), "swing", settings=settings())


def test_live_features_merge_stock_market_and_strategy() -> None:
    stock = {
        "symbol": "ABC",
        "session_date": date(2026, 10, 30),
        "close": 500.0,
        "rsi14": 27.0,
        "vol_z20": None,
        "label_swing": 1.0,
        "expired_swing": False,
        "strategy_breakout_20d": 1.0,  # stale one-hot on the row: replaced
        "strategy": "pullback_20dma",
    }
    market = {"ret_5d": 1.2, "ret_20d": float("nan"), "vix_level": 14.5, "sp500_ret_1d": 0.3}
    row = im.live_features(stock, market, "mean_reversion_rsi")
    assert row["rsi14"] == 27.0 and row["vol_z20"] is None
    assert "close" not in row and "label_swing" not in row and "symbol" not in row
    assert row["mkt_ret_5d"] == 1.2 and row["mkt_ret_20d"] is None and row["mkt_vix_level"] == 14.5
    assert "mkt_sp500_ret_1d" not in row and set(im.MARKET_CONTEXT) == {
        "ret_5d",
        "ret_20d",
        "vix_level",
        "pct_above_200dma",
        "adv_decl_ratio",
        "rvol_20d",
    }
    assert row["strategy_mean_reversion_rsi"] == 1.0 and row["strategy_breakout_20d"] == 0.0
    assert im.live_features(pd.Series(stock), None, "breakout_20d")["strategy_breakout_20d"] == 1.0


def pipeline_shaped(frame: pd.DataFrame, horizon: str = "swing") -> pd.DataFrame:
    """The synthetic frame in the shape `ideas_features.build_candidate_frame` returns:
    CANDIDATE_COLUMNS, horizon, horizon_days, META_COLUMNS, FEATURE_NAMES (the ones the
    synthetic frame has; the rest NaN), LABEL_COLUMNS with `label` / `expired` /
    `realised_return_pct`, RangeIndex."""
    spec = horizons()[horizon]
    df = frame.copy()
    df["horizon"] = horizon
    df["horizon_days"] = spec.days
    df["instrument_id"] = df["symbol"].str[1:].astype(int) + 1
    df["n_bars"] = 400
    df["eligible"] = True
    df["entry_session"] = [d + pd.Timedelta(days=1) for d in df["session_date"]]
    df["entry_price"] = df["close"] * 1.001
    df["target_pct"] = spec.target_atr * df["atr14_pct"]
    df["stop_pct"] = spec.stop_atr * df["atr14_pct"]
    df["target_price"] = df["entry_price"] * (1 + df["target_pct"] / 100)
    df["stop_price"] = df["entry_price"] * (1 - df["stop_pct"] / 100)
    df["window_closed"] = True
    df["label"] = df.pop(f"label_{horizon}")
    df["expired"] = df.pop(f"expired_{horizon}")
    realised = df.pop(f"realised_{horizon}_pct")
    df["exit_session"] = df["entry_session"]
    df["exit_price"] = np.where(
        df["label"] == 1.0,
        df["target_price"],
        np.where(df["expired"], df["entry_price"], df["stop_price"]),
    )
    df["realised_return_pct"] = np.where(
        df["label"] == 1.0, df["target_pct"], np.where(df["expired"], realised, -df["stop_pct"])
    )
    df["days_held"] = 3.0
    other = [c for c in df.columns if c.startswith("label_") and c != "label"]
    df = df.drop(columns=other)
    cols = [
        *idf.CANDIDATE_COLUMNS,
        "horizon",
        "horizon_days",
        *idf.META_COLUMNS,
        *idf.FEATURE_NAMES,
        *idf.LABEL_COLUMNS,
    ]
    return df.reindex(columns=cols).reset_index(drop=True)


def test_label_columns_resolve_both_spellings(frame: pd.DataFrame) -> None:
    assert im.label_columns(frame, "swing") == (
        "label_swing",
        "expired_swing",
        "realised_swing_pct",
    )
    assert im.label_columns(frame, "positional") == ("label_positional", None, None)
    shaped = pipeline_shaped(frame)
    assert im.label_columns(shaped, "swing") == ("label", "expired", "realised_return_pct")
    with pytest.raises(ValueError, match="neither 'label_swing' nor 'label'"):
        im.label_columns(shaped.drop(columns=["label"]), "swing")
    assert set(im.LABEL_NAMES) >= set(idf.LABEL_COLUMNS) | {"label"}
    assert set(im.NON_FEATURE_COLUMNS) >= set(idf.META_COLUMNS) | {"horizon", "horizon_days"}
    for name in (*idf.LABEL_COLUMNS, "label_swing", "expired_positional", "realised_swing_pct"):
        assert im.is_label_column(name), name
    for name in idf.FEATURE_NAMES + one_hot_names():
        assert not im.is_label_column(name) and name not in im.NON_FEATURE_COLUMNS, name


def test_pipeline_shaped_frame_trains_the_same_model(
    frame: pd.DataFrame, universe: pd.DataFrame, trained: TrainResult
) -> None:
    """`build_candidate_frame` output (label / expired / realised_return_pct, a `horizon`
    column, META_COLUMNS) is read as the per-horizon spelling is: same features, same
    labels, same artifact."""
    shaped = pipeline_shaped(frame)
    prepared = im.prepare_frame(shaped, "swing")
    names = im.feature_columns(prepared)
    # META, labels and the horizon are not features; every pipeline feature column is.
    assert not set(names) & (set(idf.META_COLUMNS) | set(idf.LABEL_COLUMNS))
    assert not set(names) & {"label", "horizon", "horizon_days", "window_closed", "eligible"}
    assert [n for n in names if n in idf.FEATURE_NAMES] == idf.FEATURE_NAMES
    assert names[-4:] == one_hot_names()
    assert len(prepared) == len(frame) and prepared["label"].isin([0.0, 1.0]).all()
    # Columns the synthetic frame never had are NaN, not filled: only the shared columns,
    # in the synthetic frame's order (LightGBM's column subsampling follows the order), can
    # make the model identical, so train both on those.
    shared = shaped.drop(columns=[c for c in idf.FEATURE_NAMES if c not in frame.columns])
    shared = shared[
        [c for c in frame.columns if c in shared.columns]
        + [c for c in shared.columns if c not in frame.columns]
    ]
    again = im.train_walk_forward(shared, "swing", settings=settings(), universe=universe)
    assert again.feature_names == trained.feature_names
    assert again.artifact == trained.artifact
    assert again.metrics.as_dict() == trained.metrics.as_dict()
    # Another horizon's rows are never borrowed.
    with pytest.raises(ValueError, match="holds horizons"):
        im.prepare_frame(shaped, "positional")
    mixed = pd.concat([shaped, shaped.assign(horizon="positional")], ignore_index=True)
    assert len(im.prepare_frame(mixed, "swing")) == len(frame)


# ------------------------------------------------------------------ walk-forward


def test_folds_are_time_ordered_with_the_horizon_as_embargo(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    sessions = session_list(frame)
    pos = {d: i for i, d in enumerate(sessions)}
    folds = trained.metrics.folds
    assert folds and all(isinstance(f, WalkForwardFold) for f in folds)
    assert pos[folds[0].test_start] == MIN_TRAIN + EMBARGO
    assert pos[folds[0].train_end] == MIN_TRAIN - 1
    for f in folds:
        assert f.train_end < f.test_start <= f.test_end
        assert pos[f.test_start] - pos[f.train_end] - 1 == EMBARGO  # sessions skipped between
        assert pos[f.test_end] - pos[f.test_start] + 1 <= STEP
        assert f.n_test > 0 and 0.0 <= f.brier <= 1.0 and f.log_loss > 0.0
    for a, b in pairwise(folds):
        assert pos[b.test_start] == pos[a.test_end] + 1  # contiguous, no gap, no overlap
        assert pos[b.train_end] == pos[a.train_end] + STEP  # the window expands by a step
    assert folds[-1].test_end == sessions[-1] == trained.trained_through
    # The same plan the feature pipeline publishes (one source of truth for the embargo).
    plan = idf.walk_forward_folds(
        sessions, min_train_sessions=MIN_TRAIN, step=STEP, embargo=EMBARGO
    )
    assert [(f.train_end, f.test_start, f.test_end) for f in folds] == [tuple(f) for f in plan]
    assert sum(f.n_test for f in folds) == trained.metrics.n_oos
    # Every out-of-sample row belongs to a session after MIN_TRAIN + EMBARGO.
    n_oos_expected = int((frame["session_date"] >= folds[0].test_start).sum())
    assert trained.metrics.n_oos == n_oos_expected
    assert trained.train_rows == len(frame)


def test_metrics_are_populated_and_serialisable(trained: TrainResult, frame: pd.DataFrame) -> None:
    m = trained.metrics
    assert isinstance(m, im.IdeaModelMetrics)
    assert (m.horizon, m.horizon_days, m.embargo_sessions, m.n_sessions) == (
        "swing",
        10,
        EMBARGO,
        N_SESSIONS,
    )
    assert 0.0 < m.brier < 0.25 and 0.0 < m.log_loss < math.log(2) and 0.0 < m.accuracy < 1.0
    assert m.n_selected > 0 and 0.0 <= m.hit_rate <= 1.0
    assert 0 < m.n_ranked <= m.n_selected and 0.0 <= m.hit_rate_ranked <= 1.0
    oos = frame[frame["session_date"] >= m.folds[0].test_start]
    all_c = m.baselines["all_candidates"]
    assert all_c["n"] == m.n_oos == len(oos)
    assert all_c["hit_rate"] == pytest.approx(oos["label_swing"].mean())
    assert 0.2 < all_c["brier"] <= 0.25  # forecasting the base rate scores ~ p (1 - p)
    rnd = m.baselines["random_universe"]
    assert rnd["n"] == m.n_oos and abs(rnd["hit_rate"] - 0.35) < 0.06
    assert set(m.per_strategy) == set(STRATEGY_NAMES)
    for entry in m.per_strategy.values():
        assert entry["n"] > 0 and 0.0 <= entry["hit_rate"] <= 1.0 and "brier" in entry
    assert sum(e["n"] for e in m.per_strategy.values()) == m.n_oos
    assert set(m.class_frequencies) == {"target", "stop", "expired"}
    assert sum(m.class_frequencies.values()) == pytest.approx(1.0)
    assert [row["confidence"] for row in m.by_confidence] == ["selected", "rejected"]
    assert sum(int(row["n"]) for row in m.by_confidence) == m.n_oos
    assert m.calibration and all(row["n"] > 0 for row in m.calibration)
    assert all(0.0 <= row["observed"] <= 1.0 for row in m.calibration)
    assert math.isfinite(m.expectancy_after_costs_pct) and math.isfinite(m.expectancy_ranked_pct)
    assert math.isfinite(m.expectancy_all_candidates_pct)
    assert m.cost_round_trip_pct == pytest.approx(ROUND_TRIP)
    assert len(m.fold_params) == len(m.folds)
    assert m.fold_params[0] == {
        "n_estimators": im.FIRST_FOLD_ROUNDS,
        "shrink": im.FIRST_FOLD_SHRINK,
    }
    assert m.notes == []
    d = m.as_dict()
    json.dumps(d)  # what the registry stores
    assert d["hit_rate"] == pytest.approx(m.hit_rate) and d["n_selected"] == m.n_selected
    assert d["folds"][0]["test_start"] == m.folds[0].test_start.isoformat()
    assert d["per_strategy"]["breakout_20d"]["n"] == m.per_strategy["breakout_20d"]["n"]
    assert trained.params["n_estimators"] in im.models.round_grid(400)
    assert 0.0 <= trained.params["shrink"] < 1.0  # type: ignore[operator]


def test_model_learns_the_synthetic_signal(trained: TrainResult) -> None:
    m = trained.metrics
    base = m.baselines["all_candidates"]
    assert m.hit_rate > base["hit_rate"] + 0.08  # the selected ideas hit more often
    assert m.hit_rate_ranked > base["hit_rate"] + 0.08
    assert m.brier < base["brier"] - 0.005  # and the probabilities beat the base rate
    assert m.expectancy_after_costs_pct > m.expectancy_all_candidates_pct
    # The planted setup effect shows up per strategy.
    assert m.per_strategy["breakout_20d"]["hit_rate"] > m.per_strategy["pullback_20dma"]["hit_rate"]


def test_training_is_deterministic(
    frame: pd.DataFrame, universe: pd.DataFrame, trained: TrainResult
) -> None:
    again = im.train_walk_forward(frame, "swing", settings=settings(), universe=universe)
    assert again.artifact == trained.artifact
    assert again.metrics.as_dict() == trained.metrics.as_dict()
    assert again.params == trained.params and again.feature_names == trained.feature_names
    assert im.models.artifact_hash(again.artifact) == im.load_artifact(trained.artifact)["hash"]


def test_a_folds_own_rows_never_pick_its_rounds_or_shrink(frame: pd.DataFrame) -> None:
    base = im.train_walk_forward(frame, "swing", settings=settings())
    folds = base.metrics.folds
    j = len(folds) // 2
    rows = (frame["session_date"] >= folds[j].test_start) & (
        frame["session_date"] <= folds[j].test_end
    )
    assert int(rows.sum()) == folds[j].n_test
    shocked = frame.copy()
    shocked.loc[rows, "label_swing"] = 1.0 - shocked.loc[rows, "label_swing"]
    again = im.train_walk_forward(shocked, "swing", settings=settings())
    bp = base.metrics.fold_params  # type: ignore[attr-defined]
    ap = again.metrics.fold_params  # type: ignore[attr-defined]
    assert ap[: j + 1] == bp[: j + 1]  # fold j chose from folds < j, which did not change
    assert again.metrics.folds[:j] == folds[:j]
    assert again.metrics.folds[j].brier != folds[j].brier  # the perturbation was real


def test_shrinkage_tames_an_overfit_model_on_noise() -> None:
    noise = synthetic_candidates(seed=3, signal=0.0)
    raw = im.train_walk_forward(
        noise, "swing", params={**LOOSE_PARAMS, "shrink": 0.0}, settings=settings()
    )
    shrunk = im.train_walk_forward(noise, "swing", params=LOOSE_PARAMS, settings=settings())
    climatology = raw.metrics.baselines["all_candidates"]["brier"]
    assert raw.metrics.brier > climatology  # overconfident: worse than the base rate
    assert shrunk.metrics.brier <= climatology + 0.003
    assert (
        shrunk.metrics.brier < raw.metrics.brier and shrunk.metrics.log_loss < raw.metrics.log_loss
    )
    assert shrunk.params["shrink"] > 0.0 and raw.params["shrink"] == 0.0  # type: ignore[operator]
    assert all(fp["shrink"] == 0.0 for fp in raw.metrics.fold_params)  # type: ignore[attr-defined]
    # With nothing to learn the live probability stays near the base rate.
    base_rate = json.loads(shrunk.artifact)["base_rate"]
    for i in range(-5, 0):
        p = im.predict(
            shrunk.artifact, shrunk.feature_names, live_row(noise, i), settings=settings()
        )
        assert abs(p.p_target - base_rate) < 0.12


def test_params_can_fix_the_shrink_and_bound_the_rounds(frame: pd.DataFrame) -> None:
    fixed = im.train_walk_forward(
        frame, "swing", params={"n_estimators": 60, "shrink": 0.3}, settings=settings()
    )
    assert fixed.params["shrink"] == 0.3 and fixed.params["n_estimators"] in (25, 50, 60)
    assert all(fp["shrink"] == 0.3 for fp in fixed.metrics.fold_params)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="shrink"):
        im.train_walk_forward(frame, "swing", params={"shrink": 1.0}, settings=settings())


def test_expectancy_uses_the_realised_return_when_the_frame_has_it(frame: pd.DataFrame) -> None:
    """With a realised return on every row (the pipeline's `realised_return_pct`) the
    expectancy is the mean of those moves less costs, not the T / -S approximation."""
    shaped = pipeline_shaped(frame)
    shared = shaped.drop(columns=[c for c in idf.FEATURE_NAMES if c not in frame.columns])
    shared = shared[
        [c for c in frame.columns if c in shared.columns]
        + [c for c in shared.columns if c not in frame.columns]
    ]
    bumped = shared.copy()
    bumped["realised_return_pct"] = bumped["realised_return_pct"] + 1.0  # every exit 1 pt better
    a = im.train_walk_forward(shared, "swing", settings=settings()).metrics
    b = im.train_walk_forward(bumped, "swing", settings=settings()).metrics
    assert isinstance(a, im.IdeaModelMetrics) and isinstance(b, im.IdeaModelMetrics)
    assert b.expectancy_all_candidates_pct == pytest.approx(a.expectancy_all_candidates_pct + 1.0)
    assert b.expectancy_after_costs_pct == pytest.approx(a.expectancy_after_costs_pct + 1.0)
    assert b.hit_rate == a.hit_rate and b.brier == a.brier  # the probabilities did not move


def test_expectancy_is_nan_without_atr_and_the_note_says_so(frame: pd.DataFrame) -> None:
    result = im.train_walk_forward(
        frame.drop(columns=["atr14_pct", "expired_swing", "realised_swing_pct"]),
        "swing",
        settings=settings(),
    )
    m = result.metrics
    assert isinstance(m, im.IdeaModelMetrics)
    assert math.isnan(m.expectancy_after_costs_pct) and math.isnan(m.expectancy_all_candidates_pct)
    assert m.n_selected > 0 and not math.isnan(m.hit_rate)  # the p bar still applies
    assert any("atr14_pct" in n for n in m.notes) and any("expired" in n for n in m.notes)
    assert set(m.class_frequencies) == {"target", "not_target"}
    assert m.baselines["random_universe"] == {"n": 0}  # no universe given: no number
    assert m.as_dict()["expectancy_after_costs_pct"] is None


def test_random_universe_baseline_is_seeded_and_sized_like_the_candidates(
    frame: pd.DataFrame, universe: pd.DataFrame
) -> None:
    counts = {d: 3 for d in session_list(frame)[:50]}
    a = im._random_universe_baseline(universe, "swing", counts)
    b = im._random_universe_baseline(universe, "swing", counts)
    assert a == b and a["n"] == 150 and 0.0 <= a["hit_rate"] <= 1.0
    small = universe[universe["symbol"].isin(["S000", "S001"])]
    assert im._random_universe_baseline(small, "swing", counts)["n"] == 100  # 2 a session
    assert im._random_universe_baseline(None, "swing", counts) == {"n": 0}
    with pytest.raises(ValueError, match="label_swing"):
        im._random_universe_baseline(universe.drop(columns=["label_swing"]), "swing", counts)


# ------------------------------------------------------------------ inference


def test_predict_returns_a_probability_and_ranked_drivers(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    s = settings()
    row = live_row(frame)
    out = im.predict(trained.artifact, trained.feature_names, row, settings=s)
    assert isinstance(out, im.IdeaPrediction)
    p, drivers, all_drivers = out  # a NamedTuple unpacks like the contract's 3-tuple
    assert 0.0 < p < 1.0 and out.p_target == p
    assert len(drivers) == min(s.drivers_top_n, len(all_drivers)) == 8
    assert drivers == all_drivers[:8]
    assert len(all_drivers) <= len(trained.feature_names)
    mags = [abs(d.contribution) for d in all_drivers]
    assert mags == sorted(mags, reverse=True) and all(c != 0.0 for c in mags)
    assert all(
        isinstance(d, Driver) and d.text.endswith("the odds of target before stop") for d in drivers
    )
    assert {d.feature for d in all_drivers} <= set(trained.feature_names)
    setup = next(d for d in all_drivers if d.feature == "strategy_breakout_20d")
    assert setup.value == 1.0 and setup.text.startswith("Setup: 20-day breakout = yes")
    fewer = im.predict(
        trained.artifact, trained.feature_names, row, settings=settings(drivers_top_n=3)
    )
    assert len(fewer.drivers) == 3 and fewer.all_drivers == all_drivers
    with pytest.raises(ValueError, match="feature names"):
        im.predict(trained.artifact, trained.feature_names[:-1], row, settings=s)


def test_predict_applies_the_artifacts_shrinkage(frame: pd.DataFrame, trained: TrainResult) -> None:
    art = im.load_artifact(trained.artifact)
    assert art["horizon"] == "swing" and art["shrink"] == trained.params["shrink"]
    assert art["base_rate"] == pytest.approx(frame["label_swing"].mean())
    assert art["feature_names"] == trained.feature_names
    row = live_row(frame, -2, "mean_reversion_rsi")
    x = im.models.row_vector(trained.feature_names, row)
    raw = float(np.clip(art["classifier"].predict(x)[0], im.PROB_FLOOR, 1 - im.PROB_FLOOR))
    expected = (1 - art["shrink"]) * raw + art["shrink"] * art["base_rate"]
    out = im.predict(trained.artifact, trained.feature_names, row, settings=settings())
    assert out.p_target == pytest.approx(expected)
    assert im.load_artifact(trained.artifact) is art  # cached by hash


def test_missing_values_stay_missing_and_still_predict(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    row = live_row(frame)
    row["rsi14"] = None
    row["mkt_vix_level"] = float("nan")
    out = im.predict(trained.artifact, trained.feature_names, row, settings=settings())
    assert 0.0 < out.p_target < 1.0
    rsi = [d for d in out.all_drivers if d.feature == "rsi14"]
    assert all(d.value is None and "(no data)" in d.text for d in rsi)


def test_model_learns_the_direction_of_the_signal(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    row = live_row(frame)
    good = {**row, "rsi14": 25.0, "dist_sma200_pct": 20.0, "vol_z20": 2.0}
    bad = {**row, "rsi14": 75.0, "dist_sma200_pct": -15.0, "vol_z20": -2.0}
    p_good = im.predict(trained.artifact, trained.feature_names, good, settings=settings()).p_target
    p_bad = im.predict(trained.artifact, trained.feature_names, bad, settings=settings()).p_target
    assert p_good > p_bad + 0.1


# ------------------------------------------------------------------ words


def test_driver_text_and_labels() -> None:
    # The feature pipeline's vocabulary first, then the columns the strategies add.
    assert im.feature_label("strategy_momentum_12_1") == "Setup: 12-1 month momentum"
    assert im.feature_label("mkt_vix_level") == idf.FEATURE_LABELS["mkt_vix_level"]
    assert im.feature_label("mkt_vix_level") == "India VIX level, previous session"
    assert im.feature_label("rsi14") == "RSI (14)"
    assert im.feature_label("rs_12_1_rank") == "Relative-strength percentile in the universe"
    # An unknown mkt_* column falls back to the market-call vocabulary, prefixed.
    assert im.feature_label("mkt_fii_net_cr") == f"Nifty 50 {im.models.feature_label('fii_net_cr')}"
    assert im.feature_label("mkt_fii_net_cr").startswith("Nifty 50 FII net")
    assert im.feature_label("some_new_thing") == "some new thing"
    for name in idf.FEATURE_NAMES + one_hot_names():
        assert im.feature_label(name) and im.feature_label(name) != name
    assert im.value_phrase("strategy_breakout_20d", 1.0) == "= yes"
    assert im.value_phrase("mkt_ret_5d", -1.234) == "at -1.23%"
    assert im.value_phrase("rs_12_1", 12.34) == "at +12.3 pp"
    assert im.value_phrase("rs_12_1_rank", 0.95) == "at the 95th percentile"
    assert im.value_phrase("turnover_20d_cr", 1234.5) == "at Rs 1,234 cr"
    assert im.value_phrase("days_since_high250", 3.0) == "= 3 sessions"
    assert im.value_phrase("down_streak", 1.0) == "= 1 session"
    assert im.value_phrase("up_streak", -3.0) == "= 3 down sessions"
    assert im.value_phrase("up_streak", 1.0) == "= 1 up session"
    assert im.value_phrase("up_streak", 0.0) == "= 0 sessions"
    assert im.value_phrase("rel_vol_20d", 1.87) == "at 1.87x"
    assert im.value_phrase("rsi14", None) == "(no data)"
    assert (
        im.driver_text("rsi14", 27.0, 0.3) == "RSI (14) at 27 raises the odds of target before stop"
    )
    assert im.driver_text("mkt_ret_5d", -2.0, -0.2) == (
        "Nifty 50 5-day return, previous session at -2.00% lowers the odds of target before stop"
    )
    assert im.driver_text("dist_sma200_pct", 8.0, 0.1) == (
        "Distance from 200-DMA at +8.00% raises the odds of target before stop"
    )
    assert im.driver_text("rs_60d", -4.0, -0.1) == (
        "Relative strength vs Nifty 50, 60 days at -4.0 pp lowers the odds of target before stop"
    )
    assert im.driver_text("rel_vol_20d", 2.5, 0.2) == (
        "Volume vs 20-session average at 2.50x raises the odds of target before stop"
    )


def test_drivers_skip_features_that_did_nothing() -> None:
    names = ["a", "b", "c", "d"]
    row = {"a": 1.0, "b": None, "c": 3.0, "d": 4.0}
    out = im.drivers(names, row, np.array([0.1, -0.5, 0.0, float("nan")]))
    assert [d.feature for d in out] == ["b", "a"] and out[0].value is None
    assert [d.feature for d in im.drivers(names, row, np.array([0.1, -0.5, 0.2, 0.0]), n=2)] == [
        "b",
        "c",
    ]


# ------------------------------------------------------------------ binary scores


def test_binary_metric_helpers() -> None:
    y = np.array([1, 0, 1, 0])
    assert im.brier_binary(np.array([1.0, 0.0, 1.0, 0.0]), y) == 0.0
    assert im.brier_binary(np.full(4, 0.5), y) == 0.25
    assert im.log_loss_binary(np.full(4, 0.5), y) == pytest.approx(math.log(2))
    assert im.log_loss_binary(np.array([1.0, 0.0, 1.0, 0.0]), y) == pytest.approx(1e-6, abs=1e-6)
    assert im.accuracy_binary(np.array([0.9, 0.4, 0.5, 0.6]), y) == 0.75
    assert im.hit_rate(y) == 0.5 and math.isnan(im.hit_rate(np.array([])))
    assert math.isnan(im.brier_binary(np.array([]), np.array([])))
    bins = im.calibration_bins_binary(np.array([0.1, 0.15, 0.9, 1.0]), np.array([0, 1, 1, 1]))
    assert [(b["bin_lo"], b["bin_hi"], b["n"]) for b in bins] == [(0.0, 0.2, 2), (0.8, 1.0, 2)]
    assert bins[0]["observed"] == 0.5 and bins[0]["predicted_mean"] == pytest.approx(0.125)
    assert im.calibration_bins_binary(np.array([]), np.array([])) == []


def test_shrink_binary_math() -> None:
    p = np.array([0.9, 0.1])
    np.testing.assert_allclose(im.shrink_binary(p, 0.5, 0.4), [0.65, 0.25])
    np.testing.assert_array_equal(im.shrink_binary(p, 0.0, None), p)
    stacked = im.shrink_binary(np.stack([p, p]), 0.5, 0.4)
    assert stacked.shape == (2, 2) and np.allclose(stacked[1], [0.65, 0.25])
    assert im.shrink_binary(np.array([0.0, 1.0]), 0.0, None).tolist() == [
        im.PROB_FLOOR,
        1 - im.PROB_FLOOR,
    ]
    with pytest.raises(ValueError, match="base rate"):
        im.shrink_binary(p, 0.5, None)
    with pytest.raises(ValueError, match="base rate"):
        im.shrink_binary(p, 0.5, float("nan"))
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="shrink"):
            im.shrink_binary(p, bad, 0.4)


# ------------------------------------------------------------------ registry


def test_registry_names_and_champion(db: Session, trained: TrainResult) -> None:
    assert im.ideas_champion(db, "swing") is None
    mv = im.save_ideas_model(db, trained, promote=True, note="first ideas model")
    db.commit()
    assert (mv.name, mv.target, mv.kind) == ("ideas_swing", "NIFTY500", "lightgbm")
    assert mv.is_champion and mv.promotion_note == "first ideas model"
    assert mv.feature_names == trained.feature_names
    assert mv.metrics["hit_rate"] == pytest.approx(trained.metrics.hit_rate)  # type: ignore[attr-defined]
    assert mv.metrics["horizon"] == "swing" and mv.metrics["brier"] == pytest.approx(
        trained.metrics.brier
    )
    assert mv.trained_through == trained.trained_through and mv.train_rows == trained.train_rows
    champ = im.ideas_champion(db, "swing")
    assert champ is not None and champ.id == mv.id
    assert im.ideas_champion(db, "positional") is None
    assert registry.champion(db, "NIFTY 50") is None  # the market-call registry is untouched
    light = im.ideas_champion_light(db, "swing")
    assert light is not None and light.id == mv.id
    assert im.result_horizon(trained) == "swing"
    # should_promote reads the same metrics layout (brier, accuracy, folds).
    ok, why = registry.should_promote(None, trained)
    assert ok and "no champion" in why
    ok, why = registry.should_promote(champ, trained)
    assert not ok and "not more training rows" in why  # same rows, same recipe: nothing new
    assert ok is False and isinstance(why, str)


# ------------------------------------------------------------------ training data (db)


def _random_walk_bars(rng: np.random.Generator, days: list[date], start: float) -> list[Bar]:
    close = start * np.exp(np.cumsum(rng.normal(0.0006, 0.015, len(days))))
    out: list[Bar] = []
    for i, (d, c) in enumerate(zip(days, close, strict=True)):
        prev = close[i - 1] if i else c
        o = prev * (1 + rng.normal(0, 0.004))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.007)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.007)))
        out.append(Bar(d, float(o), float(hi), float(lo), float(c), float(rng.uniform(3e5, 9e5))))
    return out


@pytest.fixture()
def seeded(db: Session) -> tuple[Session, list[date], int]:
    """Nifty 50 and 16 universe stocks with ~450 weekday sessions of random-walk bars (no
    holidays stored, so every weekday is a session); the last bar date is returned."""
    rng = np.random.default_rng(11)
    instruments.ensure_reference_instruments(db)
    db.flush()
    days = list(pd.bdate_range("2023-03-01", periods=450).date)
    nifty = instruments.get_instrument(db, "NIFTY 50", "INDEX")
    assert nifty is not None
    upsert_daily_bars(db, nifty.id, _random_walk_bars(rng, days, 18_000.0), "yfinance")
    for i in range(16):
        eq = instruments.get_or_create_equity(db, f"TST{i:02d}", "NSE")
        eq.in_universe = True
        upsert_daily_bars(
            db, eq.id, _random_walk_bars(rng, days, float(rng.uniform(300, 2500))), "yfinance"
        )
    db.flush()
    return db, days, 16


def test_training_data_from_the_database(seeded: tuple[Session, list[date], int]) -> None:
    db, days, n_stocks = seeded
    s = settings(wf_min_train_days=40, wf_step_days=10)
    spec: HorizonSpec = horizons(s)["swing"]
    td = im.training_data(db, "swing", settings=s)
    assert isinstance(td, im.TrainingData)
    assert td.horizon == "swing" and td.through == days[-1] and td.n_stocks == n_stocks
    frame = td.frame
    expected_cols = [
        *idf.CANDIDATE_COLUMNS,
        "horizon",
        "horizon_days",
        *idf.META_COLUMNS,
        *idf.FEATURE_NAMES,
        *idf.LABEL_COLUMNS,
        "rs_12_1_rank",
    ]
    assert list(frame.columns) == expected_cols
    assert td.n_candidates > 0 and td.n_labelled > 0
    assert set(frame["strategy"]) <= set(STRATEGY_NAMES) and (frame["horizon"] == "swing").all()
    assert frame["eligible"].all()  # ineligible rows never became candidates
    assert frame["rs_12_1_rank"].notna().all() and (frame["rs_12_1_rank"] <= 1.0).all()
    mom = frame[frame["strategy"] == "momentum_12_1"]
    assert not mom.empty and (mom["rs_12_1_rank"] >= 0.9).all()
    # Purging: the last `days` sessions are unlabelled, everything before is labelled.
    last_labelled = max(days[: len(days) - spec.days])
    assert frame.loc[frame["session_date"] > last_labelled, "label"].isna().all()
    assert frame.loc[frame["session_date"] <= last_labelled, "label"].notna().all()
    assert frame["label"].dropna().isin([0.0, 1.0]).all()
    # Every candidate is one of the labelled universe rows, with the same label.
    uni = td.universe
    assert list(uni.columns) == list(im.UNIVERSE_LABEL_COLUMNS)
    assert len(uni) > len(frame) and uni["instrument_id"].nunique() == n_stocks
    key = ["symbol", "session_date"]
    joined = frame.merge(uni, on=key, how="left", suffixes=("", "_u"))
    assert joined["label_u"].notna().sum() == td.n_labelled
    same = joined["label"].notna()
    assert (joined.loc[same, "label"] == joined.loc[same, "label_u"]).all()
    assert (
        joined.loc[same, "realised_return_pct"] == joined.loc[same, "realised_return_pct_u"]
    ).all()
    # ... and the frame trains, with the universe baseline sized like the candidates.
    result = im.train_walk_forward(frame, "swing", settings=s, universe=uni)
    m = result.metrics
    assert isinstance(m, im.IdeaModelMetrics) and m.n_oos > 0 and m.notes == []
    assert m.baselines["random_universe"]["n"] == m.n_oos
    assert 0.0 <= m.baselines["random_universe"]["hit_rate"] <= 1.0
    assert result.feature_names == idf.FEATURE_NAMES + ["rs_12_1_rank"] + one_hot_names()
    assert result.trained_through <= last_labelled
    # The live path: the same strategies on the same features give rows predict() accepts.
    from app.ml import strategies as st

    feats = idf.stock_features(
        db, [int(i) for i in uni["instrument_id"].unique()], days[-1], settings=s
    )
    live = st.candidates(feats, session=days[-1])
    assert live, "no candidate on the last session of the seeded world"
    cf = st.candidate_frame(feats, session=days[-1])
    row = im.live_features(cf.iloc[0].to_dict(), None, str(cf.iloc[0]["strategy"]))
    assert set(result.feature_names) <= set(row)
    p = im.predict(result.artifact, result.feature_names, row, settings=s)
    assert 0.0 < p.p_target < 1.0
    # Nothing after `through` is read: cutting the world there changes nothing.
    earlier = days[-30]
    td2 = im.training_data(db, "swing", through=earlier, settings=s)
    assert td2.through == earlier and td2.frame["session_date"].max() <= earlier
    head = frame[frame["session_date"] <= earlier - pd.Timedelta(days=0)]
    labelled_before = head[head["session_date"] <= max(days[: days.index(earlier) + 1 - spec.days])]
    merged = labelled_before.merge(
        td2.frame, on=["symbol", "session_date", "strategy"], suffixes=("", "_2")
    )
    assert len(merged) == len(labelled_before)
    assert (merged["label"] == merged["label_2"]).all()
    assert np.allclose(merged["rsi14"], merged["rsi14_2"], equal_nan=True)


def test_training_data_needs_a_universe(db: Session) -> None:
    with pytest.raises(ValueError, match="universe"):
        im.training_data(db, "swing")


# ------------------------------------------------------------------ API isolation


def test_ideas_modules_keep_lightgbm_out_of_the_api() -> None:
    code = (
        "import sys\n"
        "import app.main, app.ml.strategies, app.ml.costs, app.ml.ranking, app.ml.ideas_models\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'lightgbm', 'sklearn'})\n"
        "print(heavy)\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "[]"

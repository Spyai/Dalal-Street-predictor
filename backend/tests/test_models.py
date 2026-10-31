"""Market-call model: walk-forward training, prediction, baselines, metrics, API isolation.

No database and no network. The synthetic frame has a weak but real signal, so the
model has something to learn and every assertion is deterministic (fixed seeds). The
pure-noise frames prove the two out-of-sample choices (boosting rounds, shrinkage towards
climatology) protect against an overconfident model, and that a fold's own rows never
take part in its own choice.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml import baselines, metrics, models
from app.ml.core import (
    BASELINE_NAMES,
    CLASS_INDEX,
    CLASSES,
    CONFIDENCE_LEVELS,
    TrainResult,
    class_of,
    confidence_label,
)

N_ROWS = 720
MIN_TRAIN = 500
STEP = 21
FEATURES = [
    "ret_1d",
    "ret_5d",
    "rsi_14",
    "es_fut_ret_1d",
    "sp500_ret_1d",
    "vix_level",
    "fii_net_cr",
    "is_month_last_week",
    "dow",
]


def synthetic_frame(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """One row per weekday session; the label leans on ES futures, yesterday's return and
    VIX, plus noise. The last row is a live row (no label), like `features.build_frame`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n + 1).date
    df = pd.DataFrame(
        {
            "ret_1d": rng.normal(0, 1, n + 1),
            "ret_5d": rng.normal(0, 2, n + 1),
            "rsi_14": rng.uniform(25, 75, n + 1),
            "es_fut_ret_1d": rng.normal(0, 1, n + 1),
            "sp500_ret_1d": rng.normal(0, 1, n + 1),
            "vix_level": rng.uniform(10, 25, n + 1),
            "fii_net_cr": rng.normal(0, 2000, n + 1),
            "is_month_last_week": (rng.uniform(size=n + 1) < 0.2).astype(float),
            "dow": rng.integers(0, 5, n + 1).astype(float),
        },
        index=idx,
    )
    df.loc[df.index[::15], "fii_net_cr"] = np.nan  # some flows are missing, as in life
    signal = 0.35 * df["es_fut_ret_1d"] + 0.15 * df["ret_1d"] - 0.02 * (df["vix_level"] - 17)
    label = signal + rng.normal(0, 0.8, n + 1)
    df["label_pct"] = label
    df["label_class"] = [class_of(float(v), 0.30) for v in label]
    df["reference_date"] = [d - pd.Timedelta(days=1) for d in idx]
    df["reference_close"] = 20000.0
    live = df.index[-1]
    df.loc[live, "label_pct"] = np.nan
    df.loc[live, "label_class"] = None
    df.attrs["target"] = "NIFTY 50"
    return df


def live_row(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, float | None]:
    last = frame.iloc[-1]
    return {f: (None if pd.isna(last[f]) else float(last[f])) for f in feature_names}


def noise_frame(n: int = N_ROWS, seed: int = 1, n_features: int = 20) -> pd.DataFrame:
    """Pure noise: independent features and labels. Nothing here can be learnt."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n + 1).date
    df = pd.DataFrame({f"f{i}": rng.normal(size=n + 1) for i in range(n_features)}, index=idx)
    label = rng.normal(0, 1, n + 1)
    df["label_pct"] = label
    df["label_class"] = [class_of(float(v), 0.30) for v in label]
    df["reference_date"] = [d - pd.Timedelta(days=1) for d in idx]
    df["reference_close"] = 20000.0
    df.loc[df.index[-1], "label_pct"] = np.nan
    df.loc[df.index[-1], "label_class"] = None
    df.attrs["target"] = "NIFTY 50"
    return df


# Deliberately overfit-prone LightGBM settings (deep trees, tiny leaves, no L2, every feature,
# a fast learning rate): what shrinkage has to protect against.
LOOSE_PARAMS: dict[str, object] = {
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "lambda_l2": 0.0,
    "feature_fraction": 1.0,
    "learning_rate": 0.1,
    "n_estimators": 200,
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def trained(frame: pd.DataFrame) -> TrainResult:
    return models.train_walk_forward(frame, min_train_days=MIN_TRAIN, step_days=STEP)


# ------------------------------------------------------------------ walk-forward training


def test_default_params_follow_the_contract() -> None:
    p = models.DEFAULT_PARAMS
    assert p["objective"] == "multiclass" and p["num_class"] == 3
    assert (p["num_leaves"], p["min_data_in_leaf"], p["learning_rate"]) == (5, 60, 0.03)
    assert p["n_estimators"] == models.MAX_ROUNDS == 400  # the ceiling of the round search
    assert (p["feature_fraction"], p["bagging_fraction"], p["bagging_freq"]) == (0.6, 0.8, 1)
    assert (p["lambda_l2"], p["seed"], p["verbose"]) == (5.0, 42, -1)
    assert (p["deterministic"], p["force_row_wise"], p["num_threads"]) == (True, True, 1)
    assert models.round_grid(400) == tuple(range(25, 401, 25))
    assert models.round_grid(60) == (25, 50, 60) and models.round_grid(10) == (10,)
    assert models.SHRINK_GRID == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert models.FIRST_FOLD_ROUNDS == 50 and models.FIRST_FOLD_SHRINK == 0.5
    # The walk-forward's own keys never reach LightGBM.
    assert not set(models._booster_params(p)) & {"n_estimators", "shrink"}


def test_walk_forward_is_strictly_expanding_and_ordered(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    m = trained.metrics
    labelled_dates = [d for d, v in frame["label_pct"].items() if not pd.isna(v)]
    assert trained.train_rows == len(labelled_dates) == N_ROWS
    assert trained.trained_through == labelled_dates[-1]
    assert trained.feature_names == FEATURES
    assert m.n_oos == N_ROWS - MIN_TRAIN == sum(f.n_test for f in m.folds)
    assert len(m.folds) == math.ceil((N_ROWS - MIN_TRAIN) / STEP)
    for i, fold in enumerate(m.folds):
        assert fold.train_end < fold.test_start <= fold.test_end  # no test row <= train_end
        assert fold.train_end == labelled_dates[MIN_TRAIN + i * STEP - 1]
        assert fold.test_start == labelled_dates[MIN_TRAIN + i * STEP]
        if i:
            assert m.folds[i - 1].test_end < fold.test_start  # folds never overlap
            assert m.folds[i - 1].train_end < fold.train_end  # window only grows
        assert 0.0 <= fold.accuracy <= 1.0 and 0.0 <= fold.brier <= 2.0 and fold.log_loss > 0


def test_walk_forward_metrics_are_populated(trained: TrainResult) -> None:
    m = trained.metrics
    assert 0.0 <= m.accuracy <= 1.0 and 0.0 <= m.brier <= 2.0 and m.log_loss > 0.0
    assert set(m.class_frequencies) == set(CLASSES)
    assert math.isclose(sum(m.class_frequencies.values()), 1.0)
    assert set(m.baselines) == set(BASELINE_NAMES)
    for entry in m.baselines.values():
        assert entry["n"] == m.n_oos
        assert 0.0 <= entry["accuracy"] <= 1.0 and 0.0 <= entry["brier"] <= 2.0
    assert [r["confidence"] for r in m.by_confidence] == list(CONFIDENCE_LEVELS)
    assert sum(int(r["n"]) for r in m.by_confidence) == m.n_oos
    assert m.calibration and sum(int(b["n"]) for b in m.calibration) == m.n_oos
    assert all(b["bin_lo"] < b["bin_hi"] for b in m.calibration)
    assert isinstance(m, models.MarketCallMetrics)
    assert 0.0 <= m.range_coverage <= 1.0
    assert len(m.fold_params) == len(m.folds)
    rounds_seen = set(models.round_grid(models.MAX_ROUNDS)) | {models.FIRST_FOLD_ROUNDS}
    shrinks_seen = set(models.SHRINK_GRID) | {models.FIRST_FOLD_SHRINK}
    for fp in m.fold_params:
        assert set(fp) == {"n_estimators", "shrink"}
        assert fp["n_estimators"] in rounds_seen and fp["shrink"] in shrinks_seen
    d = m.as_dict()
    assert d["range_coverage"] == pytest.approx(m.range_coverage)  # finite(): 10 decimals
    assert d["fold_params"] == m.fold_params
    # The dict form must be JSON for the model_version.metrics column: no NaN anywhere.
    json.dumps(d, allow_nan=False)


def test_model_learns_the_synthetic_signal(trained: TrainResult) -> None:
    m = trained.metrics
    assert m.brier < 2.0 / 3.0  # better than the uniform forecast
    assert m.brier < m.baselines["random"]["brier"]  # and than climatology
    assert m.accuracy > m.baselines["always_up"]["accuracy"]
    assert m.log_loss < math.log(3)
    assert isinstance(m, models.MarketCallMetrics)
    assert 0.5 <= m.range_coverage <= 0.95  # a nominal 80 % range, from per-fold quantile fits


def test_too_few_labelled_rows_raises() -> None:
    small = synthetic_frame(n=MIN_TRAIN + STEP - 2)
    with pytest.raises(ValueError, match="labelled sessions"):
        models.train_walk_forward(small, min_train_days=MIN_TRAIN, step_days=STEP)


def test_training_is_deterministic(frame: pd.DataFrame, trained: TrainResult) -> None:
    again = models.train_walk_forward(frame, min_train_days=MIN_TRAIN, step_days=STEP)
    assert models.artifact_hash(again.artifact) == models.artifact_hash(trained.artifact)
    assert again.metrics.as_dict() == trained.metrics.as_dict()
    row = live_row(frame, trained.feature_names)
    assert models.predict(again.artifact, again.feature_names, row) == models.predict(
        trained.artifact, trained.feature_names, row
    )


def test_artifact_layout_and_cache(trained: TrainResult) -> None:
    payload = json.loads(trained.artifact)
    keys = {"classifier", "q10", "q90", "feature_names", "params", "class_frequencies"}
    assert keys <= set(payload)
    assert payload["feature_names"] == FEATURES
    assert payload["params"] == trained.params
    assert payload["class_frequencies"] == trained.metrics.class_frequencies
    # The chosen pair is recorded and the final models were fitted with those rounds.
    chosen = payload["params"]["n_estimators"]
    assert chosen in models.round_grid(models.MAX_ROUNDS)
    assert payload["params"]["shrink"] in models.SHRINK_GRID
    first = models.load_artifact(trained.artifact)
    assert models.load_artifact(trained.artifact) is first  # cached by hash
    assert first["classifier"].feature_name() == FEATURES
    assert first["classifier"].current_iteration() == chosen
    assert first["q10"].current_iteration() == first["q90"].current_iteration() == chosen
    assert first["shrink"] == payload["params"]["shrink"]
    np.testing.assert_allclose(first["climatology"].sum(), 1.0)
    assert list(first["climatology"]) == [payload["class_frequencies"][c] for c in CLASSES]


# ------------------------------------------------------------------ out-of-sample choices


def test_first_fold_uses_the_conservative_defaults(trained: TrainResult) -> None:
    fp = trained.metrics.fold_params  # type: ignore[attr-defined]
    assert fp[0] == {"n_estimators": models.FIRST_FOLD_ROUNDS, "shrink": models.FIRST_FOLD_SHRINK}


def test_selection_pool_holds_only_earlier_folds(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When fold j's pair is chosen, exactly the rows of folds < j have been scored."""
    seen: list[tuple[int, float]] = []
    real = models.select_rounds_and_shrink

    def spy(loss_sum: np.ndarray, n_scored: int, *args: object) -> tuple[int, float]:
        seen.append((n_scored, float(loss_sum.sum())))
        return real(loss_sum, n_scored, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(models, "select_rounds_and_shrink", spy)
    result = models.train_walk_forward(frame, min_train_days=MIN_TRAIN, step_days=STEP)
    folds = result.metrics.folds
    assert len(seen) == len(folds) + 1  # one choice per fold, then the final fit's
    pooled = np.cumsum([0, *(f.n_test for f in folds)])
    assert [n for n, _ in seen] == list(pooled)
    assert seen[0] == (0, 0.0)  # nothing to choose from: the defaults
    assert all(b > a for (_, a), (_, b) in pairwise(seen))  # the pool only ever grows


def test_a_folds_own_rows_never_pick_its_rounds_or_shrink(frame: pd.DataFrame) -> None:
    """Rewriting the labels of one fold's test rows changes that fold's scores but not its
    (rounds, shrink), nor anything about the folds before it."""
    base = models.train_walk_forward(frame, min_train_days=MIN_TRAIN, step_days=STEP)
    folds = base.metrics.folds
    j = len(folds) // 2
    rows = [d for d in frame.index if folds[j].test_start <= d <= folds[j].test_end]
    assert len(rows) == folds[j].n_test
    shocked = frame.copy()
    flipped = -3.0 * shocked.loc[rows, "label_pct"].to_numpy(dtype=float)
    shocked.loc[rows, "label_pct"] = flipped
    shocked.loc[rows, "label_class"] = [class_of(float(v), 0.30) for v in flipped]
    again = models.train_walk_forward(shocked, min_train_days=MIN_TRAIN, step_days=STEP)

    bp, ap = base.metrics.fold_params, again.metrics.fold_params  # type: ignore[attr-defined]
    assert ap[: j + 1] == bp[: j + 1]  # fold j chose from folds < j, which did not change
    assert again.metrics.folds[:j] == folds[:j]  # earlier folds never saw the change
    assert again.metrics.folds[j].test_start == folds[j].test_start
    assert again.metrics.folds[j].brier != folds[j].brier  # the perturbation was real


def test_select_rounds_and_shrink_prefers_the_conservative_tie() -> None:
    rounds, shrinks = (25, 50, 75), (0.0, 0.5, 0.9)
    assert models.select_rounds_and_shrink(np.zeros((3, 3)), 0, rounds, shrinks, (50, 0.5)) == (
        50,
        0.5,
    )
    loss = np.full((3, 3), 10.0)
    loss[1, 0] = 4.0
    assert models.select_rounds_and_shrink(loss, 2, rounds, shrinks, (50, 0.5)) == (50, 0.0)
    loss[2, 2] = 4.0  # a tie: fewer rounds win ...
    assert models.select_rounds_and_shrink(loss, 2, rounds, shrinks, (50, 0.5)) == (50, 0.0)
    loss[1, 2] = 4.0  # ... and at equal rounds, more shrinkage
    assert models.select_rounds_and_shrink(loss, 2, rounds, shrinks, (50, 0.5)) == (50, 0.9)


def test_shrink_probs_math() -> None:
    p = np.array([[0.9, 0.05, 0.05]])
    clim = np.array([0.3, 0.3, 0.4])
    np.testing.assert_allclose(models.shrink_probs(p, 0.5, clim), [[0.6, 0.175, 0.225]])
    np.testing.assert_array_equal(models.shrink_probs(p, 0.0, None), p)
    stacked = models.shrink_probs(np.stack([p, p]), 0.5, clim)  # (K, n, 3) broadcasts too
    assert stacked.shape == (2, 1, 3) and np.allclose(stacked[1], [[0.6, 0.175, 0.225]])
    with pytest.raises(ValueError, match="class frequencies"):
        models.shrink_probs(p, 0.5, None)
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="shrink"):
            models.shrink_probs(p, bad, clim)
    # Overconfident and wrong: shrinking towards climatology repairs the log loss.
    y = np.array([CLASS_INDEX["UP"]])
    assert metrics.log_loss(models.shrink_probs(p, 0.9, clim), y) < metrics.log_loss(p, y)


def test_shrinkage_tames_an_overfit_model_on_noise() -> None:
    noise = noise_frame()
    raw = models.train_walk_forward(
        noise, params={**LOOSE_PARAMS, "shrink": 0.0}, min_train_days=MIN_TRAIN, step_days=STEP
    )
    shrunk = models.train_walk_forward(
        noise, params=LOOSE_PARAMS, min_train_days=MIN_TRAIN, step_days=STEP
    )
    assert raw.metrics.log_loss > math.log(3)  # overconfident: worse than saying "no idea"
    assert shrunk.metrics.log_loss <= math.log(3) + 0.01
    assert shrunk.metrics.log_loss < raw.metrics.log_loss
    assert shrunk.metrics.brier < raw.metrics.brier
    assert shrunk.params["shrink"] > 0.0 and raw.params["shrink"] == 0.0
    assert all(fp["shrink"] == 0.0 for fp in raw.metrics.fold_params)  # type: ignore[attr-defined]
    # With nothing to learn the live call stays near the class frequencies.
    out = models.predict(
        shrunk.artifact, shrunk.feature_names, live_row(noise, shrunk.feature_names)
    )
    freq = shrunk.metrics.class_frequencies
    assert all(abs(out.probs[c] - freq[c]) < 0.1 for c in CLASSES)


def test_params_overrides_bound_rounds_and_fix_shrink(frame: pd.DataFrame) -> None:
    small = synthetic_frame(n=MIN_TRAIN + 3 * STEP)
    result = models.train_walk_forward(
        small, params={"n_estimators": 60, "shrink": 0.2}, min_train_days=MIN_TRAIN, step_days=STEP
    )
    fps = result.metrics.fold_params  # type: ignore[attr-defined]
    assert [fp["shrink"] for fp in fps] == [0.2, 0.2, 0.2] and result.params["shrink"] == 0.2
    assert fps[0]["n_estimators"] == 50 and all(fp["n_estimators"] in {25, 50, 60} for fp in fps)
    assert result.params["n_estimators"] in {25, 50, 60}
    assert result.params["num_leaves"] == 5  # the other defaults stay
    loaded = models.load_artifact(result.artifact)
    assert loaded["classifier"].current_iteration() == result.params["n_estimators"]
    with pytest.raises(ValueError, match="shrink"):
        models.train_walk_forward(small, params={"shrink": 1.0}, min_train_days=MIN_TRAIN)
    with pytest.raises(ValueError, match="n_estimators"):
        models.train_walk_forward(small, params={"n_estimators": 0}, min_train_days=MIN_TRAIN)


# ------------------------------------------------------------------ prediction


def test_predict_returns_normalised_probs_in_class_order(
    frame: pd.DataFrame, trained: TrainResult
) -> None:
    row = live_row(frame, trained.feature_names)
    out = models.predict(trained.artifact, trained.feature_names, row)
    assert set(out.probs) == {"UP", "FLAT", "DOWN"}
    assert math.isclose(sum(out.probs.values()), 1.0, abs_tol=1e-9)
    assert all(1e-4 <= p <= 1.0 for p in out.probs.values())
    assert out.predicted_class == max(out.probs, key=out.probs.__getitem__)
    assert out.range_p10_pct is not None and out.range_p90_pct is not None
    assert out.range_p10_pct <= out.range_p90_pct
    assert out.confidence in CONFIDENCE_LEVELS
    assert (out.confidence, out.no_edge) == confidence_label(out.probs)
    assert 1 <= len(out.drivers) <= 5
    contribs = [abs(d.contribution) for d in out.drivers]
    assert contribs == sorted(contribs, reverse=True)
    for d in out.drivers:
        assert d.feature in FEATURES and d.text and out.predicted_class in d.text


def test_predict_applies_the_artifacts_shrinkage(frame: pd.DataFrame, trained: TrainResult) -> None:
    row = live_row(frame, trained.feature_names)
    x = models.row_vector(trained.feature_names, row)
    art = models.load_artifact(trained.artifact)
    raw = models._probs(art["classifier"], x)[0]
    clim = art["climatology"]

    def variant(**changes: object) -> str:
        payload = json.loads(trained.artifact)
        for key, value in changes.items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        return json.dumps(payload, sort_keys=True)

    # A fixed, known shrink: predict() mixes exactly as shrink_probs does.
    params = {**trained.params, "shrink": 0.6}
    out = models.predict(variant(params=params), trained.feature_names, row)
    expected = models.shrink_probs(raw[None, :], 0.6, clim)[0]
    for c in ("UP", "FLAT", "DOWN"):
        assert out.probs[c] == pytest.approx(expected[CLASS_INDEX[c]])
    for c in CLASSES:  # a convex mix: every class lands between the model and climatology
        lo, hi = sorted((raw[CLASS_INDEX[c]], clim[CLASS_INDEX[c]]))
        assert lo - 1e-9 <= out.probs[c] <= hi + 1e-9
    # Shrink 0, or an artifact from before shrinkage existed: the classifier's own numbers.
    for legacy in (
        variant(params={**trained.params, "shrink": 0.0}),
        variant(params={}, class_frequencies=None),
    ):
        out = models.predict(legacy, trained.feature_names, row)
        for c in ("UP", "FLAT", "DOWN"):
            assert out.probs[c] == pytest.approx(raw[CLASS_INDEX[c]])
    # Shrinkage asked for without class frequencies is an error, never a made-up prior.
    with pytest.raises(ValueError, match="class frequencies"):
        models.predict(variant(params=params, class_frequencies=None), trained.feature_names, row)


def test_predict_class_order_matches_the_signal(frame: pd.DataFrame, trained: TrainResult) -> None:
    """A huge overnight futures move must map to the right end of the class order."""
    row = live_row(frame, trained.feature_names)
    row["ret_1d"] = 0.0
    row["vix_level"] = 17.0
    up = models.predict(trained.artifact, trained.feature_names, {**row, "es_fut_ret_1d": 6.0})
    down = models.predict(trained.artifact, trained.feature_names, {**row, "es_fut_ret_1d": -6.0})
    assert up.predicted_class == "UP" and down.predicted_class == "DOWN"
    assert up.probs["UP"] > up.probs["DOWN"] and down.probs["DOWN"] > down.probs["UP"]
    assert up.range_p90_pct is not None and down.range_p10_pct is not None
    assert up.range_p90_pct > 0 > down.range_p10_pct


def test_missing_values_stay_missing(frame: pd.DataFrame, trained: TrainResult) -> None:
    row = live_row(frame, trained.feature_names)
    row["fii_net_cr"] = None
    row["es_fut_ret_1d"] = float("nan")
    vec = models.row_vector(trained.feature_names, row)
    assert vec.shape == (1, len(FEATURES))
    assert math.isnan(vec[0, FEATURES.index("fii_net_cr")])
    assert math.isnan(vec[0, FEATURES.index("es_fut_ret_1d")])  # never 0-filled
    out = models.predict(trained.artifact, trained.feature_names, row)
    assert math.isclose(sum(out.probs.values()), 1.0, abs_tol=1e-9)
    for d in out.drivers:
        if d.feature in {"fii_net_cr", "es_fut_ret_1d"}:
            assert d.value is None and "(no data)" in d.text


def test_predict_rejects_other_feature_names(frame: pd.DataFrame, trained: TrainResult) -> None:
    row = live_row(frame, trained.feature_names)
    with pytest.raises(ValueError, match="feature names"):
        models.predict(trained.artifact, trained.feature_names[:-1], row)


def test_frame_matrix_keeps_order_and_nans(frame: pd.DataFrame) -> None:
    x = models.frame_matrix(frame, ["dow", "fii_net_cr", "not_a_feature"])
    assert x.shape == (len(frame), 3)
    assert np.array_equal(x[:, 0], frame["dow"].to_numpy(dtype=float))
    assert np.isnan(x[0, 1]) and np.isnan(x[:, 2]).all()
    assert models.frame_matrix(frame, []).shape == (len(frame), 0)


# ------------------------------------------------------------------ drivers in plain English


def test_driver_text_uses_feature_labels_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("app.ml.features")
    fake.FEATURE_LABELS = {"es_fut_ret_1d": "US futures overnight"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.ml.features", fake)
    assert models.feature_label("es_fut_ret_1d") == "US futures overnight"
    assert models.feature_label("rsi_14") == "rsi 14"  # not in the labels: words, not code
    text = models.driver_text("es_fut_ret_1d", 1.8, 0.7, "UP")
    assert text == "US futures overnight at +1.80% pushes toward UP"
    # A unit already in the value phrase is not repeated from the label.
    fake.FEATURE_LABELS.update(  # type: ignore[attr-defined]
        {"ret_1d": "1-day return (%)", "fii_net_cr": "FII net flow (Rs cr)", "rsi14": "RSI (14)"}
    )
    assert models.driver_text("ret_1d", -0.4, -0.2, "UP") == (
        "1-day return at -0.40% pulls away from UP"
    )
    assert models.driver_text("fii_net_cr", 1250.0, 0.2, "UP") == (
        "FII net flow at Rs 1,250 cr pushes toward UP"
    )
    assert models.driver_text("rsi14", 61.0, 0.2, "UP") == "RSI (14) at 61 pushes toward UP"
    assert models.driver_text("ret_1d", None, 0.2, "UP") == (
        "1-day return (%) (no data) pushes toward UP"
    )
    assert models.driver_text("es_fut_ret_1d", -1.2, -0.4, "UP").endswith("pulls away from UP")


def test_driver_text_falls_back_without_features_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "app.ml.features", None)  # makes the import fail
    assert models.feature_label("vix_level") == "vix level"
    assert models.driver_text("fii_net_cr", None, 0.1, "DOWN") == (
        "fii net cr (no data) pushes toward DOWN"
    )


@pytest.mark.parametrize(
    ("name", "value", "phrase"),
    [
        ("ret_1d", 0.4215, "at +0.42%"),
        ("sp500_ret_1d", -1.2, "at -1.20%"),
        ("dist_dma200_pct", 3.456, "at +3.46%"),
        ("rsi_14", 61.4, "at 61"),
        ("vol_z20", 1.25, "at +1.25 sd"),
        ("bollinger_z", -0.5, "at -0.50 sd"),
        ("is_month_last_week", 1.0, "= yes"),
        ("budget_day", 0.0, "= no"),
        ("fomc_overnight", 1.0, "= yes"),
        ("dow", 1.0, "= Tuesday"),
        ("up_streak", 3.0, "= 3 days"),
        ("up_streak", 1.0, "= 1 day"),
        ("sessions_to_month_end", 4.0, "= 4 days"),
        ("fii_net_cr", -1250.4, "at Rs -1,250 cr"),
        ("vix_level", 14.26, "at 14.3"),
        ("vix_chg_1d", 4.0, "at +4.00%"),
        ("us10y_chg_1d", 0.05, "at +0.05 pts"),
        ("adv_decl_ratio", 1.37, "at 1.37"),
        ("pct_above_200dma", 62.3, "at 62%"),
        ("clv", 0.35, "at +0.35"),
        ("anything", None, "(no data)"),
    ],
)
def test_value_phrase(name: str, value: float | None, phrase: str) -> None:
    assert models.value_phrase(name, value) == phrase


def test_top_drivers_skip_features_that_did_nothing() -> None:
    contrib = np.array([0.0, 0.3, -0.5, 0.0, 0.1, 0.05, 0.02])
    names = [f"f{i}" for i in range(7)]
    row = {n: float(i) for i, n in enumerate(names)}
    out = models.top_drivers(names, row, contrib, "FLAT", n=5)
    assert [d.feature for d in out] == ["f2", "f1", "f4", "f5", "f6"]
    assert out[0].contribution == -0.5 and out[0].value == 2.0
    assert models.top_drivers(names, row, np.zeros(7), "FLAT") == []


# ------------------------------------------------------------------ baselines


ROW = {"ret_1d": 0.5, "es_fut_ret_1d": 1.0, "sp500_ret_1d": -0.8}
FREQ = {"UP": 0.4, "FLAT": 0.25, "DOWN": 0.35}
D = date(2026, 10, 30)


def test_baseline_calls_rules() -> None:
    calls = baselines.baseline_calls(ROW, "NIFTY 50", D, FREQ)
    assert set(calls) == set(BASELINE_NAMES)
    assert calls["always_up"] == "UP"
    assert calls["same_as_yesterday"] == "UP"
    assert calls["overnight_gap"] == "UP"  # 0.5 * 1.0 = +0.5 > 0.30
    assert calls["random"] in CLASSES


@pytest.mark.parametrize(
    ("ret_1d", "expected"), [(0.31, "UP"), (0.30, "FLAT"), (-0.30, "FLAT"), (-0.31, "DOWN")]
)
def test_same_as_yesterday_threshold_boundary(ret_1d: float, expected: str) -> None:
    calls = baselines.baseline_calls({"ret_1d": ret_1d}, "NIFTY 50", D, FREQ)
    assert calls["same_as_yesterday"] == expected


def test_baselines_are_none_when_inputs_are_missing() -> None:
    calls = baselines.baseline_calls({"ret_1d": None, "es_fut_ret_1d": float("nan")}, "X", D, FREQ)
    assert calls["same_as_yesterday"] is None and calls["overnight_gap"] is None
    # ES futures missing -> the S&P 500 cash return stands in (0.5 * -0.8 = -0.4 -> DOWN).
    calls = baselines.baseline_calls({"es_fut_ret_1d": None, "sp500_ret_1d": -0.8}, "X", D, FREQ)
    assert calls["overnight_gap"] == "DOWN"
    assert baselines.gap_expected_pct({"es_fut_ret_1d": 0.4, "sp500_ret_1d": 9.0}) == 0.2
    assert baselines.gap_expected_pct({}) is None
    assert baselines.baseline_calls({}, "X", D, {})["random"] is None


def test_random_baseline_is_reproducible_and_follows_frequencies() -> None:
    a = baselines.random_call("NIFTY 50", D, FREQ)
    assert a == baselines.random_call("NIFTY 50", D, FREQ)
    assert baselines.stable_seed("NIFTY 50", D) != baselines.stable_seed("NIFTY BANK", D)
    days = [date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(200)]
    draws = {baselines.random_call("NIFTY 50", d, FREQ) for d in days}
    assert draws == set(CLASSES)
    never_flat = {baselines.random_call("NIFTY 50", d, {"UP": 0.5, "DOWN": 0.5}) for d in days}
    assert "FLAT" not in never_flat


def test_baseline_probs_sum_to_one() -> None:
    hard = baselines.baseline_probs("always_up", "UP", FREQ)
    assert math.isclose(sum(hard.values()), 1.0) and hard["UP"] > 0.999
    clim = baselines.baseline_probs("random", "UP", FREQ)
    assert math.isclose(sum(clim.values()), 1.0)
    assert math.isclose(clim["UP"], 0.4, rel_tol=1e-3)
    no_call = baselines.baseline_probs("overnight_gap", None, FREQ)
    assert no_call == clim
    uniform = baselines.baseline_probs("overnight_gap", None, {})
    assert all(math.isclose(p, 1 / 3) for p in uniform.values())
    assert math.isclose(sum(baselines.normalise_probs({"UP": 2.0, "FLAT": 0.0}).values()), 1.0)


# ------------------------------------------------------------------ metrics


def _onehot(labels: list[int]) -> np.ndarray:
    out = np.zeros((len(labels), 3))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def test_brier_log_loss_accuracy_values() -> None:
    y = np.array([CLASS_INDEX["UP"], CLASS_INDEX["DOWN"], CLASS_INDEX["FLAT"]])
    perfect = _onehot(list(y))
    assert metrics.brier(perfect, y) == 0.0
    assert metrics.log_loss(perfect, y) == pytest.approx(0.0, abs=1e-9)
    assert metrics.accuracy(perfect.argmax(axis=1), y) == 1.0
    uniform = np.full((3, 3), 1 / 3)
    assert metrics.brier(uniform, y) == pytest.approx(2 / 3)
    assert metrics.log_loss(uniform, y) == pytest.approx(math.log(3))
    wrong = _onehot([CLASS_INDEX["DOWN"], CLASS_INDEX["UP"], CLASS_INDEX["UP"]])
    assert metrics.brier(wrong, y) == pytest.approx(2.0)
    assert metrics.log_loss(wrong, y) == pytest.approx(-math.log(1e-6))
    assert metrics.accuracy(wrong.argmax(axis=1), y) == 0.0
    assert metrics.accuracy(np.array([2, 0, 0]), y) == pytest.approx(2 / 3)
    for fn in (metrics.brier, metrics.log_loss):
        assert math.isnan(fn(np.empty((0, 3)), np.empty(0, dtype=int)))
    assert math.isnan(metrics.accuracy(np.empty(0), np.empty(0)))


def test_calibration_bins() -> None:
    max_prob = np.array([0.34, 0.40, 0.50, 0.55, 0.95, 1.0])
    correct = np.array([False, True, True, False, True, True])
    bins = metrics.calibration_bins(max_prob, correct, bins=5)
    assert [b["bin_lo"] for b in bins] == [0.3333, 0.4667, 0.8667]  # empty bins are dropped
    assert [b["n"] for b in bins] == [2, 2, 2]
    assert bins[0]["observed"] == 0.5 and bins[0]["predicted_mean"] == pytest.approx(0.37)
    assert bins[2]["observed"] == 1.0 and bins[2]["bin_hi"] == 1.0  # p = 1.0 is counted
    assert metrics.calibration_bins(np.empty(0), np.empty(0, dtype=bool)) == []


def test_by_confidence_rows() -> None:
    labels = ["high", "high", "low", "none"]
    rows = metrics.by_confidence(labels, np.array([True, False, True, False]))
    assert [r["confidence"] for r in rows] == list(CONFIDENCE_LEVELS)
    assert rows[0] == {"confidence": "high", "n": 2, "accuracy": 0.5}
    assert rows[1] == {"confidence": "medium", "n": 0}  # no number for an empty bucket
    assert rows[2]["accuracy"] == 1.0 and rows[3]["accuracy"] == 0.0


# ------------------------------------------------------------------ API isolation


def test_ml_not_in_api() -> None:
    """The API process never imports LightGBM/scikit-learn (they are not in the Vercel bundle)."""
    code = (
        "import sys\n"
        "import app.main, app.ml.core, app.ml.metrics, app.ml.baselines\n"
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

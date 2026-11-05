"""The market-call model: LightGBM, trained and validated strictly walk-forward (brief §4.3).

LightGBM is imported inside functions only. The API process imports `app.ml.core`,
`app.ml.metrics` and `app.ml.baselines`, never this module's heavy parts; a test asserts
`import app.main` leaves `lightgbm` out of `sys.modules`.

Validation is an expanding window: fold j fits on labelled sessions [0, k) and predicts
the next `step_days` sessions; every out-of-sample row is predicted by a model that never
saw it or anything after it. No random splits.

Two knobs are chosen out of sample, sequentially, so the reported metrics stay honest:

- the number of boosting rounds: each fold trains up to `n_estimators` rounds and its
  out-of-sample rows are scored at every ROUND_STEP-th round (`num_iteration`);
- a shrinkage weight towards the fold's training class frequencies,
  p = (1 - a) * p_model + a * p_climatology, a in SHRINK_GRID.

Fold j is predicted with the (rounds, shrink) pair that had the lowest pooled
out-of-sample log loss over folds < j; the first fold uses conservative defaults. A fold's
own rows only ever inform the choice for LATER folds. The final fit uses the pair chosen on
every fold and records it in params["n_estimators"] / params["shrink"]. Given the same
frame, params and seed the artifact is byte-identical (single thread, deterministic
histograms).

The artifact is JSON text {"classifier", "q10", "q90", "feature_names", "params",
"class_frequencies"} where the three model strings come from `Booster.model_to_string()`;
it lives in the database (`ModelVersion.artifact`), not on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.ml import metrics
from app.ml.baselines import PROB_FLOOR, baseline_calls, baseline_probs
from app.ml.core import (
    BASELINE_NAMES,
    CLASS_INDEX,
    CLASSES,
    Driver,
    FeatureRow,
    MarketCallOutput,
    TrainResult,
    WalkForwardFold,
    WalkForwardMetrics,
    confidence_label,
    finite,
    predicted_class,
)

if TYPE_CHECKING:
    from lightgbm import Booster

MAX_ROUNDS = 400  # the most boosting rounds tried ("n_estimators" in DEFAULT_PARAMS)
ROUND_STEP = 25  # rounds are scored out of sample at ROUND_STEP, 2 * ROUND_STEP, ..., the ceiling
FIRST_FOLD_ROUNDS = 50  # before any out-of-sample evidence exists: few rounds ...
FIRST_FOLD_SHRINK = 0.5  # ... and half-way to climatology
SHRINK_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(10))  # 0.0 .. 0.9
# Chosen out of sample by the walk-forward, so not part of the model recipe a caller sets.
SELECTED_PARAM_KEYS: frozenset[str] = frozenset({"n_estimators", "shrink"})
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
    "num_leaves": 5,
    "min_data_in_leaf": 60,
    "learning_rate": 0.03,
    "n_estimators": MAX_ROUNDS,  # ceiling of the round search; the artifact records the choice
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "seed": 42,
    "verbose": -1,
    # Reproducibility: same data + params + seed -> the same trees, byte for byte.
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
QUANTILES: tuple[tuple[str, float], ...] = (("q10", 0.1), ("q90", 0.9))
LABEL_COLUMNS: tuple[str, ...] = ("label_pct", "label_class", "reference_date", "reference_close")
TOP_DRIVERS = 5
_ARTIFACT_CACHE_SIZE = 8
_ARTIFACT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


@dataclass
class MarketCallMetrics(WalkForwardMetrics):
    """`WalkForwardMetrics` plus what the walk-forward chose and how the range held up.

    range_coverage: share of out-of-sample `label_pct` inside [p10, p90] where the quantile
    models of each fold were fitted on that fold's training rows only (nominally 0.8).
    fold_params: the (n_estimators, shrink) each fold's out-of-sample rows were scored with.
    """

    range_coverage: float = math.nan
    fold_params: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        out = super().as_dict()
        out["range_coverage"] = finite(self.range_coverage)
        out["fold_params"] = [dict(fp) for fp in self.fold_params]
        return out


# --------------------------------------------------------------------------- frame helpers


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The feature columns of a `features.build_frame` frame, in frame order (labels excluded)."""
    return [str(c) for c in frame.columns if c not in LABEL_COLUMNS]


def frame_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """A float matrix in `feature_names` order; missing values stay NaN (never filled)."""
    if not feature_names:
        return np.empty((len(frame), 0), dtype=float)
    cols = frame.reindex(columns=list(feature_names))
    return np.column_stack(
        [pd.to_numeric(cols[c], errors="coerce").to_numpy(dtype=float) for c in feature_names]
    )


def row_vector(feature_names: Sequence[str], row: Mapping[str, float | None]) -> np.ndarray:
    """One live feature row as a (1, n) float matrix, NaN where a number is missing."""
    vals = [finite(row.get(f)) for f in feature_names]
    return np.array([[math.nan if v is None else v for v in vals]], dtype=float)


def _as_date(x: object) -> date:
    if isinstance(x, datetime):  # pd.Timestamp is a datetime subclass
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(str(x)).date()


def _class_frequencies(y: np.ndarray) -> dict[str, float]:
    n = int(y.size)
    if n == 0:
        return {}
    return {c: float(np.sum(y == CLASS_INDEX[c])) / n for c in CLASSES}


def _climatology(freq: Mapping[str, float]) -> np.ndarray | None:
    """Class frequencies as a probability vector in CLASSES order, or None when there are none."""
    v = np.array([max(float(freq.get(c, 0.0) or 0.0), 0.0) for c in CLASSES], dtype=float)
    total = float(v.sum())
    if total <= 0.0:
        return None
    return v / total


# --------------------------------------------------------------------------- LightGBM fits


def _booster_params(params: Mapping[str, object]) -> dict[str, object]:
    """The LightGBM parameters: `params` without the walk-forward's own keys."""
    return {k: v for k, v in params.items() if k not in SELECTED_PARAM_KEYS}


def _max_rounds(params: Mapping[str, object]) -> int:
    raw = params.get("n_estimators", MAX_ROUNDS)
    rounds = int(raw) if isinstance(raw, int | float | str) else MAX_ROUNDS
    if rounds < 1:
        raise ValueError("n_estimators must be positive")
    return rounds


def round_grid(max_rounds: int) -> tuple[int, ...]:
    """Round counts scored out of sample: every ROUND_STEP up to and including `max_rounds`."""
    grid = list(range(ROUND_STEP, max_rounds, ROUND_STEP))
    grid.append(max_rounds)
    return tuple(grid)


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    rounds: int,
) -> Booster:
    import lightgbm as lgb

    data = lgb.Dataset(x, label=y, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train(_booster_params(params), data, num_boost_round=rounds)


def _fit_quantile(
    x: np.ndarray,
    y_pct: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    alpha: float,
    rounds: int,
) -> Booster:
    import lightgbm as lgb

    p = _booster_params(params)
    p.pop("num_class", None)
    p.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
    data = lgb.Dataset(x, label=y_pct, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train(p, data, num_boost_round=rounds)


def _quantile_range(
    x_train: np.ndarray,
    y_pct_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(p10, p90) for `x_test` from quantile models fitted on the training rows only.

    Ordered per row, so a crossing pair still gives a well-defined range.
    """
    preds = [
        np.asarray(
            _fit_quantile(x_train, y_pct_train, feature_names, params, alpha, rounds).predict(
                x_test
            ),
            dtype=float,
        ).reshape(-1)
        for _, alpha in QUANTILES
    ]
    return np.minimum(preds[0], preds[1]), np.maximum(preds[0], preds[1])


def _normalise(p: np.ndarray) -> np.ndarray:
    """Clip every class to [PROB_FLOOR, 1] and renormalise along the last axis.

    The array form of `baselines.normalise_probs`; works on (n, 3) and (K, n, 3) alike.
    """
    clipped = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1.0)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def _probs(booster: Booster, x: np.ndarray, rounds: int | None = None) -> np.ndarray:
    """(n, 3) probabilities in CLASSES order from the first `rounds` trees (all when None)."""
    raw = booster.predict(x, num_iteration=rounds)
    return _normalise(np.asarray(raw, dtype=float).reshape(x.shape[0], len(CLASSES)))


def shrink_probs(p: np.ndarray, shrink: float, climatology: np.ndarray | None) -> np.ndarray:
    """(1 - shrink) * p + shrink * climatology, renormalised; `shrink` 0 leaves `p` alone.

    `climatology` is the training class frequencies in CLASSES order (broadcast over the
    leading axes of `p`). Raises ValueError when shrinkage is asked for without them: a
    missing number is never replaced by a made-up prior.
    """
    if not 0.0 <= shrink < 1.0:
        raise ValueError(f"shrink must be in [0, 1), got {shrink}")
    if shrink == 0.0:
        return np.asarray(p, dtype=float)
    if climatology is None:
        raise ValueError("shrinkage needs the training class frequencies")
    return _normalise((1.0 - shrink) * np.asarray(p, dtype=float) + shrink * climatology)


def _probs_dict(p: np.ndarray) -> dict[str, float]:
    return {c: float(p[CLASS_INDEX[c]]) for c in ("UP", "FLAT", "DOWN")}


# --------------------------------------------------------------------------- selection


def select_rounds_and_shrink(
    loss_sum: np.ndarray,
    n_scored: int,
    rounds: Sequence[int],
    shrinks: Sequence[float],
    default: tuple[int, float],
) -> tuple[int, float]:
    """The (rounds, shrink) pair with the lowest pooled log loss so far; `default` before any.

    `loss_sum[i, j]` is the summed out-of-sample log loss of rounds[i] with shrinks[j] over
    the `n_scored` rows scored so far. Ties go to the more conservative pair: fewer rounds
    first, then more shrinkage.
    """
    if n_scored <= 0:
        return default
    mean = np.asarray(loss_sum, dtype=float) / n_scored
    best = float(np.nanmin(mean))
    for i in range(len(rounds)):
        for j in reversed(range(len(shrinks))):
            if mean[i, j] <= best + 1e-12:
                return int(rounds[i]), float(shrinks[j])
    raise AssertionError("unreachable: the minimum is in the grid")


def _shrink_grid(params: Mapping[str, object]) -> tuple[tuple[float, ...], float]:
    """(grid, first-fold default): SHRINK_GRID, or the single fixed value in params["shrink"]."""
    fixed = params.get("shrink")
    if fixed is None:
        return SHRINK_GRID, FIRST_FOLD_SHRINK
    a = float(fixed)  # type: ignore[arg-type]
    if not 0.0 <= a < 1.0:
        raise ValueError(f"shrink must be in [0, 1), got {a}")
    return (a,), a


# --------------------------------------------------------------------------- training


def train_walk_forward(
    frame: pd.DataFrame,
    *,
    params: Mapping[str, object] | None = None,
    min_train_days: int | None = None,
    step_days: int | None = None,
    settings: Settings | None = None,
    target: str | None = None,
) -> TrainResult:
    """Strict expanding-window walk-forward validation, then a final fit on every label.

    For k = min_train, min_train + step, ...: fit on labelled rows [0, k) with the round
    ceiling, score rows [k, k + step) with the (rounds, shrink) pair chosen on the folds
    before this one (see the module docstring), and add this fold's scores at every grid
    pair to the pool the later folds choose from. The out-of-sample probabilities give the
    headline metrics and, on the same rows, each baseline's accuracy/Brier (class
    frequencies for the random baseline and for the shrinkage come from the fold's own
    training rows). Each fold also fits the two quantile models on its training rows with
    the fold's rounds, for `range_coverage`. Rows with a NaN label are excluded. Raises
    ValueError when there are fewer than `min_train_days + step_days` labelled rows.

    `params` override DEFAULT_PARAMS: "n_estimators" is the ceiling of the round search and
    "shrink", when given, fixes the shrinkage instead of choosing it. The returned params
    carry the chosen "n_estimators" and "shrink". `target` only seeds the random baseline
    (falls back to `frame.attrs["target"]`).
    """
    s = settings or get_settings()
    min_train = s.wf_min_train_days if min_train_days is None else int(min_train_days)
    step = s.wf_step_days if step_days is None else int(step_days)
    if min_train < 1 or step < 1:
        raise ValueError("min_train_days and step_days must be positive")
    effective: dict[str, object] = {**DEFAULT_PARAMS, **dict(params or {})}
    max_rounds = _max_rounds(effective)
    rounds_grid = round_grid(max_rounds)
    shrink_grid, first_shrink = _shrink_grid(effective)
    first_choice = (min(FIRST_FOLD_ROUNDS, max_rounds), first_shrink)
    tgt = target if target is not None else str(frame.attrs.get("target", ""))

    labelled = frame[frame["label_pct"].notna() & frame["label_class"].notna()]
    n = len(labelled)
    if n < min_train + step:
        raise ValueError(
            f"{n} labelled sessions; walk-forward needs at least {min_train + step} "
            f"(min_train_days={min_train} + step_days={step})"
        )
    feature_names = feature_columns(labelled)
    if not feature_names:
        raise ValueError("frame has no feature columns")
    x = frame_matrix(labelled, feature_names)
    y = labelled["label_class"].map(CLASS_INDEX).to_numpy(dtype=int)
    y_pct = labelled["label_pct"].to_numpy(dtype=float)
    dates = [_as_date(d) for d in labelled.index]
    rows: list[dict[str, float]] = [
        dict(zip(feature_names, map(float, xi), strict=True)) for xi in x
    ]

    oos_probs = np.full((n, len(CLASSES)), np.nan)
    oos = np.zeros(n, dtype=bool)
    base_pred: dict[str, np.ndarray] = {b: np.full(n, -1, dtype=int) for b in BASELINE_NAMES}
    base_probs: dict[str, np.ndarray] = {
        b: np.full((n, len(CLASSES)), np.nan) for b in BASELINE_NAMES
    }
    loss_sum = np.zeros((len(rounds_grid), len(shrink_grid)))  # pooled over folds scored so far
    n_scored = 0
    in_range = 0
    folds: list[WalkForwardFold] = []
    fold_params: list[dict[str, float]] = []
    for k in range(min_train, n, step):
        end = min(k + step, n)
        # Chosen on folds < this one only: this fold's rows are still unseen at this point.
        rounds, shrink = select_rounds_and_shrink(
            loss_sum, n_scored, rounds_grid, shrink_grid, first_choice
        )
        freq_train = _class_frequencies(y[:k])
        clim = _climatology(freq_train)
        booster = _fit_classifier(x[:k], y[:k], feature_names, effective, max_rounds)
        xk, yk = x[k:end], y[k:end]
        p = shrink_probs(_probs(booster, xk, rounds), shrink, clim)
        oos_probs[k:end] = p
        oos[k:end] = True
        for i in range(k, end):
            calls = baseline_calls(rows[i], tgt, dates[i], freq_train, s)
            for b in BASELINE_NAMES:
                call = calls[b]
                if call is not None:
                    base_pred[b][i] = CLASS_INDEX[call]
                bp = baseline_probs(b, call, freq_train)
                base_probs[b][i] = [bp[c] for c in CLASSES]
        lo, hi = _quantile_range(x[:k], y_pct[:k], xk, feature_names, effective, rounds)
        in_range += int(np.sum((lo <= y_pct[k:end]) & (y_pct[k:end] <= hi)))
        folds.append(
            WalkForwardFold(
                train_end=dates[k - 1],
                test_start=dates[k],
                test_end=dates[end - 1],
                n_test=end - k,
                accuracy=metrics.accuracy(p.argmax(axis=1), yk),
                brier=metrics.brier(p, yk),
                log_loss=metrics.log_loss(p, yk),
            )
        )
        fold_params.append({"n_estimators": rounds, "shrink": shrink})
        # Now, and only now, this fold's rows join the pool the later folds choose from.
        grid_probs = np.stack([_probs(booster, xk, r) for r in rounds_grid])  # (K, n_test, 3)
        for j, a in enumerate(shrink_grid):
            mixed = shrink_probs(grid_probs, a, clim)
            for i in range(len(rounds_grid)):
                loss_sum[i, j] += metrics.log_loss(mixed[i], yk) * (end - k)
        n_scored += end - k

    po, yo = oos_probs[oos], y[oos]
    pred = po.argmax(axis=1)
    correct = pred == yo
    labels = [confidence_label(_probs_dict(p), s)[0] for p in po]
    baselines: dict[str, dict[str, float]] = {}
    for b in BASELINE_NAMES:
        has_call = oos & (base_pred[b] >= 0)
        nb = int(has_call.sum())
        entry: dict[str, float] = {"n": nb}
        if nb:
            entry["accuracy"] = metrics.accuracy(base_pred[b][has_call], y[has_call])
            entry["brier"] = metrics.brier(base_probs[b][has_call], y[has_call])
        baselines[b] = entry
    n_oos = int(oos.sum())
    wf = MarketCallMetrics(
        n_oos=n_oos,
        accuracy=metrics.accuracy(pred, yo),
        brier=metrics.brier(po, yo),
        log_loss=metrics.log_loss(po, yo),
        class_frequencies=_class_frequencies(y),
        baselines=baselines,
        by_confidence=metrics.by_confidence(labels, correct),
        calibration=metrics.calibration_bins(po.max(axis=1), correct),
        folds=folds,
        range_coverage=in_range / n_oos if n_oos else math.nan,
        fold_params=fold_params,
    )

    # The final pair is chosen on every fold's out-of-sample rows (all unseen by their fits).
    rounds, shrink = select_rounds_and_shrink(
        loss_sum, n_scored, rounds_grid, shrink_grid, first_choice
    )
    effective["n_estimators"] = rounds
    effective["shrink"] = shrink
    classifier = _fit_classifier(x, y, feature_names, effective, rounds)
    quantiles = {
        key: _fit_quantile(x, y_pct, feature_names, effective, alpha, rounds)
        for key, alpha in QUANTILES
    }
    artifact = json.dumps(
        {
            "classifier": classifier.model_to_string(),
            **{key: q.model_to_string() for key, q in quantiles.items()},
            "feature_names": feature_names,
            "params": effective,
            "class_frequencies": wf.class_frequencies,
        },
        sort_keys=True,
    )
    return TrainResult(
        artifact=artifact,
        feature_names=feature_names,
        params=effective,
        metrics=wf,
        trained_through=dates[-1],
        train_rows=n,
    )


# --------------------------------------------------------------------------- inference


def artifact_hash(artifact: str) -> str:
    return hashlib.sha256(artifact.encode("utf-8")).hexdigest()


def load_artifact(artifact: str) -> dict[str, Any]:
    """Boosters rebuilt from the artifact JSON, cached in-process by the artifact's sha256.

    Returns {"classifier", "q10", "q90": Booster, "feature_names": list[str], "hash": str,
    "shrink": float, "climatology": np.ndarray | None}. An artifact from before shrinkage
    existed loads with shrink 0 and no climatology.
    """
    key = artifact_hash(artifact)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        _ARTIFACT_CACHE.move_to_end(key)
        return cached
    import lightgbm as lgb

    payload = json.loads(artifact)
    loaded: dict[str, Any] = {
        name: lgb.Booster(model_str=payload[name]) for name in ("classifier", "q10", "q90")
    }
    names = payload.get("feature_names") or loaded["classifier"].feature_name()
    loaded["feature_names"] = [str(f) for f in names]
    loaded["hash"] = key
    params = payload.get("params") or {}
    raw_shrink = params.get("shrink", 0.0) if isinstance(params, Mapping) else 0.0
    loaded["shrink"] = float(raw_shrink or 0.0)
    freq = payload.get("class_frequencies")
    loaded["climatology"] = _climatology(freq) if isinstance(freq, Mapping) else None
    _ARTIFACT_CACHE[key] = loaded
    while len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_SIZE:
        _ARTIFACT_CACHE.popitem(last=False)
    return loaded


def predict(
    artifact: str,
    feature_names: list[str],
    row: FeatureRow,
    *,
    settings: Settings | None = None,
) -> MarketCallOutput:
    """The market call for one live feature row.

    Probabilities are the classifier's, shrunk towards the training class frequencies by
    the artifact's `shrink` (chosen out of sample at training time), and sum to 1; the
    10th/90th percentile of the close-to-close % come from the quantile regressors; drivers
    are the five features with the largest contribution (`pred_contrib`) to the predicted
    class's raw score, in plain English. Raises ValueError if `feature_names` differ from
    the names the artifact was trained with, or if the artifact asks for shrinkage but
    carries no class frequencies.
    """
    s = settings or get_settings()
    art = load_artifact(artifact)
    trained_names = list(art["classifier"].feature_name())
    if list(feature_names) != trained_names:
        raise ValueError(
            f"feature names do not match the model: got {len(feature_names)}, "
            f"trained with {len(trained_names)}"
        )
    x = row_vector(feature_names, row)
    p_model = _probs(art["classifier"], x)
    probs = _probs_dict(shrink_probs(p_model, art["shrink"], art["climatology"])[0])
    cls = predicted_class(probs)
    p10 = finite(np.asarray(art["q10"].predict(x), dtype=float).reshape(-1)[0])
    p90 = finite(np.asarray(art["q90"].predict(x), dtype=float).reshape(-1)[0])
    if p10 is not None and p90 is not None and p10 > p90:
        p10, p90 = p90, p10  # quantile crossing: report the ordered pair
    contrib = np.asarray(art["classifier"].predict(x, pred_contrib=True), dtype=float).reshape(
        len(CLASSES), len(feature_names) + 1
    )[CLASS_INDEX[cls], :-1]  # the last column is the bias term
    drivers = top_drivers(feature_names, row, contrib, cls)
    confidence, no_edge = confidence_label(probs, s)
    return MarketCallOutput(
        probs=probs,
        predicted_class=cls,
        range_p10_pct=p10,
        range_p90_pct=p90,
        drivers=drivers,
        confidence=confidence,
        no_edge=no_edge,
    )


# --------------------------------------------------------------------------- drivers


def top_drivers(
    feature_names: Sequence[str],
    row: Mapping[str, float | None],
    contrib: np.ndarray,
    predicted: str,
    n: int = TOP_DRIVERS,
) -> list[Driver]:
    """The `n` features that moved the predicted class's score most, largest first.

    Features with a zero contribution are left out: a feature that did nothing is not a
    driver, so a flat model can legitimately report fewer than `n` reasons.
    """
    c = np.asarray(contrib, dtype=float).reshape(-1)
    order = np.argsort(-np.abs(c), kind="stable")
    out: list[Driver] = []
    for i in order:
        if len(out) >= n or c[i] == 0.0 or math.isnan(c[i]):
            break
        name = feature_names[int(i)]
        value = finite(row.get(name))
        out.append(
            Driver(
                feature=name,
                value=value,
                contribution=float(c[i]),
                text=driver_text(name, value, float(c[i]), predicted),
            )
        )
    return out


def _feature_labels() -> Mapping[str, str]:
    """`app.ml.features.FEATURE_LABELS`, or {} when that module is absent or malformed."""
    try:
        from app.ml.features import FEATURE_LABELS
    except Exception:  # the raw feature name is still an honest fallback
        return {}
    return FEATURE_LABELS if isinstance(FEATURE_LABELS, Mapping) else {}


def feature_label(name: str) -> str:
    """Plain-English label for a feature, else its name in words ("rsi_14" -> "rsi 14")."""
    label = _feature_labels().get(name)
    return str(label) if label else name.replace("_", " ")


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_PCT_HINTS = ("ret", "pct", "chg", "change", "gap", "range", "dist", "vol", "rv_", "atr", "macd")


def value_phrase(name: str, value: float | None) -> str:
    """How the value reads in a driver sentence, inferred from the feature's name.

    Returns e.g. "at +0.42%", "at 61", "at +1.3 sd", "= yes", "= Tuesday", "= 3 days",
    "at Rs 1,250 cr", or "(no data)" when the number is missing.
    """
    if value is None:
        return "(no data)"
    key = name.lower()
    if key.startswith("is_") or key.endswith(("_day", "_overnight", "_flag", "_loaded")):
        return f"= {'yes' if value >= 0.5 else 'no'}"
    if key in {"dow", "day_of_week", "weekday"} and 0 <= value < len(_WEEKDAYS):
        return f"= {_WEEKDAYS[int(round(value))]}"
    if "streak" in key or key.startswith("sessions_to") or key.endswith("_days"):
        n = int(round(value))
        return f"= {n} day{'' if abs(n) == 1 else 's'}"
    if key.startswith(("fii_", "dii_")) or key.endswith("_cr"):
        return f"at Rs {value:,.0f} cr"
    if "rsi" in key:
        return f"at {value:.0f}"
    if "zscore" in key or key.endswith(("_z", "_z20")) or "boll" in key:
        return f"at {value:+.2f} sd"
    if key.endswith("_ratio") or key.startswith("adv_decl"):
        return f"at {value:.2f}"
    if key.startswith("pct_above"):
        return f"at {value:.0f}%"
    if key.startswith("us10y"):
        return f"at {value:+.2f} pts"
    if key in {"vix", "vix_level"} or key.startswith("vix_level"):
        return f"at {value:.1f}"
    if any(h in key for h in _PCT_HINTS):
        return f"at {value:+.2f}%"
    return f"at {value:+.4g}"


# A label's trailing unit, e.g. "1-day return (%)", "FII net flow (Rs cr)", "change (points)".
_UNIT_SUFFIX = re.compile(r"\s*\((?:[^()]*%[^()]*|Rs cr|points)\)$")


def driver_text(name: str, value: float | None, contribution: float, predicted: str) -> str:
    """One plain-English sentence: label, value, and which way it pushed the call.

    E.g. "S&P 500 futures 1-day return at -1.20% pulls away from UP". The label's unit
    suffix is dropped when the value phrase already carries the unit.
    """
    label = feature_label(name)
    phrase = value_phrase(name, value)
    if phrase.endswith(("%", " cr", " pts")):
        label = _UNIT_SUFFIX.sub("", label)
    verb = "pushes toward" if contribution > 0 else "pulls away from"
    return f"{label} {phrase} {verb} {predicted}"

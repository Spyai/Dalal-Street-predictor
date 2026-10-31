"""The market-call model: LightGBM, trained and validated strictly walk-forward (brief §4.3).

LightGBM is imported inside functions only. The API process imports `app.ml.core`,
`app.ml.metrics` and `app.ml.baselines`, never this module's heavy parts; a test asserts
`import app.main` leaves `lightgbm` out of `sys.modules`.

Validation is an expanding window: fold k fits on labelled sessions [0, k) and predicts
the next `step_days` sessions; every out-of-sample row is predicted by a model that never
saw it or anything after it. No random splits. Given the same frame, params and seed the
artifact is byte-identical (single thread, deterministic histograms).

The artifact is JSON text {"classifier", "q10", "q90", "feature_names", "params"} where the
three model strings come from `Booster.model_to_string()`; it lives in the database
(`ModelVersion.artifact`), not on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.ml import metrics
from app.ml.baselines import baseline_calls, baseline_probs, normalise_probs
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

DEFAULT_ROUNDS = 300  # boosting rounds ("n_estimators" in DEFAULT_PARAMS)
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
    "num_leaves": 7,
    "min_data_in_leaf": 40,
    "learning_rate": 0.03,
    "n_estimators": DEFAULT_ROUNDS,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
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


# --------------------------------------------------------------------------- LightGBM fits


def _booster_params(params: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """Split the sklearn-style `n_estimators` out of a params dict for `lgb.train`."""
    p = dict(params)
    raw = p.pop("n_estimators", DEFAULT_ROUNDS)
    rounds = int(raw) if isinstance(raw, int | float | str) else DEFAULT_ROUNDS
    return p, rounds


def _fit_classifier(
    x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], params: Mapping[str, object]
) -> Booster:
    import lightgbm as lgb

    p, rounds = _booster_params(params)
    data = lgb.Dataset(x, label=y, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train(p, data, num_boost_round=rounds)


def _fit_quantile(
    x: np.ndarray,
    y_pct: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    alpha: float,
) -> Booster:
    import lightgbm as lgb

    p, rounds = _booster_params(params)
    p.pop("num_class", None)
    p.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
    data = lgb.Dataset(x, label=y_pct, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train(p, data, num_boost_round=rounds)


def _probs(booster: Booster, x: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in CLASSES order, clipped to [1e-4, 1] and renormalised per row."""
    raw = np.asarray(booster.predict(x), dtype=float).reshape(x.shape[0], len(CLASSES))
    out = np.empty_like(raw)
    for i, r in enumerate(raw):
        d = normalise_probs({c: float(r[j]) for j, c in enumerate(CLASSES)})
        out[i] = [d[c] for c in CLASSES]
    return out


def _probs_dict(p: np.ndarray) -> dict[str, float]:
    return {c: float(p[CLASS_INDEX[c]]) for c in ("UP", "FLAT", "DOWN")}


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

    For k = min_train, min_train + step, ...: fit on labelled rows [0, k), predict rows
    [k, k + step). The out-of-sample probabilities give the headline metrics and, with the
    same rows, each baseline's accuracy/Brier (class frequencies for the random baseline
    come from the fold's own training rows). Rows with a NaN label are excluded. Raises
    ValueError when there are fewer than `min_train_days + step_days` labelled rows.

    `target` only seeds the random baseline (falls back to `frame.attrs["target"]`).
    """
    s = settings or get_settings()
    min_train = s.wf_min_train_days if min_train_days is None else int(min_train_days)
    step = s.wf_step_days if step_days is None else int(step_days)
    if min_train < 1 or step < 1:
        raise ValueError("min_train_days and step_days must be positive")
    effective: dict[str, object] = {**DEFAULT_PARAMS, **dict(params or {})}
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
    folds: list[WalkForwardFold] = []
    for k in range(min_train, n, step):
        end = min(k + step, n)
        booster = _fit_classifier(x[:k], y[:k], feature_names, effective)
        p = _probs(booster, x[k:end])
        oos_probs[k:end] = p
        oos[k:end] = True
        freq_train = _class_frequencies(y[:k])
        for i in range(k, end):
            calls = baseline_calls(rows[i], tgt, dates[i], freq_train, s)
            for b in BASELINE_NAMES:
                call = calls[b]
                if call is not None:
                    base_pred[b][i] = CLASS_INDEX[call]
                bp = baseline_probs(b, call, freq_train)
                base_probs[b][i] = [bp[c] for c in CLASSES]
        yk = y[k:end]
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
    wf = WalkForwardMetrics(
        n_oos=int(oos.sum()),
        accuracy=metrics.accuracy(pred, yo),
        brier=metrics.brier(po, yo),
        log_loss=metrics.log_loss(po, yo),
        class_frequencies=_class_frequencies(y),
        baselines=baselines,
        by_confidence=metrics.by_confidence(labels, correct),
        calibration=metrics.calibration_bins(po.max(axis=1), correct),
        folds=folds,
    )

    classifier = _fit_classifier(x, y, feature_names, effective)
    quantiles = {
        key: _fit_quantile(x, y_pct, feature_names, effective, alpha) for key, alpha in QUANTILES
    }
    artifact = json.dumps(
        {
            "classifier": classifier.model_to_string(),
            **{key: q.model_to_string() for key, q in quantiles.items()},
            "feature_names": feature_names,
            "params": effective,
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

    Returns {"classifier", "q10", "q90": Booster, "feature_names": list[str], "hash": str}.
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

    Probabilities sum to 1; the 10th/90th percentile of the close-to-close % come from the
    quantile regressors; drivers are the five features with the largest contribution
    (`pred_contrib`) to the predicted class's raw score, in plain English. Raises
    ValueError if `feature_names` differ from the names the artifact was trained with.
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
    probs = _probs_dict(_probs(art["classifier"], x)[0])
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

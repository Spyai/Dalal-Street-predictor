"""Scoring rules for the market call (brief §6.1, §6.3). numpy only: the API imports this.

Conventions
- `probs` is an (n, 3) array in `core.CLASSES` order (DOWN, FLAT, UP); `y` holds class
  indices (ints); `pred` holds predicted class indices.
- Empty inputs score NaN, never a made-up number; callers pass NaN through
  `core.finite()` so it becomes None in JSON.
"""

from __future__ import annotations

import math

import numpy as np

from app.ml.core import CLASSES, CONFIDENCE_LEVELS

# Max probability of a 3-class forecast is at least 1/3, so calibration bins span [1/3, 1].
_MIN_MAX_PROB = 1.0 / len(CLASSES)


def _one_hot(y: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    yi = np.asarray(y, dtype=int).reshape(-1)
    out = np.zeros((yi.size, n_classes), dtype=float)
    out[np.arange(yi.size), yi] = 1.0
    return out


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean over rows of sum_k (p_k - 1[y = k])^2.

    0.0 for a perfect forecast, 2/3 for the uniform forecast, 2.0 for a confident miss.
    """
    p = np.asarray(probs, dtype=float).reshape(-1, len(CLASSES))
    if p.shape[0] == 0:
        return math.nan
    return float(np.mean(np.sum((p - _one_hot(y)) ** 2, axis=1)))


def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Mean negative log probability of the realised class (ln 3 for the uniform forecast)."""
    p = np.asarray(probs, dtype=float).reshape(-1, len(CLASSES))
    if p.shape[0] == 0:
        return math.nan
    yi = np.asarray(y, dtype=int).reshape(-1)
    p_true = np.clip(p[np.arange(yi.size), yi], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Share of rows whose predicted class index equals the realised one."""
    a = np.asarray(pred).reshape(-1)
    if a.size == 0:
        return math.nan
    return float(np.mean(a == np.asarray(y).reshape(-1)))


def calibration_bins(
    max_prob: np.ndarray, correct: np.ndarray, bins: int = 5
) -> list[dict[str, float | int]]:
    """Reliability table: is the model right ~70 % of the time when it says 70 %?

    Rows are equal-width bins of the top-class probability over [1/3, 1]; empty bins are
    left out (there is nothing honest to show for them). Each row:
    {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}.
    """
    mp = np.asarray(max_prob, dtype=float).reshape(-1)
    ok = np.asarray(correct, dtype=bool).reshape(-1)
    if mp.size == 0 or bins < 1:
        return []
    edges = np.linspace(_MIN_MAX_PROB, 1.0, bins + 1)
    # The last bin is closed on the right so a probability of exactly 1.0 is counted.
    idx = np.clip(np.searchsorted(edges, mp, side="right") - 1, 0, bins - 1)
    out: list[dict[str, float | int]] = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "bin_lo": round(float(edges[b]), 4),
                "bin_hi": round(float(edges[b + 1]), 4),
                "n": n,
                "predicted_mean": float(mp[mask].mean()),
                "observed": float(ok[mask].mean()),
            }
        )
    return out


def by_confidence(labels: list[str], correct: np.ndarray) -> list[dict[str, float | str | int]]:
    """Accuracy per confidence label, one row per `core.CONFIDENCE_LEVELS` entry in order.

    A level with no calls keeps its row ({"confidence", "n": 0}) but has no "accuracy"
    key: there is no number to report.
    """
    lab = np.asarray(list(labels), dtype=object).reshape(-1)
    ok = np.asarray(correct, dtype=bool).reshape(-1)
    out: list[dict[str, float | str | int]] = []
    for level in CONFIDENCE_LEVELS:
        mask = lab == level
        n = int(mask.sum())
        row: dict[str, float | str | int] = {"confidence": level, "n": n}
        if n:
            row["accuracy"] = float(ok[mask].mean())
        out.append(row)
    return out

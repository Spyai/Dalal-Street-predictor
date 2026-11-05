"""Naive baselines the model is always compared against (brief §4.3, tracked forever).

Pure functions, numpy/stdlib only: the API imports this. Baselines see the same feature
row as the model (data dated <= D-1), so they obey the same point-in-time rule.

- always_up:          "UP", every day.
- same_as_yesterday:  the class of the reference session's own close-to-close move
                      (the `ret_1d` feature); None if that return is missing.
- overnight_gap:      the class of the gap the US futures imply:
                      `GAP_BETA * es_fut_ret_1d` (fallback `sp500_ret_1d`); None if both
                      are missing. Also the model's `gap_expected_pct` (contract §2).
- random:             a draw from the training-set class frequencies, seeded by a stable
                      hash of (target, session_date) so the call is reproducible.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from datetime import date

from app.config import Settings
from app.ml.core import BASELINE_NAMES, CLASSES, class_of, finite

GAP_BETA = 0.5  # Nifty open vs overnight S&P futures move; a transparent rule, not a fit
PROB_FLOOR = 1e-4


def _value(row: Mapping[str, float | None], key: str) -> float | None:
    return finite(row.get(key))  # None for a missing / NaN / inf number, never a default


def gap_expected_pct(row: Mapping[str, float | None]) -> float | None:
    """The overnight-gap rule: 0.5 x ES futures 1-day return, else 0.5 x S&P 500 return."""
    for key in ("es_fut_ret_1d", "sp500_ret_1d"):
        v = _value(row, key)
        if v is not None:
            return GAP_BETA * v
    return None


def stable_seed(target: str, session_date: date) -> int:
    """A process-independent seed for the random baseline (Python's hash() is salted)."""
    digest = hashlib.sha256(f"{target}|{session_date.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def random_call(target: str, session_date: date, class_freq: Mapping[str, float]) -> str | None:
    """Draw a class with the historical class frequencies; None if no frequencies."""
    weights = [max(float(class_freq.get(c, 0.0) or 0.0), 0.0) for c in CLASSES]
    if sum(weights) <= 0:
        return None
    rng = random.Random(stable_seed(target, session_date))
    return rng.choices(list(CLASSES), weights=weights, k=1)[0]


def baseline_calls(
    row: Mapping[str, float | None],
    target: str,
    session_date: date,
    class_freq: Mapping[str, float],
    settings: Settings | None = None,
) -> dict[str, str | None]:
    """The four baseline calls for one session, keyed by `core.BASELINE_NAMES`."""
    thr = settings.market_flat_threshold_pct if settings is not None else None
    ret_1d = _value(row, "ret_1d")
    gap = gap_expected_pct(row)
    calls: dict[str, str | None] = {
        "always_up": "UP",
        "same_as_yesterday": class_of(ret_1d, thr) if ret_1d is not None else None,
        "overnight_gap": class_of(gap, thr) if gap is not None else None,
        "random": random_call(target, session_date, class_freq),
    }
    assert set(calls) == set(BASELINE_NAMES)
    return calls


def baseline_probs(
    name: str, call: str | None, class_freq: Mapping[str, float]
) -> dict[str, float]:
    """The probability vector a baseline is scored with (Brier / log loss).

    A hard call puts (almost) all mass on its class, so its Brier is ~0 when right and ~2
    when wrong. `random` is scored as the climatology forecast it is drawn from, and so is
    any baseline that has no call today; with no frequencies either, the uniform forecast.
    """
    probs: dict[str, float]
    if name == "random" or call is None:
        probs = {c: max(float(class_freq.get(c, 0.0) or 0.0), 0.0) for c in CLASSES}
        if sum(probs.values()) <= 0:
            probs = dict.fromkeys(CLASSES, 1.0 / len(CLASSES))
    else:
        probs = {c: (1.0 if c == call else 0.0) for c in CLASSES}
    return normalise_probs(probs)


def normalise_probs(probs: Mapping[str, float]) -> dict[str, float]:
    """Clip every class to [PROB_FLOOR, 1] and renormalise so UP + FLAT + DOWN = 1."""
    clipped = {c: min(max(float(probs.get(c, 0.0)), PROB_FLOOR), 1.0) for c in CLASSES}
    total = sum(clipped.values())
    return {c: v / total for c, v in clipped.items()}

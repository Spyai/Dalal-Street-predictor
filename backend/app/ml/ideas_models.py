"""The stock-ideas model: one LightGBM binary classifier per horizon, P(target before
stop), trained and validated strictly walk-forward on historical strategy candidates
(brief §4.2, §4.3; Phase 3 contract §2 "Model").

Training rows are the candidates `app.ml.strategies` would have produced on every past
session, all strategies pooled, with the strategy as one-hot features (the setup is
1:1 with the strategy, so the same one-hots carry it). Each row holds the stock's
features at the signal session D and the NIFTY 50 context of D (`mkt_*` columns) from
`app.ml.ideas_features.stock_features`, and the triple-barrier label of the horizon from
`ideas_features.build_candidate_frame` (`label`: 1 = target_1 touched before the stop
within `days` sessions after entering at the next open, else 0; `expired` and
`realised_return_pct` say when neither barrier was touched and what the move to the exit
was; the `horizon` column names the horizon the labels belong to). A frame that carries
several horizons side by side as `label_<horizon>` / `expired_<horizon>` /
`realised_<horizon>_pct` is accepted too. `training_data(db, horizon)` builds the frame
from the database along that path; `strategies.candidate_frame` decides which rows are
candidates.

Validation walks forward over SESSIONS, not rows (`ideas_features.walk_forward_folds`):
fold j fits on every candidate of the sessions [0, k - embargo) and scores the candidates
of sessions [k, k + step), with an embargo of `days` sessions (the horizon) between the
last training session and the first test session, and a row whose label resolved on or
after the test start (its stock skipped sessions) is purged from that fold's training set
as well, so no training label can have been resolved by a path that overlaps the test
rows' features. Two knobs are chosen out of
sample, sequentially, exactly as in `app.ml.models` (rounds scored every ROUND_STEP,
shrinkage towards the fold's own training base rate = the candidates' hit rate); a fold's
rows only ever inform the choice for later folds, and the final fit uses the pair chosen
on every fold.

Reported out of sample: hit rate and Brier/log loss/calibration of the probabilities,
per-strategy hit rates, the `all_candidates` baseline (every candidate's hit rate, and the
Brier of forecasting the base rate), the `random_universe` baseline (same number of
random universe stocks per session with the same barriers, seeded) when the labelled
universe is given, and the expectancy after costs of the ideas that would have cleared
the bar. Everything the frame does not carry is reported as NaN/absent, never estimated.

LightGBM is imported inside functions only; the API may import this module.
"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.logging import get_logger
from app.ml import ideas_features as idf
from app.ml import models, registry
from app.ml.baselines import PROB_FLOOR, stable_seed
from app.ml.core import (
    Driver,
    FeatureRow,
    HorizonSpec,
    TrainResult,
    WalkForwardFold,
    WalkForwardMetrics,
    finite,
    horizons,
)
from app.ml.costs import EXPIRY_RETURN_PCT, round_trip_cost_pct
from app.ml.strategies import (
    IDEA_FEATURE_LABELS,
    ONE_HOT_PREFIX,
    RANK_COLUMN,
    STRATEGY_NAMES,
    candidate_frame,
    normalise_frame,
    one_hot_names,
    one_hot_row,
    one_hots,
)
from app.models import DailyBar, ModelVersion
from app.services.calendar import TradingCalendar
from app.services.instruments import universe_instruments
from app.services.market_data import daily_frame

if TYPE_CHECKING:
    from lightgbm import Booster

log = get_logger(__name__)

MODEL_NAME_PREFIX = "ideas_"  # ModelVersion.name: ideas_swing | ideas_positional
# NIFTY 50 context columns (from `features.build_frame(db, "NIFTY 50")` row D), prefixed so
# they never collide with the stock's own ret_5d / ret_20d / rvol_20d.
MARKET_CONTEXT: tuple[str, ...] = (
    "ret_5d",
    "ret_20d",
    "vix_level",
    "pct_above_200dma",
    "adv_decl_ratio",
    "rvol_20d",
)
MARKET_PREFIX = "mkt_"
ID_COLUMNS: tuple[str, str] = ("symbol", "session_date")
# Never features: identifiers, text, raw price levels (not comparable across stocks), the
# feature pipeline's META_COLUMNS (instrument_id, close, n_bars, eligible) and the horizon.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "symbol",
        "exchange",
        "session_date",
        "instrument_id",
        "strategy",
        "setup",
        "reasons",
        "invalidation",
        "horizon",
        "horizon_days",
        "n_bars",
        "eligible",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "delivery_pct",
        *idf.META_COLUMNS,
    }
)
# Anything about the outcome of a row (any horizon) is a label, never a feature: the
# pipeline's LABEL_COLUMNS by name, and these prefixes for the per-horizon spelling.
LABEL_NAMES: frozenset[str] = frozenset(
    {"label", "expired", "realised_return_pct", *idf.LABEL_COLUMNS}
)
LABEL_PREFIXES: tuple[str, ...] = (
    "label_",
    "expired_",
    "realised_",
    "outcome_",
    "exit_",
    "entry_",
    "days_held",
    "status",
    "hit_",
    "touched_",
    "target_price",
    "stop_price",
    "target_pct",
    "stop_pct",
    "window_closed",
)

MAX_ROUNDS = 400
ROUND_STEP = models.ROUND_STEP
FIRST_FOLD_ROUNDS = 50
FIRST_FOLD_SHRINK = 0.5
SHRINK_GRID = models.SHRINK_GRID
SELECTED_PARAM_KEYS = models.SELECTED_PARAM_KEYS
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "num_leaves": 7,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "n_estimators": MAX_ROUNDS,  # ceiling of the round search; the artifact records the choice
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "seed": 42,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
_ARTIFACT_CACHE_SIZE = 8
_ARTIFACT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


# --------------------------------------------------------------------------- names


def model_name(horizon: str) -> str:
    """`ModelVersion.name` for a horizon: "ideas_swing" / "ideas_positional"."""
    return f"{MODEL_NAME_PREFIX}{horizon}"


def model_target(settings: Settings | None = None) -> str:
    """`ModelVersion.target` for the ideas models: the configured universe, upper-cased."""
    s = settings or get_settings()
    return s.universe.strip().upper() or "UNIVERSE"


def label_column(horizon: str) -> str:
    """The per-horizon spelling of the label column (`label_swing`)."""
    return f"label_{horizon}"


def expired_column(horizon: str) -> str:
    return f"expired_{horizon}"


def realised_column(horizon: str) -> str:
    return f"realised_{horizon}_pct"


class LabelColumns(NamedTuple):
    """Which columns of a frame hold the horizon's label, expiry flag and realised return
    (the last two None when the frame has none)."""

    label: str
    expired: str | None
    realised: str | None


def label_columns(frame: pd.DataFrame, horizon: str) -> LabelColumns:
    """Resolve the label columns for `horizon`: the per-horizon spelling
    (`label_<horizon>`, `expired_<horizon>`, `realised_<horizon>_pct`) when present, else
    the feature pipeline's `label` / `expired` / `realised_return_pct`
    (`ideas_features.build_candidate_frame`, one horizon per frame). Raises ValueError when
    the frame has neither."""
    cols = set(map(str, frame.columns))
    suffixed = label_column(horizon)
    if suffixed in cols:
        return LabelColumns(
            suffixed,
            expired_column(horizon) if expired_column(horizon) in cols else None,
            realised_column(horizon) if realised_column(horizon) in cols else None,
        )
    if "label" in cols:
        return LabelColumns(
            "label",
            "expired" if "expired" in cols else None,
            "realised_return_pct" if "realised_return_pct" in cols else None,
        )
    raise ValueError(
        f"candidate frame has neither {suffixed!r} nor 'label' for horizon {horizon!r} "
        "(ideas_features.build_candidate_frame writes 'label')"
    )


def horizon_spec(horizon: str, settings: Settings | None = None) -> HorizonSpec:
    specs = horizons(settings)
    if horizon not in specs:
        raise ValueError(f"unknown horizon {horizon!r}; known: {sorted(specs)}")
    return specs[horizon]


# --------------------------------------------------------------------------- frame helpers


def is_label_column(name: str) -> bool:
    return name in LABEL_NAMES or any(name.startswith(p) for p in LABEL_PREFIXES)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The feature columns of a prepared candidate frame: every numeric column that is
    not an identifier, text, raw price level or label, in frame order, then the strategy
    one-hots in STRATEGY_NAMES order (added by `prepare_frame`)."""
    hots = set(one_hot_names())
    out: list[str] = []
    for col in map(str, frame.columns):
        if col in NON_FEATURE_COLUMNS or col in hots or is_label_column(col):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(
            frame[col]
        ):
            continue
        out.append(col)
    out.extend(h for h in one_hot_names() if h in frame.columns)
    return out


def prepare_frame(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Labelled candidate rows for `horizon`, flat, sorted by session / symbol / strategy,
    with the strategy one-hots added and the label as float in `label_columns().label`.

    A `horizon` column (ideas_features.build_candidate_frame) keeps only that horizon's
    rows: labels of another horizon are never borrowed. Raises ValueError without
    `strategy` or the label, for unknown strategies, or when no row is for `horizon`.
    """
    df = normalise_frame(frame)
    if "strategy" not in df.columns:
        raise ValueError("candidate frame needs a 'strategy' column (strategies.candidate_frame)")
    if "horizon" in df.columns:
        present = sorted(set(df["horizon"].dropna().astype(str)))
        df = df[df["horizon"].astype(str) == horizon]
        if df.empty and present:
            raise ValueError(f"candidate frame holds horizons {present}, not {horizon!r}")
    lab = label_columns(df, horizon).label
    unknown = sorted(set(df["strategy"].dropna().astype(str)) - set(STRATEGY_NAMES))
    if unknown:
        raise ValueError(f"unknown strategies in frame: {unknown}")
    labels = pd.to_numeric(df[lab], errors="coerce")
    df = df[labels.notna() & df["strategy"].notna()].copy()
    df[lab] = pd.to_numeric(df[lab], errors="coerce").astype(float)
    df["strategy"] = df["strategy"].astype(str)
    hots = one_hots(df["strategy"])
    for col in hots.columns:
        df[col] = hots[col].to_numpy()
    order = {s: i for i, s in enumerate(STRATEGY_NAMES)}
    df["_order"] = df["strategy"].map(order)
    df = df.sort_values(["session_date", "symbol", "_order"], kind="stable").drop(columns="_order")
    return df.reset_index(drop=True)


def market_context_row(nifty_row: Mapping[str, object]) -> dict[str, float | None]:
    """The `mkt_*` features from a NIFTY 50 feature row (`features.live_row` snapshot)."""
    return {f"{MARKET_PREFIX}{k}": finite(nifty_row.get(k)) for k in MARKET_CONTEXT}


def live_features(
    stock_row: Mapping[str, object] | pd.Series,
    market_row: Mapping[str, object] | None,
    strategy: str,
) -> FeatureRow:
    """One candidate's live feature row: the stock's numeric features at D, the NIFTY 50
    context (prefixed; skipped when None because the stock row already carries `mkt_*`)
    and the strategy one-hots. Values go through `finite()` (None when missing)."""
    items = stock_row.items() if isinstance(stock_row, Mapping) else stock_row.to_dict().items()
    out: FeatureRow = {}
    for key, value in items:
        name = str(key)
        if name in NON_FEATURE_COLUMNS or is_label_column(name) or name.startswith(ONE_HOT_PREFIX):
            continue
        out[name] = finite(value)
    if market_row is not None:
        out.update(market_context_row(market_row))
    out.update(one_hot_row(strategy))
    return out


def _as_date(x: object) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(str(x)).date()


def _exit_ordinals(df: pd.DataFrame) -> np.ndarray | None:
    """Each row's exit session as a proleptic ordinal (NaN when unknown), from the
    `exit_session` column the label pipeline writes; None when the frame has none."""
    if "exit_session" not in df.columns:
        return None
    out = np.full(len(df), np.nan)
    for i, v in enumerate(df["exit_session"].tolist()):
        if v is None or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
            continue
        try:
            out[i] = float(_as_date(v).toordinal())
        except (TypeError, ValueError):
            continue
    return out


def _optional_float(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _optional_bool(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.to_numpy(dtype=bool)
    num = pd.to_numeric(raw, errors="coerce")
    return (num.fillna(0.0) > 0.5).to_numpy(dtype=bool)


# --------------------------------------------------------------------------- binary scores


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    """Mean (p - y)^2; 0 for a perfect forecast, 0.25 for a coin-flip forecast."""
    pp = np.asarray(p, dtype=float).reshape(-1)
    if pp.size == 0:
        return math.nan
    return float(np.mean((pp - np.asarray(y, dtype=float).reshape(-1)) ** 2))


def log_loss_binary(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    pp = np.clip(np.asarray(p, dtype=float).reshape(-1), eps, 1.0 - eps)
    if pp.size == 0:
        return math.nan
    yy = np.asarray(y, dtype=float).reshape(-1)
    return float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))


def hit_rate(y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float).reshape(-1)
    return float(yy.mean()) if yy.size else math.nan


def accuracy_binary(p: np.ndarray, y: np.ndarray) -> float:
    pp = np.asarray(p, dtype=float).reshape(-1)
    if pp.size == 0:
        return math.nan
    return float(np.mean((pp >= 0.5) == (np.asarray(y, dtype=float).reshape(-1) >= 0.5)))


def calibration_bins_binary(
    p: np.ndarray, y: np.ndarray, bins: int = 5
) -> list[dict[str, float | int]]:
    """Reliability table over equal-width bins of P(target) on [0, 1]; empty bins are left
    out. Rows: {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}."""
    pp = np.asarray(p, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if pp.size == 0 or bins < 1:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, pp, side="right") - 1, 0, bins - 1)
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
                "predicted_mean": float(pp[mask].mean()),
                "observed": float(yy[mask].mean()),
            }
        )
    return out


def lift_table(
    p: np.ndarray,
    y: np.ndarray,
    net: np.ndarray | None = None,
    bins: int = 10,
    groups: np.ndarray | None = None,
) -> list[dict[str, float | int | None]]:
    """Hit rate (and the mean of `net`, when given) by decile of P(target), lowest
    probabilities first (decile 1) to highest (decile `bins`).

    Without `groups`: equal-count bins over all rows. With `groups` (one key per row, the
    session in training): each row is ranked among the rows of its own group and its decile
    is that within-group percentile, so the table measures the ranking at the moment of
    choice (which of today's candidates to show) and not the drift of the base rate between
    periods, which a global split would show as lift or as its opposite. Rows: {"decile",
    "n", "p_mean", "hit_rate", "expectancy_pct"}; the last is None without `net` or when
    every value in the bin is NaN. Empty for no rows.
    """
    pp = np.asarray(p, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if pp.size == 0 or bins < 1:
        return []
    decile = np.zeros(pp.size, dtype=int)
    if groups is None:
        for d, idx in enumerate(np.array_split(np.argsort(pp, kind="stable"), bins), start=1):
            decile[idx] = d
    else:
        g = np.asarray(groups).reshape(-1)
        if g.size != pp.size:
            raise ValueError("groups must have one key per row")
        for key in np.unique(g):
            members = np.flatnonzero(g == key)
            ordered = members[np.argsort(pp[members], kind="stable")]
            pct = (np.arange(ordered.size) + 0.5) / ordered.size
            decile[ordered] = np.minimum((pct * bins).astype(int) + 1, bins)
    net_arr = None if net is None else np.asarray(net, dtype=float).reshape(-1)
    out: list[dict[str, float | int | None]] = []
    for d in range(1, bins + 1):
        idx = np.flatnonzero(decile == d)
        if idx.size == 0:
            continue
        row: dict[str, float | int | None] = {
            "decile": d,
            "n": int(idx.size),
            "p_mean": finite(pp[idx].mean()),
            "hit_rate": finite(yy[idx].mean()),
            "expectancy_pct": None,
        }
        if net_arr is not None:
            vals = net_arr[idx]
            vals = vals[~np.isnan(vals)]
            row["expectancy_pct"] = finite(vals.mean()) if vals.size else None
        out.append(row)
    return out


def shrink_binary(p: np.ndarray, shrink: float, base_rate: float | None) -> np.ndarray:
    """(1 - shrink) * p + shrink * base_rate, clipped to [PROB_FLOOR, 1 - PROB_FLOOR].

    `base_rate` is the training candidates' hit rate. Raises ValueError when shrinkage is
    asked for without one: a missing number is never replaced by a made-up prior.
    """
    if not 0.0 <= shrink < 1.0:
        raise ValueError(f"shrink must be in [0, 1), got {shrink}")
    arr = np.asarray(p, dtype=float)
    if shrink > 0.0:
        if base_rate is None or math.isnan(base_rate):
            raise ValueError("shrinkage needs the training base rate")
        arr = (1.0 - shrink) * arr + shrink * base_rate
    return np.clip(arr, PROB_FLOOR, 1.0 - PROB_FLOOR)


# --------------------------------------------------------------------------- metrics


@dataclass
class IdeaModelMetrics(WalkForwardMetrics):
    """`WalkForwardMetrics` for a binary P(target) model, plus what an ideas desk asks.

    Inherited fields mean: `accuracy` = share of rows where (p >= 0.5) matches the label;
    `class_frequencies` = {"target", "stop", "expired"} shares of all labelled rows (no
    "expired" when the frame has no expired column); `baselines` = {"all_candidates":
    {n, hit_rate, brier}, "random_universe": {n, hit_rate} or {n: 0}}; `by_confidence` =
    rows for "selected" (cleared the p / EV bar) and "rejected"; `calibration` = bins of p.

    hit_rate / n_selected: out-of-sample hit rate of the candidates that cleared the bar
    (p >= ideas_min_p_target and, when the frame carries atr14_pct, EV >= ideas_min_ev_pct).
    hit_rate_ranked / n_ranked: the same after one-per-symbol and the per-session cap, the
    closest replay of what the desk would have shown.
    expectancy_*_pct: mean net return (%) per idea after the round-trip cost: +T on a hit,
    -S on a stop, the realised close-to-close move on expiry (0 when the frame has no
    realised column). NaN when the frame has no atr14_pct to size T and S from.
    hit_rate_top / n_top / expectancy_top_pct: the top `ideas_max_per_horizon` candidates
    a session by EV (one per symbol), bar ignored: what the desk would have shown had it
    always shown ideas. Compare with the all_candidates baseline: a ranking that carries
    information beats it.
    lift: the out-of-sample rows by decile of P(target) within their own session (1 = the
    candidates the model liked least that day, 10 = the most): n, p_mean, hit_rate,
    expectancy_pct. A useful ranking climbs from decile 1 to 10; a flat table means the
    probabilities carry nothing beyond the base rate at the moment of choice.
    """

    horizon: str = ""
    horizon_days: int = 0
    hit_rate: float = math.nan
    n_selected: int = 0
    hit_rate_ranked: float = math.nan
    n_ranked: int = 0
    per_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    expectancy_after_costs_pct: float = math.nan
    expectancy_ranked_pct: float = math.nan
    expectancy_all_candidates_pct: float = math.nan
    hit_rate_top: float = math.nan
    n_top: int = 0
    expectancy_top_pct: float = math.nan
    lift: list[dict[str, float | int | None]] = field(default_factory=list)
    cost_round_trip_pct: float = math.nan
    n_sessions: int = 0
    embargo_sessions: int = 0
    fold_params: list[dict[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        out = super().as_dict()
        out.update(
            {
                "horizon": self.horizon,
                "horizon_days": self.horizon_days,
                "hit_rate": finite(self.hit_rate),
                "n_selected": self.n_selected,
                "hit_rate_ranked": finite(self.hit_rate_ranked),
                "n_ranked": self.n_ranked,
                "per_strategy": {
                    k: {m: finite(v) for m, v in d.items()} for k, d in self.per_strategy.items()
                },
                "expectancy_after_costs_pct": finite(self.expectancy_after_costs_pct),
                "expectancy_ranked_pct": finite(self.expectancy_ranked_pct),
                "expectancy_all_candidates_pct": finite(self.expectancy_all_candidates_pct),
                "hit_rate_top": finite(self.hit_rate_top),
                "n_top": self.n_top,
                "expectancy_top_pct": finite(self.expectancy_top_pct),
                "lift": [dict(r) for r in self.lift],
                "cost_round_trip_pct": finite(self.cost_round_trip_pct),
                "n_sessions": self.n_sessions,
                "embargo_sessions": self.embargo_sessions,
                "fold_params": [dict(fp) for fp in self.fold_params],
                "notes": list(self.notes),
            }
        )
        return out


# --------------------------------------------------------------------------- LightGBM


def _booster_params(params: Mapping[str, object]) -> dict[str, object]:
    return {k: v for k, v in params.items() if k not in SELECTED_PARAM_KEYS}


def _max_rounds(params: Mapping[str, object]) -> int:
    raw = params.get("n_estimators", MAX_ROUNDS)
    rounds = int(raw) if isinstance(raw, int | float | str) else MAX_ROUNDS
    if rounds < 1:
        raise ValueError("n_estimators must be positive")
    return rounds


def _shrink_grid(params: Mapping[str, object]) -> tuple[tuple[float, ...], float]:
    fixed = params.get("shrink")
    if fixed is None:
        return SHRINK_GRID, FIRST_FOLD_SHRINK
    a = float(fixed)  # type: ignore[arg-type]
    if not 0.0 <= a < 1.0:
        raise ValueError(f"shrink must be in [0, 1), got {a}")
    return (a,), a


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    rounds: int,
) -> Booster:
    import lightgbm as lgb

    data = lgb.Dataset(x, label=y, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train(_booster_params(params), data, num_boost_round=rounds)


def _p(booster: Booster, x: np.ndarray, rounds: int | None = None) -> np.ndarray:
    raw = booster.predict(x, num_iteration=rounds)
    return np.clip(np.asarray(raw, dtype=float).reshape(-1), PROB_FLOOR, 1.0 - PROB_FLOOR)


# --------------------------------------------------------------------------- training


def _random_universe_baseline(
    universe: pd.DataFrame | None,
    horizon: str,
    counts: Mapping[date, int],
) -> dict[str, float]:
    """Hit rate of `counts[session]` random labelled universe stocks per out-of-sample
    session, with the same barriers (the universe frame's own label), seeded per
    (horizon, session). {"n": 0} when no universe is given."""
    if universe is None:
        return {"n": 0}
    u = normalise_frame(universe)
    try:
        lab = label_columns(u, horizon).label
    except ValueError as exc:
        raise ValueError(f"universe frame has no {label_column(horizon)!r} column") from exc
    labels = pd.to_numeric(u[lab], errors="coerce")
    u = u[labels.notna()].assign(**{lab: labels[labels.notna()].astype(float)})
    by_session = {d: g[lab].to_numpy(dtype=float) for d, g in u.groupby("session_date")}
    hits = 0.0
    n = 0
    for session, m in sorted(counts.items()):
        pool = by_session.get(session)
        if pool is None or pool.size == 0 or m <= 0:
            continue
        rng = np.random.default_rng(stable_seed(model_name(horizon), session))
        take = min(int(m), int(pool.size))
        draw = rng.choice(pool, size=take, replace=False)
        hits += float(draw.sum())
        n += take
    return {"n": n, "hit_rate": hits / n} if n else {"n": 0}


def _ranked_mask(
    selected: np.ndarray,
    ev: np.ndarray,
    p: np.ndarray,
    sessions: np.ndarray,
    symbols: np.ndarray,
    cap: int,
) -> np.ndarray:
    """Among selected rows: one per symbol per session (best EV), at most `cap` a session."""
    out = np.zeros(selected.shape[0], dtype=bool)
    if cap <= 0:
        return out
    idx = np.flatnonzero(selected)
    if idx.size == 0:
        return out
    frame = pd.DataFrame(
        {
            "i": idx,
            "session": sessions[idx],
            "symbol": symbols[idx],
            "ev": np.where(np.isnan(ev[idx]), -np.inf, ev[idx]),
            "p": p[idx],
        }
    )
    frame = frame.sort_values(
        ["session", "ev", "p", "symbol"], ascending=[True, False, False, True]
    )
    frame = frame.drop_duplicates(["session", "symbol"], keep="first")
    frame = frame.groupby("session", sort=False).head(cap)
    out[frame["i"].to_numpy(dtype=int)] = True
    return out


def train_walk_forward(
    frame: pd.DataFrame,
    horizon: str,
    *,
    params: Mapping[str, object] | None = None,
    settings: Settings | None = None,
    universe: pd.DataFrame | None = None,
) -> TrainResult:
    """Strict session-level walk-forward with an embargo, then a final fit on every label.

    `frame`: labelled candidate rows (see the module docstring); `universe`: optional
    labelled universe stock-sessions (`symbol`, `session_date`, `label` or
    `label_<horizon>`) for the random_universe baseline. Folds come from
    `ideas_features.walk_forward_folds(sessions, min_train_sessions, step, embargo)`: for
    k = min_train + embargo, + step, ... over the sorted labelled sessions, fit on sessions
    [0, k - embargo), score sessions [k, k + step) with the (rounds, shrink) chosen on
    earlier folds (defaults for the first), then add this fold's grid scores to the pool.
    min_train = settings.wf_min_train_days, step = settings.wf_step_days, embargo = the
    horizon's days. Raises ValueError with fewer than min_train + embargo + step labelled
    sessions, or without the strategy / label columns.

    `params` override DEFAULT_PARAMS; "n_estimators" is the round ceiling, "shrink" fixes
    the shrinkage. The returned params carry the chosen "n_estimators" and "shrink"; the
    metrics are an `IdeaModelMetrics`.
    """
    s = settings or get_settings()
    spec = horizon_spec(horizon, s)
    min_train, step, embargo = int(s.wf_min_train_days), int(s.wf_step_days), int(spec.days)
    if min_train < 1 or step < 1 or embargo < 0:
        raise ValueError("wf_min_train_days and wf_step_days must be positive")
    effective: dict[str, object] = {**DEFAULT_PARAMS, **dict(params or {})}
    max_rounds = _max_rounds(effective)
    rounds_grid = models.round_grid(max_rounds)
    shrink_grid, first_shrink = _shrink_grid(effective)
    first_choice = (min(FIRST_FOLD_ROUNDS, max_rounds), first_shrink)

    df = prepare_frame(frame, horizon)
    n = len(df)
    if n == 0:
        raise ValueError(f"no labelled candidates for horizon {horizon!r}")
    feature_names = feature_columns(df)
    if not feature_names:
        raise ValueError("candidate frame has no feature columns")
    cols = label_columns(df, horizon)
    x = models.frame_matrix(df, feature_names)
    y = (df[cols.label].to_numpy(dtype=float) >= 0.5).astype(int)
    session_dates = [_as_date(d) for d in df["session_date"]]
    sessions: list[date] = sorted(set(session_dates))
    n_sessions = len(sessions)
    if n_sessions < min_train + embargo + step:
        raise ValueError(
            f"{n_sessions} labelled sessions; walk-forward needs at least "
            f"{min_train + embargo + step} (wf_min_train_days={min_train} + embargo={embargo} "
            f"+ wf_step_days={step})"
        )
    pos = {d: i for i, d in enumerate(sessions)}
    sess_idx = np.array([pos[d] for d in session_dates], dtype=int)
    symbols = df["symbol"].to_numpy(dtype=object)
    strategies = df["strategy"].to_numpy(dtype=object)
    expired = _optional_bool(df, cols.expired) if cols.expired else None
    realised = _optional_float(df, cols.realised) if cols.realised else None
    atr = _optional_float(df, "atr14_pct")
    cost = round_trip_cost_pct(s)
    notes: list[str] = []
    if atr is None:
        notes.append("no atr14_pct column: EV bar and expectancy not computed (NaN)")
    if expired is None:
        notes.append("no expired column: every non-hit counts as a stop in the expectancy")
    elif realised is None:
        notes.append(
            f"no realised column: expired ideas count as {EXPIRY_RETURN_PCT:g}% in the expectancy"
        )

    exit_ord = _exit_ordinals(df)
    n_purged_extra = 0
    oos_p = np.full(n, np.nan)
    oos = np.zeros(n, dtype=bool)
    base_used = np.full(n, np.nan)
    loss_sum = np.zeros((len(rounds_grid), len(shrink_grid)))
    n_scored = 0
    folds: list[WalkForwardFold] = []
    fold_params: list[dict[str, float]] = []
    plan = idf.walk_forward_folds(
        sessions, min_train_sessions=min_train, step=step, embargo=embargo
    )
    for fold in plan:
        train = sess_idx <= pos[fold.train_end]
        if exit_ord is not None:
            # Purge by the row's own exit: a label that resolved on or after the test
            # start (a bar gap stretched its window past the session embargo) never trains
            # this fold.
            late = train & (exit_ord >= float(fold.test_start.toordinal()))
            n_purged_extra += int(late.sum())
            train &= ~late
        test = (sess_idx >= pos[fold.test_start]) & (sess_idx <= pos[fold.test_end])
        n_test = int(test.sum())
        if n_test == 0 or not train.any():
            continue
        rounds, shrink = models.select_rounds_and_shrink(
            loss_sum, n_scored, rounds_grid, shrink_grid, first_choice
        )
        base = float(y[train].mean())
        booster = _fit(x[train], y[train], feature_names, effective, max_rounds)
        xt, yt = x[test], y[test]
        p = shrink_binary(_p(booster, xt, rounds), shrink, base)
        oos_p[test] = p
        oos[test] = True
        base_used[test] = base
        folds.append(
            WalkForwardFold(
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                n_test=n_test,
                accuracy=accuracy_binary(p, yt),
                brier=brier_binary(p, yt),
                log_loss=log_loss_binary(p, yt),
            )
        )
        fold_params.append({"n_estimators": rounds, "shrink": shrink})
        grid = np.stack([_p(booster, xt, r) for r in rounds_grid])  # (K, n_test)
        for j, a in enumerate(shrink_grid):
            mixed = shrink_binary(grid, a, base)
            for i in range(len(rounds_grid)):
                loss_sum[i, j] += log_loss_binary(mixed[i], yt) * n_test
        n_scored += n_test
    if not folds:
        raise ValueError("walk-forward produced no out-of-sample fold")
    if n_purged_extra:
        notes.append(
            f"{n_purged_extra} training row(s) purged beyond the embargo: their label "
            "resolved inside a test window (bar gaps after the signal)"
        )

    po, yo = oos_p[oos], y[oos]
    n_oos = int(oos.sum())
    # The bar, the ranking and the expectancy replayed on the out-of-sample rows. The
    # gross return of a row is its realised move (exit vs entry, gaps included) when the
    # frame has it; otherwise +T on a hit, -S on a stop and EXPIRY_RETURN_PCT on expiry.
    if atr is not None:
        t_pct = spec.target_atr * atr[oos]
        s_pct = spec.stop_atr * atr[oos]
        ev = po * t_pct - (1.0 - po) * s_pct - cost
        selected = (po >= s.ideas_min_p_target) & (ev >= s.ideas_min_ev_pct)
        if expired is None:
            approx = np.where(yo == 1, t_pct, -s_pct)
        else:
            approx = np.where(yo == 1, t_pct, np.where(expired[oos], EXPIRY_RETURN_PCT, -s_pct))
        gross = (
            np.where(np.isfinite(realised[oos]), realised[oos], approx)
            if realised is not None
            else approx
        )
        net = gross - cost
    else:
        ev = np.full(n_oos, np.nan)
        selected = po >= s.ideas_min_p_target
        net = np.full(n_oos, np.nan)
    cap = int(s.ideas_max_per_horizon)
    ranked = _ranked_mask(selected, ev, po, sess_idx[oos], symbols[oos], cap)
    # The same replay with the bar ignored: the best `cap` a session, every session.
    top = _ranked_mask(np.ones(n_oos, dtype=bool), ev, po, sess_idx[oos], symbols[oos], cap)

    def _mean(a: np.ndarray, mask: np.ndarray) -> float:
        vals = a[mask]
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if vals.size else math.nan

    per_strategy: dict[str, dict[str, float]] = {}
    strat_oos = strategies[oos]
    for name in STRATEGY_NAMES:
        m = strat_oos == name
        if not m.any():
            continue
        entry: dict[str, float] = {
            "n": int(m.sum()),
            "hit_rate": hit_rate(yo[m]),
            "brier": brier_binary(po[m], yo[m]),
            "n_selected": int((m & selected).sum()),
        }
        if (m & selected).any():
            entry["selected_hit_rate"] = hit_rate(yo[m & selected])
            entry["expectancy_after_costs_pct"] = _mean(net, m & selected)
        per_strategy[name] = entry

    counts: dict[date, int] = {}
    for i in np.flatnonzero(oos).tolist():
        session = sessions[int(sess_idx[i])]
        counts[session] = counts.get(session, 0) + 1
    baselines: dict[str, dict[str, float]] = {
        "all_candidates": {
            "n": n_oos,
            "hit_rate": hit_rate(yo),
            "brier": brier_binary(base_used[oos], yo),
        },
        "random_universe": _random_universe_baseline(universe, horizon, counts),
    }
    freq: dict[str, float] = {"target": hit_rate(y)}
    if expired is not None:
        freq["expired"] = float(np.mean((y == 0) & expired))
        freq["stop"] = float(np.mean((y == 0) & ~expired))
    else:
        freq["not_target"] = 1.0 - freq["target"]
    by_conf: list[dict[str, float | str | int]] = []
    for label, mask in (("selected", selected), ("rejected", ~selected)):
        row: dict[str, float | str | int] = {"confidence": label, "n": int(mask.sum())}
        if mask.any():
            row["hit_rate"] = hit_rate(yo[mask])
        by_conf.append(row)

    wf = IdeaModelMetrics(
        n_oos=n_oos,
        accuracy=accuracy_binary(po, yo),
        brier=brier_binary(po, yo),
        log_loss=log_loss_binary(po, yo),
        class_frequencies=freq,
        baselines=baselines,
        by_confidence=by_conf,
        calibration=calibration_bins_binary(po, yo),
        folds=folds,
        horizon=horizon,
        horizon_days=spec.days,
        hit_rate=hit_rate(yo[selected]) if selected.any() else math.nan,
        n_selected=int(selected.sum()),
        hit_rate_ranked=hit_rate(yo[ranked]) if ranked.any() else math.nan,
        n_ranked=int(ranked.sum()),
        per_strategy=per_strategy,
        expectancy_after_costs_pct=_mean(net, selected),
        expectancy_ranked_pct=_mean(net, ranked),
        expectancy_all_candidates_pct=_mean(net, np.ones(n_oos, dtype=bool)),
        hit_rate_top=hit_rate(yo[top]) if top.any() else math.nan,
        n_top=int(top.sum()),
        expectancy_top_pct=_mean(net, top),
        lift=lift_table(po, yo, net, groups=sess_idx[oos]),
        cost_round_trip_pct=cost,
        n_sessions=n_sessions,
        embargo_sessions=embargo,
        fold_params=fold_params,
        notes=notes,
    )

    rounds, shrink = models.select_rounds_and_shrink(
        loss_sum, n_scored, rounds_grid, shrink_grid, first_choice
    )
    effective["n_estimators"] = rounds
    effective["shrink"] = shrink
    classifier = _fit(x, y, feature_names, effective, rounds)
    artifact = json.dumps(
        {
            "classifier": classifier.model_to_string(),
            "feature_names": feature_names,
            "params": effective,
            "base_rate": float(y.mean()),
            "horizon": horizon,
            "horizon_days": spec.days,
            "target_atr": spec.target_atr,
            "stop_atr": spec.stop_atr,
            "objective": "binary",
            "label_version": idf.LABEL_VERSION,
        },
        sort_keys=True,
    )
    return TrainResult(
        artifact=artifact,
        feature_names=feature_names,
        params=effective,
        metrics=wf,
        trained_through=sessions[-1],
        train_rows=n,
    )


# --------------------------------------------------------------------------- inference


def load_artifact(artifact: str) -> dict[str, Any]:
    """The Booster rebuilt from the artifact JSON, cached in-process by sha256.

    Returns {"classifier": Booster, "feature_names", "hash", "shrink", "base_rate",
    "horizon"}."""
    key = models.artifact_hash(artifact)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        _ARTIFACT_CACHE.move_to_end(key)
        return cached
    import lightgbm as lgb

    payload = json.loads(artifact)
    booster = lgb.Booster(model_str=payload["classifier"])
    names = payload.get("feature_names") or booster.feature_name()
    params = payload.get("params") or {}
    raw_shrink = params.get("shrink", 0.0) if isinstance(params, Mapping) else 0.0
    loaded: dict[str, Any] = {
        "classifier": booster,
        "feature_names": [str(f) for f in names],
        "hash": key,
        "shrink": float(raw_shrink or 0.0),
        "base_rate": finite(payload.get("base_rate")),
        "horizon": str(payload.get("horizon", "")),
    }
    _ARTIFACT_CACHE[key] = loaded
    while len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_SIZE:
        _ARTIFACT_CACHE.popitem(last=False)
    return loaded


class IdeaPrediction(NamedTuple):
    """P(target before stop) with its drivers: the top `drivers_top_n` for the card and
    every non-zero contribution for the "all factors" view (same order, largest first)."""

    p_target: float
    drivers: list[Driver]
    all_drivers: list[Driver]


def predict(
    artifact: str,
    feature_names: list[str],
    row: FeatureRow,
    *,
    settings: Settings | None = None,
) -> IdeaPrediction:
    """P(target_1 before stop) for one candidate's live feature row (`live_features`).

    The classifier's probability is shrunk towards the training base rate by the
    artifact's `shrink` (chosen out of sample). Drivers come from `pred_contrib` on the
    raw score: `drivers` is the top `settings.drivers_top_n` by |contribution|,
    `all_drivers` every feature that moved the score. Raises ValueError if
    `feature_names` differ from the names the artifact was trained with.
    """
    s = settings or get_settings()
    art = load_artifact(artifact)
    trained = list(art["classifier"].feature_name())
    if list(feature_names) != trained:
        raise ValueError(
            f"feature names do not match the model: got {len(feature_names)}, "
            f"trained with {len(trained)}"
        )
    x = models.row_vector(feature_names, row)
    p = float(shrink_binary(_p(art["classifier"], x), art["shrink"], art["base_rate"])[0])
    contrib = np.asarray(art["classifier"].predict(x, pred_contrib=True), dtype=float).reshape(
        1, len(feature_names) + 1
    )[0, :-1]  # the last column is the bias term
    all_drivers = drivers(feature_names, row, contrib)
    return IdeaPrediction(
        p_target=p, drivers=all_drivers[: max(int(s.drivers_top_n), 0)], all_drivers=all_drivers
    )


# --------------------------------------------------------------------------- drivers


def drivers(
    feature_names: Sequence[str],
    row: Mapping[str, float | None],
    contrib: np.ndarray,
    n: int | None = None,
) -> list[Driver]:
    """Features that moved P(target), largest |contribution| first; at most `n` (all when
    None). A zero contribution is not a driver."""
    c = np.asarray(contrib, dtype=float).reshape(-1)
    order = np.argsort(-np.abs(c), kind="stable")
    out: list[Driver] = []
    for i in order:
        if (n is not None and len(out) >= n) or c[i] == 0.0 or math.isnan(c[i]):
            break
        name = feature_names[int(i)]
        value = finite(row.get(name))
        out.append(
            Driver(
                feature=name,
                value=value,
                contribution=float(c[i]),
                text=driver_text(name, value, float(c[i])),
            )
        )
    return out


def feature_label(name: str) -> str:
    """Plain-English label: the feature pipeline's (`ideas_features.FEATURE_LABELS`), then
    the columns the strategies add (`IDEA_FEATURE_LABELS`: one-hots, the momentum rank),
    then the market-call labels for an unknown `mkt_*` (prefixed "Nifty 50"), else the
    name in words."""
    label = idf.FEATURE_LABELS.get(name) or IDEA_FEATURE_LABELS.get(name)
    if label:
        return str(label)
    if name.startswith(MARKET_PREFIX):
        return f"Nifty 50 {models.feature_label(name[len(MARKET_PREFIX) :])}"
    return models.feature_label(name)


def value_phrase(name: str, value: float | None) -> str:
    """How a value reads in a driver sentence, by feature name ("(no data)" when missing)."""
    if value is None:
        return "(no data)"
    if name.startswith(ONE_HOT_PREFIX):
        return f"= {'yes' if value >= 0.5 else 'no'}"
    if name.startswith(MARKET_PREFIX):
        return models.value_phrase(name[len(MARKET_PREFIX) :], value)
    if name.endswith("_rank"):
        return f"at the {value * 100:.0f}th percentile"
    if name.startswith("rs_"):
        return f"at {value:+.1f} pp"
    if name.endswith("_cr"):
        return f"at Rs {value:,.0f} cr"
    if name == "up_streak":
        n = int(round(value))
        if n == 0:
            return "= 0 sessions"
        return f"= {abs(n)} {'up' if n > 0 else 'down'} session{'' if abs(n) == 1 else 's'}"
    if name.startswith("days_since") or name.endswith("_streak"):
        n = int(round(value))
        return f"= {n} session{'' if n == 1 else 's'}"
    if name.startswith("rel_vol"):
        return f"at {value:.2f}x"
    return models.value_phrase(name, value)


# A label's trailing unit, dropped when the value phrase already carries it.
_UNIT_SUFFIX = re.compile(r"\s*\((?:[^()]*%[^()]*|Rs cr|points|pp|x)\)$")


def driver_text(name: str, value: float | None, contribution: float) -> str:
    """E.g. "RSI (14) at 27 raises the odds of target before stop"."""
    label = feature_label(name)
    phrase = value_phrase(name, value)
    if phrase.endswith(("%", " cr", " pp", "x")):
        label = _UNIT_SUFFIX.sub("", label)
    verb = "raises" if contribution > 0 else "lowers"
    return f"{label} {phrase} {verb} the odds of target before stop"


# --------------------------------------------------------------------------- registry


def result_horizon(result: TrainResult) -> str:
    """The horizon a TrainResult was trained for (metrics, else the artifact)."""
    m = result.metrics
    if isinstance(m, IdeaModelMetrics) and m.horizon:
        return m.horizon
    h = json.loads(result.artifact).get("horizon")
    if not h:
        raise ValueError("TrainResult carries no horizon")
    return str(h)


def save_ideas_model(
    db: Session,
    result: TrainResult,
    *,
    promote: bool,
    note: str | None = None,
    settings: Settings | None = None,
) -> ModelVersion:
    """`registry.save_model` under name "ideas_<horizon>" and the universe target."""
    return registry.save_model(
        db,
        model_target(settings),
        result,
        promote=promote,
        note=note,
        name=model_name(result_horizon(result)),
    )


def ideas_champion(
    db: Session, horizon: str, settings: Settings | None = None
) -> ModelVersion | None:
    """The champion ideas model for a horizon, or None before the first training."""
    return registry.champion(db, model_target(settings), name=model_name(horizon))


LEGACY_LABEL_VERSION = "legacy (next-open anchored, no entry zone)"


def label_version_of(mv: ModelVersion) -> str:
    """The label definition an ideas model was trained on, from its artifact; models from
    before the field existed are "legacy"."""
    if not mv.artifact:
        return LEGACY_LABEL_VERSION
    try:
        payload = json.loads(mv.artifact)
    except ValueError:
        return LEGACY_LABEL_VERSION
    if not isinstance(payload, Mapping):
        return LEGACY_LABEL_VERSION
    version = payload.get("label_version")
    return str(version) if version else LEGACY_LABEL_VERSION


def label_change_reason(champion: ModelVersion | None) -> str | None:
    """Why a freshly trained model must replace the champion whatever the metrics say:
    the champion was trained on another label definition, so its out-of-sample numbers
    describe a different event and cannot be compared. None when the labels agree (or
    there is no champion: the ordinary promotion rule decides)."""
    if champion is None:
        return None
    old = label_version_of(champion)
    if old == idf.LABEL_VERSION:
        return None
    return (
        f"label definition changed ({old} -> {idf.LABEL_VERSION}): the champion's "
        "out-of-sample metrics describe a different event and are not comparable"
    )


def ideas_champion_light(
    db: Session, horizon: str, settings: Settings | None = None
) -> ModelVersion | None:
    """`ideas_champion` without loading the artifact (for the API)."""
    return registry.champion_light(db, model_target(settings), name=model_name(horizon))


# --------------------------------------------------------------------------- training data


UNIVERSE_LABEL_COLUMNS: tuple[str, ...] = (
    "symbol",
    "session_date",
    "instrument_id",
    "label",
    "expired",
    "realised_return_pct",
)


@dataclass(frozen=True)
class TrainingData:
    """What `training_data` reads for one horizon: the candidate frame the model trains on
    and the labelled universe the `random_universe` baseline draws from.

    `frame`: `ideas_features.build_candidate_frame` rows (CANDIDATE_COLUMNS, horizon,
    horizon_days, META_COLUMNS, FEATURE_NAMES, LABEL_COLUMNS) plus `rs_12_1_rank`, the
    momentum percentile the strategies ranked on the eligible cross-section (the live
    `candidate_frame` carries the same column). The last `horizon_days` sessions have NaN
    labels (purged). `universe`: one row per eligible (symbol, session) with
    UNIVERSE_LABEL_COLUMNS, labelled with the same barriers.
    """

    horizon: str
    through: date  # the last session read; bars dated after it were never touched
    frame: pd.DataFrame
    universe: pd.DataFrame
    n_stocks: int  # universe members with at least one session bar

    @property
    def n_candidates(self) -> int:
        return len(self.frame)

    @property
    def n_labelled(self) -> int:
        if self.frame.empty or "label" not in self.frame.columns:
            return 0
        return int(pd.to_numeric(self.frame["label"], errors="coerce").notna().sum())

    @property
    def n_invalidated(self) -> int:
        """Candidates the 09:20 rule would not have filled (next open outside the entry
        zone): never traded, so never labelled, and excluded from training."""
        if self.frame.empty or "invalidated" not in self.frame.columns:
            return 0
        return int(self.frame["invalidated"].fillna(False).astype(bool).sum())


def _session_bars_through(bars: pd.DataFrame, cal: TradingCalendar, through: date) -> pd.DataFrame:
    """`market_data.daily_frame` bars on sessions <= through (weekend/holiday bars dropped)."""
    if bars.empty:
        return bars
    dates = [t.date() for t in bars.index]
    keep = np.array([d <= through and cal.is_trading_day(d) for d in dates], dtype=bool)
    return bars[keep]


def universe_labels(
    db: Session, feats: pd.DataFrame, spec: HorizonSpec, through: date
) -> pd.DataFrame:
    """Triple-barrier labels for every eligible (symbol, session) of a `stock_features`
    frame, with the horizon's barriers (`ideas_features.triple_barrier_labels`); bars
    dated after `through` are not read. Columns UNIVERSE_LABEL_COLUMNS; `label` is NaN for
    sessions whose window has not closed (the last `spec.days` sessions)."""
    empty = pd.DataFrame(columns=list(UNIVERSE_LABEL_COLUMNS))
    if feats.empty:
        return empty
    elig = feats[feats["eligible"].astype(bool)] if "eligible" in feats.columns else feats
    if elig.empty:
        return empty
    cal = TradingCalendar.load(db)
    parts: list[pd.DataFrame] = []
    for symbol, group in elig.groupby(level="symbol", sort=True):
        iid = int(group["instrument_id"].to_numpy()[0])
        sessions = [_as_date(d) for d in group.index.get_level_values("session_date")]
        bars = _session_bars_through(daily_frame(db, iid), cal, through)
        lab = idf.triple_barrier_labels(bars, sessions, spec)
        parts.append(
            pd.DataFrame(
                {
                    "symbol": str(symbol),
                    "session_date": pd.Series(sessions, dtype=object),
                    "instrument_id": iid,
                    "label": lab["label"].to_numpy(dtype=float),
                    "expired": lab["expired"].to_numpy(dtype=bool),
                    "realised_return_pct": lab["realised_return_pct"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True).reindex(columns=list(UNIVERSE_LABEL_COLUMNS))


def training_data(
    db: Session,
    horizon: str,
    *,
    through: date | None = None,
    settings: Settings | None = None,
) -> TrainingData:
    """The training frame for `horizon` from the database, along the one path the live job
    also takes: `ideas_features.stock_features` for every `in_universe` NSE instrument
    through `through` (default: the universe's last bar), `strategies.candidate_frame`
    (eligible rows only, momentum ranked per session), `ideas_features.build_candidate_frame`
    for the features + triple-barrier labels, and `universe_labels` for the baseline.

    Raises ValueError when the universe is empty or has no bars. An empty candidate frame
    is returned as such (train_walk_forward will then say so), never padded.
    """
    s = settings or get_settings()
    spec = horizon_spec(horizon, s)
    members = universe_instruments(db)
    if not members:
        raise ValueError("no universe instruments (in_universe NSE): run `sync-instruments`")
    ids = [m.id for m in members]
    if through is None:
        last = db.scalar(
            select(func.max(DailyBar.trade_date)).where(DailyBar.instrument_id.in_(ids))
        )
        if last is None:
            raise ValueError("no daily bars for the universe: run `backfill`")
        through = last
    through = _as_date(through)
    feats = idf.stock_features(db, ids, through, settings=s)
    n_stocks = int(feats.index.get_level_values("symbol").nunique()) if not feats.empty else 0
    cands = candidate_frame(feats)
    frame = idf.build_candidate_frame(
        db, cands.reindex(columns=list(idf.CANDIDATE_COLUMNS)), spec, through=through, settings=s
    )
    if RANK_COLUMN in cands.columns and not frame.empty:
        key = ["symbol", "session_date", "strategy"]
        rank = cands[[*key, RANK_COLUMN]].drop_duplicates(key)
        frame = frame.merge(rank, on=key, how="left")
    elif RANK_COLUMN not in frame.columns:
        frame[RANK_COLUMN] = np.nan
    universe = universe_labels(db, feats, spec, through)
    log.info(
        "ideas_training_data",
        horizon=horizon,
        through=str(through),
        stocks=n_stocks,
        candidates=len(frame),
        labelled=int(pd.to_numeric(frame["label"], errors="coerce").notna().sum())
        if "label" in frame.columns
        else 0,
        universe_rows=len(universe),
    )
    return TrainingData(
        horizon=horizon, through=through, frame=frame, universe=universe, n_stocks=n_stocks
    )

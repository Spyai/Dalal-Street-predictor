"""Shared vocabulary of the market-call engine (brief §4.1, §4.3).

Everything here is pure Python/numpy: no LightGBM, no database. `app.ml.features`,
`app.ml.models`, `app.services.ledger` and the jobs all import from this module, so
the API can use it too (the API never imports lightgbm; see test_ml_not_in_api).

Conventions
- A "session" is the IST trading day D whose close is predicted. The call is made
  before the open of D and compares close(D) with close(D-1) = `reference_close`.
- Classes: UP if close-to-close % > +threshold, DOWN if < -threshold, else FLAT.
  The threshold is `Settings.market_flat_threshold_pct` (default 0.30).
- Probability dicts always carry all three keys: {"UP": p, "FLAT": p, "DOWN": p}.
- Array order (LightGBM class index) is CLASSES = ("DOWN", "FLAT", "UP").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import TYPE_CHECKING

from app.config import IST, Settings, get_settings

if TYPE_CHECKING:
    from app.services.calendar import TradingCalendar

CLASSES: tuple[str, str, str] = ("DOWN", "FLAT", "UP")
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
MODEL_NAME = "market_call"
BASELINE_NAMES: tuple[str, ...] = ("always_up", "same_as_yesterday", "overnight_gap", "random")
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low", "none")

FeatureRow = dict[str, float | None]


def class_of(pct: float, threshold_pct: float | None = None) -> str:
    """UP / FLAT / DOWN for a close-to-close % change."""
    thr = get_settings().market_flat_threshold_pct if threshold_pct is None else threshold_pct
    if pct > thr:
        return "UP"
    if pct < -thr:
        return "DOWN"
    return "FLAT"


def gap_class_of(gap_pct: float, threshold_pct: float | None = None) -> str:
    """gap_up / gap_down / flat for an open-vs-previous-close % change."""
    thr = get_settings().gap_threshold_pct if threshold_pct is None else threshold_pct
    if gap_pct > thr:
        return "gap_up"
    if gap_pct < -thr:
        return "gap_down"
    return "flat"


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def confidence_label(probs: dict[str, float], settings: Settings | None = None) -> tuple[str, bool]:
    """(label, no_edge). "No edge today" when the top two classes are too close."""
    s = settings or get_settings()
    ordered = sorted((probs.get(c, 0.0) for c in CLASSES), reverse=True)
    top, second = ordered[0], ordered[1]
    margin = top - second
    if top >= s.confidence_high_prob and margin >= 2 * s.no_edge_margin:
        return "high", False
    if top >= 0.45 and margin >= s.no_edge_margin:
        return "medium", False
    if margin >= s.no_edge_margin / 2:
        return "low", False
    return "none", True


def predicted_class(probs: dict[str, float]) -> str:
    return max(CLASSES, key=lambda c: probs.get(c, 0.0))


def session_for(cal: TradingCalendar, now: datetime) -> date:
    """The session a prediction made at `now` is for: today if it is a trading day and
    the market hasn't closed, otherwise the next trading day."""
    now_ist = now.astimezone(IST)
    today = now_ist.date()
    if cal.is_trading_day(today) and now_ist.time() < time(15, 30):
        return today
    return cal.next_trading_day(today)


def is_late(as_of: datetime, session_date: date, settings: Settings | None = None) -> bool:
    """True if the record is written after the ledger lock time (09:15 IST) of its session."""
    s = settings or get_settings()
    lock = datetime.combine(session_date, s.ledger_lock_time, tzinfo=IST)
    return as_of.astimezone(IST) > lock


def finite(x: object) -> float | None:
    """A float for JSON and hashing (rounded to 10 decimals), or None for NaN/inf/None."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 10)


@dataclass(frozen=True)
class Driver:
    """One plain-English reason behind a call, from the model's per-feature contribution."""

    feature: str
    value: float | None
    contribution: float
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "value": finite(self.value),
            "contribution": finite(self.contribution),
            "text": self.text,
        }


@dataclass(frozen=True)
class MarketCallOutput:
    """What the model says for one target and session, before it is written to the ledger."""

    probs: dict[str, float]  # UP / FLAT / DOWN, sums to 1
    predicted_class: str
    range_p10_pct: float | None
    range_p90_pct: float | None
    drivers: list[Driver]
    confidence: str
    no_edge: bool


@dataclass(frozen=True)
class MarketCallRecordIn:
    """Everything the ledger needs for one record (app.services.ledger.record_market_call)."""

    target: str
    session_date: date
    as_of: datetime
    model_version_id: int | None
    reference_date: date
    reference_close: float
    output: MarketCallOutput
    gap_call: str | None
    gap_expected_pct: float | None
    timing_guidance: str | None
    baselines: dict[str, str | None]
    feature_snapshot: FeatureRow
    corrects_prediction_id: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class WalkForwardFold:
    train_end: date
    test_start: date
    test_end: date
    n_test: int
    accuracy: float
    brier: float
    log_loss: float


@dataclass
class WalkForwardMetrics:
    """Out-of-sample metrics from strict walk-forward validation (brief §4.3)."""

    n_oos: int
    accuracy: float
    brier: float
    log_loss: float
    class_frequencies: dict[str, float]
    baselines: dict[str, dict[str, float]]  # name -> {"accuracy": .., "brier": .., "n": ..}
    by_confidence: list[dict[str, float | str | int]]
    calibration: list[dict[str, float | int]]
    folds: list[WalkForwardFold] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_oos": self.n_oos,
            "accuracy": finite(self.accuracy),
            "brier": finite(self.brier),
            "log_loss": finite(self.log_loss),
            "class_frequencies": {k: finite(v) for k, v in self.class_frequencies.items()},
            "baselines": {
                k: {m: finite(v) for m, v in d.items()} for k, d in self.baselines.items()
            },
            "by_confidence": self.by_confidence,
            "calibration": self.calibration,
            "folds": [
                {
                    "train_end": f.train_end.isoformat(),
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "n_test": f.n_test,
                    "accuracy": finite(f.accuracy),
                    "brier": finite(f.brier),
                    "log_loss": finite(f.log_loss),
                }
                for f in self.folds
            ],
        }


@dataclass(frozen=True)
class TrainResult:
    """What `app.ml.models.train_walk_forward` returns and `app.ml.registry` stores."""

    artifact: str  # JSON text: {"classifier": <lgbm model_to_string>, "q10": .., "q90": ..}
    feature_names: list[str]
    params: dict[str, object]
    metrics: WalkForwardMetrics
    trained_through: date
    train_rows: int


# --------------------------------------------------------------------- Phase 3


@dataclass(frozen=True)
class HorizonSpec:
    """A stock-idea horizon (brief §4.2), volatility-scaled: target and stop are ATR(14)
    multiples of the entry price; the idea expires after `days` sessions."""

    name: str
    days: int
    target_atr: float
    stop_atr: float


def horizons(settings: Settings | None = None) -> dict[str, HorizonSpec]:
    s = settings or get_settings()
    return {
        "swing": HorizonSpec("swing", s.swing_horizon_days, s.swing_target_atr, s.swing_stop_atr),
        "positional": HorizonSpec(
            "positional",
            s.positional_horizon_days,
            s.positional_target_atr,
            s.positional_stop_atr,
        ),
    }


IDEA_STATUSES: tuple[str, ...] = (
    "pending",
    "open",
    "target",
    "stop",
    "expired",
    "invalidated",
    "void",
)


@dataclass(frozen=True)
class CostModel:
    """Round-trip delivery costs, % of trade value (brief §4.2: configurable)."""

    brokerage_pct: float
    stt_delivery_pct: float
    exchange_txn_pct: float
    gst_pct: float
    stamp_duty_pct: float
    sebi_fee_pct: float
    slippage_pct: float

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CostModel:
        s = settings or get_settings()
        return cls(
            brokerage_pct=s.brokerage_pct,
            stt_delivery_pct=s.stt_delivery_pct,
            exchange_txn_pct=s.exchange_txn_pct,
            gst_pct=s.gst_pct,
            stamp_duty_pct=s.stamp_duty_pct,
            sebi_fee_pct=s.sebi_fee_pct,
            slippage_pct=s.slippage_pct,
        )

    def round_trip_pct(self) -> float:
        """Total cost of buying and selling, as % of the entry value (both sides)."""
        per_side = self.brokerage_pct + self.exchange_txn_pct + self.sebi_fee_pct
        gst = (self.brokerage_pct + self.exchange_txn_pct) * self.gst_pct / 100.0
        per_side += gst + self.stt_delivery_pct + self.slippage_pct
        return 2.0 * per_side + self.stamp_duty_pct


@dataclass(frozen=True)
class IdeaOutput:
    """What the ideas engine says about one stock for one horizon, before ranking/sizing."""

    symbol: str
    exchange: str
    horizon: str
    horizon_days: int
    setup: str
    strategy: str
    entry_low: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float | None
    p_target: float
    expected_return_low_pct: float
    expected_return_high_pct: float
    risk_reward: float
    ev_pct_after_costs: float
    reasons: list[str]
    invalidation: str
    drivers: list[Driver]


@dataclass(frozen=True)
class IdeaRecordIn:
    """Everything the idea ledger needs (app.services.ledger.record_idea)."""

    session_date: date
    as_of: datetime
    instrument_id: int
    output: IdeaOutput
    position_qty: int
    position_value: float
    risk_amount: float
    rank: int
    feature_snapshot: FeatureRow
    model_version_id: int | None
    surveillance_unknown: bool = True
    corrects_idea_id: int | None = None
    note: str | None = None

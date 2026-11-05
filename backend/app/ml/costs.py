"""Cost and return arithmetic for stock ideas (brief §4.2: ranked by expected value after
costs; the cost model is configurable).

The cost model itself is `app.ml.core.CostModel` (built from Settings): brokerage, STT,
exchange charge, GST, stamp duty, SEBI fee and slippage, totalled by
`CostModel.round_trip_pct()` as a % of the entry value for a buy and a sell. This module
turns % distances (target, stop) and prices into expected values, net returns and
R-multiples. Pure Python/numpy: the API imports it.

Every function raises ValueError for a missing or non-finite input instead of assuming a
number: a cost or return computed from "0 by default" would be a fabricated figure.
"""

from __future__ import annotations

import math

from app.config import Settings, get_settings
from app.ml.core import CostModel

# Expiry is scored as a 0 % gross move in the EV formula (contract §2); the realised
# close-to-close return of expired ideas enters the out-of-sample expectancy instead.
EXPIRY_RETURN_PCT = 0.0


def _num(x: object, name: str) -> float:
    """`x` as a finite float, or ValueError naming the missing input."""
    if x is None or isinstance(x, bool):
        raise ValueError(f"{name} is missing")
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a number: {x!r}") from exc
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} is not finite: {v!r}")
    return v


def cost_model(settings: Settings | None = None) -> CostModel:
    """The configured cost model (`CostModel.from_settings`)."""
    return CostModel.from_settings(settings or get_settings())


def round_trip_cost_pct(costs: CostModel | Settings | None = None) -> float:
    """Buy + sell costs as % of the entry value, from a CostModel or Settings (default)."""
    if isinstance(costs, CostModel):
        return costs.round_trip_pct()
    return cost_model(costs).round_trip_pct()


def pct_distance(price_from: float, price_to: float) -> float:
    """% move from one price to another: (to / from - 1) x 100."""
    a, b = _num(price_from, "price_from"), _num(price_to, "price_to")
    if a <= 0.0:
        raise ValueError(f"price_from must be positive, got {a}")
    return (b / a - 1.0) * 100.0


def gross_return_pct(entry: float, exit_price: float) -> float:
    """Long-only gross return % from entry to exit (negative when exit < entry)."""
    return pct_distance(entry, exit_price)


def net_return_pct(gross_pct: float, cost_pct: float) -> float:
    """Gross return % less the round-trip cost %."""
    return _num(gross_pct, "gross_pct") - _num(cost_pct, "cost_pct")


def expected_value_pct(
    p_target: float, target_pct: float, stop_pct: float, cost_pct: float
) -> float:
    """EV after costs in %: p x T - (1 - p) x S - cost (contract §2; expiry counted as 0).

    `target_pct` and `stop_pct` are the % distances from entry to target and to stop,
    both positive. `p_target` must lie in [0, 1].
    """
    p = _num(p_target, "p_target")
    t, s, c = _num(target_pct, "target_pct"), _num(stop_pct, "stop_pct"), _num(cost_pct, "cost_pct")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p_target must be in [0, 1], got {p}")
    if t <= 0.0 or s <= 0.0:
        raise ValueError(f"target_pct and stop_pct must be positive, got {t} and {s}")
    return p * t - (1.0 - p) * s - c


def breakeven_p_target(target_pct: float, stop_pct: float, cost_pct: float) -> float:
    """The P(target) at which EV after costs is zero: (S + cost) / (T + S)."""
    t, s, c = _num(target_pct, "target_pct"), _num(stop_pct, "stop_pct"), _num(cost_pct, "cost_pct")
    if t <= 0.0 or s <= 0.0:
        raise ValueError(f"target_pct and stop_pct must be positive, got {t} and {s}")
    return (s + c) / (t + s)


def risk_reward(target_pct: float, stop_pct: float) -> float:
    """Reward-to-risk ratio T / S from the % distances to target and stop."""
    t, s = _num(target_pct, "target_pct"), _num(stop_pct, "stop_pct")
    if s <= 0.0:
        raise ValueError(f"stop_pct must be positive, got {s}")
    return t / s


def r_multiple(net_pct: float, stop_pct: float) -> float:
    """Net return as a multiple of the initial risk (1R = the % distance to the stop)."""
    n, s = _num(net_pct, "net_pct"), _num(stop_pct, "stop_pct")
    if s <= 0.0:
        raise ValueError(f"stop_pct must be positive, got {s}")
    return n / s


def cost_amount(entry_value: float, costs: CostModel | Settings | None = None) -> float:
    """Round-trip costs in rupees for a position worth `entry_value` at entry."""
    v = _num(entry_value, "entry_value")
    if v < 0.0:
        raise ValueError(f"entry_value must be non-negative, got {v}")
    return v * round_trip_cost_pct(costs) / 100.0

"""From scored candidates to the day's ideas: price levels, EV after costs, the bar, the
ranking, the caps and position sizing (brief §4.2; contract §2 "Ranking").

    levels   entry zone = close(D) +/- ENTRY_ZONE_ATR x ATR(14); stop = close x (1 - k_s x
             ATR%); target_1 = close x (1 + k_t x ATR%); target_2 = 1.5 x the target_1
             distance. k_t / k_s come from the horizon (`core.HorizonSpec`). T and S are the
             % distances from the zone's midpoint, the same distances the labels used.
    EV       p x T - (1 - p) x S - round-trip cost (`app.ml.costs`), in %.
    bar      p_target >= ideas_min_p_target and EV >= ideas_min_ev_pct.
    ranking  by EV (then p_target, then symbol), one idea per symbol per horizon, at most
             ideas_max_per_horizon per horizon; an idea the paper book cannot size (0
             shares within the risk and position caps) is dropped too.
    sizing   qty = floor(risk budget / (entry_mid - stop)), risk budget = paper_capital x
             risk_per_trade_pct %, capped so qty x entry_mid <= paper_max_position_pct %
             of paper_capital.

Nothing clears the bar -> an empty list and `no_edge_text()`: "No edge today". A
candidate without a close or an ATR cannot be priced and is never guessed at. Pure
numpy/pandas: the API imports this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.ml import costs
from app.ml.core import Driver, HorizonSpec, IdeaOutput, finite, horizons
from app.ml.strategies import STRATEGY_NAMES, Candidate

ENTRY_ZONE_ATR = 0.5  # half-width of the entry zone, in ATR(14), around the close of D
TARGET_2_MULTIPLE = 1.5  # target_2 distance = 1.5 x target_1 distance
NO_EDGE_TEXT = "No edge today"
PRICE_DECIMALS = 2


def no_edge_text(horizon: str | None = None, settings: Settings | None = None) -> str:
    """The honest line for a horizon with no idea, quoting the bar it had to clear."""
    s = settings or get_settings()
    where = f" ({horizon})" if horizon else ""
    return (
        f"{NO_EDGE_TEXT}{where}: no candidate clears P(target) >= {s.ideas_min_p_target:.0%} "
        f"and EV >= {s.ideas_min_ev_pct:g}% after costs"
    )


# --------------------------------------------------------------------------- price levels


@dataclass(frozen=True)
class Levels:
    """Entry zone, stop and targets for one candidate, with the % distances behind them."""

    entry_low: float
    entry_mid: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float
    target_pct: float  # % from entry_mid to target_1 (T)
    stop_pct: float  # % from entry_mid down to the stop (S)
    expected_return_low_pct: float  # = target_pct
    expected_return_high_pct: float  # % from entry_mid to target_2
    risk_reward: float  # T / S


def price_levels(close: float | None, atr14_pct: float | None, spec: HorizonSpec) -> Levels:
    """Volatility-scaled levels around `close` for the horizon. Raises ValueError when the
    close or the ATR is missing or not positive: levels are never made up."""
    c, a = finite(close), finite(atr14_pct)
    if c is None or c <= 0.0:
        raise ValueError(f"close is missing or not positive: {close!r}")
    if a is None or a <= 0.0:
        raise ValueError(f"atr14_pct is missing or not positive: {atr14_pct!r}")
    if spec.target_atr <= 0.0 or spec.stop_atr <= 0.0:
        raise ValueError(f"horizon {spec.name} needs positive ATR multiples")
    atr = c * a / 100.0
    target_pct = spec.target_atr * a
    stop_pct = spec.stop_atr * a
    if stop_pct >= 100.0:
        raise ValueError(f"stop distance {stop_pct:.1f}% is not a price")
    half = ENTRY_ZONE_ATR * atr
    return Levels(
        entry_low=round(c - half, PRICE_DECIMALS),
        entry_mid=round(c, PRICE_DECIMALS),
        entry_high=round(c + half, PRICE_DECIMALS),
        stop=round(c * (1.0 - stop_pct / 100.0), PRICE_DECIMALS),
        target_1=round(c * (1.0 + target_pct / 100.0), PRICE_DECIMALS),
        target_2=round(c * (1.0 + TARGET_2_MULTIPLE * target_pct / 100.0), PRICE_DECIMALS),
        target_pct=target_pct,
        stop_pct=stop_pct,
        expected_return_low_pct=target_pct,
        expected_return_high_pct=TARGET_2_MULTIPLE * target_pct,
        risk_reward=costs.risk_reward(target_pct, stop_pct),
    )


# --------------------------------------------------------------------------- scoring input


@dataclass(frozen=True)
class ScoredCandidate:
    """A strategy candidate with the model's P(target before stop) for one horizon."""

    candidate: Candidate
    p_target: float
    horizon: HorizonSpec
    drivers: list[Driver] = field(default_factory=list)
    exchange: str = "NSE"


ScoredLike = ScoredCandidate | tuple[Candidate, float, Mapping[str, object]]


def _as_scored(item: ScoredLike, settings: Settings) -> ScoredCandidate:
    """`ScoredCandidate`, or (candidate, p_target, fields) with fields {"horizon": name or
    HorizonSpec, "drivers": [...], "exchange": ...}."""
    if isinstance(item, ScoredCandidate):
        return item
    candidate, p, fields = item
    spec = fields.get("horizon")
    if isinstance(spec, str):
        spec = horizons(settings)[spec]
    if not isinstance(spec, HorizonSpec):
        raise ValueError("scored candidate needs a horizon (name or HorizonSpec)")
    raw = fields.get("drivers")
    drivers = [d for d in raw if isinstance(d, Driver)] if isinstance(raw, Iterable) else []
    exchange = fields.get("exchange", "NSE")
    return ScoredCandidate(
        candidate=candidate,
        p_target=float(p),
        horizon=spec,
        drivers=drivers,
        exchange=str(exchange),
    )


def build_output(sc: ScoredCandidate, settings: Settings | None = None) -> IdeaOutput:
    """The full `IdeaOutput` for one scored candidate (levels, EV after costs, text).

    Raises ValueError when the candidate cannot be priced (no close / ATR) or p_target
    is not a probability.
    """
    s = settings or get_settings()
    p = finite(sc.p_target)
    if p is None or not 0.0 <= p <= 1.0:
        raise ValueError(f"p_target must be a probability, got {sc.p_target!r}")
    lv = price_levels(sc.candidate.close, sc.candidate.atr14_pct, sc.horizon)
    cost_pct = costs.round_trip_cost_pct(s)
    ev = costs.expected_value_pct(p, lv.target_pct, lv.stop_pct, cost_pct)
    return IdeaOutput(
        symbol=sc.candidate.symbol,
        exchange=sc.exchange,
        horizon=sc.horizon.name,
        horizon_days=sc.horizon.days,
        setup=sc.candidate.setup,
        strategy=sc.candidate.strategy,
        entry_low=lv.entry_low,
        entry_high=lv.entry_high,
        stop=lv.stop,
        target_1=lv.target_1,
        target_2=lv.target_2,
        p_target=p,
        expected_return_low_pct=lv.expected_return_low_pct,
        expected_return_high_pct=lv.expected_return_high_pct,
        risk_reward=lv.risk_reward,
        ev_pct_after_costs=ev,
        reasons=list(sc.candidate.reasons),
        invalidation=sc.candidate.invalidation,
        drivers=list(sc.drivers),
    )


def entry_mid(out: IdeaOutput) -> float:
    return (out.entry_low + out.entry_high) / 2.0


def clears_bar(out: IdeaOutput, settings: Settings | None = None) -> bool:
    """p_target >= ideas_min_p_target and EV after costs >= ideas_min_ev_pct."""
    s = settings or get_settings()
    return out.p_target >= s.ideas_min_p_target and out.ev_pct_after_costs >= s.ideas_min_ev_pct


# --------------------------------------------------------------------------- ranking


@dataclass
class RankReport:
    """What the ranking saw, for the job details and the morning brief."""

    ideas: list[IdeaOutput]
    n_candidates: int
    n_unpriced: int  # no close / ATR: could not be priced
    n_cleared: int  # cleared the p / EV bar (before caps and sizing)
    n_unsizable: int  # cleared the bar but 0 shares fit the risk and position caps
    best_p_target: float | None
    best_ev_pct: float | None
    no_edge: dict[str, str]  # horizon -> "No edge today ..." for horizons with no idea

    def as_dict(self) -> dict[str, object]:
        return {
            "n_ideas": len(self.ideas),
            "n_candidates": self.n_candidates,
            "n_unpriced": self.n_unpriced,
            "n_cleared": self.n_cleared,
            "n_unsizable": self.n_unsizable,
            "best_p_target": finite(self.best_p_target),
            "best_ev_pct": finite(self.best_ev_pct),
            "no_edge": dict(self.no_edge),
        }


def _sort_key(out: IdeaOutput) -> tuple[float, float, int, str]:
    order = (
        STRATEGY_NAMES.index(out.strategy)
        if out.strategy in STRATEGY_NAMES
        else len(STRATEGY_NAMES)
    )
    return (-out.ev_pct_after_costs, -out.p_target, order, out.symbol)


def rank_report(scored: Iterable[ScoredLike], settings: Settings | None = None) -> RankReport:
    """Rank scored candidates into at most `ideas_max_per_horizon` ideas per horizon.

    Steps per horizon: price and score each candidate (unpriceable ones are counted and
    skipped), keep those that clear the bar, drop those the paper book cannot size, keep
    the best EV per symbol, sort by EV, cut to the cap. Horizons that end up empty get a
    "No edge today" line in `no_edge`. The ideas come back grouped by horizon in the
    `core.horizons()` order, best EV first within each.
    """
    s = settings or get_settings()
    outputs: dict[str, list[IdeaOutput]] = {}
    n_candidates = n_unpriced = n_cleared = n_unsizable = 0
    best_p: float | None = None
    best_ev: float | None = None
    seen_horizons: list[str] = []
    for item in scored:
        sc = _as_scored(item, s)
        n_candidates += 1
        if sc.horizon.name not in seen_horizons:
            seen_horizons.append(sc.horizon.name)
        try:
            out = build_output(sc, s)
        except ValueError:
            n_unpriced += 1
            continue
        best_p = out.p_target if best_p is None else max(best_p, out.p_target)
        best_ev = (
            out.ev_pct_after_costs if best_ev is None else max(best_ev, out.ev_pct_after_costs)
        )
        if not clears_bar(out, s):
            continue
        n_cleared += 1
        if size_position(entry_mid(out), out.stop, s)[0] <= 0:
            n_unsizable += 1
            continue
        outputs.setdefault(out.horizon, []).append(out)

    ideas: list[IdeaOutput] = []
    no_edge: dict[str, str] = {}
    known = list(horizons(s))
    ordered = [h for h in known if h in seen_horizons] + [
        h for h in seen_horizons if h not in known
    ]
    for h in ordered:
        best_per_symbol: dict[str, IdeaOutput] = {}
        for out in sorted(outputs.get(h, []), key=_sort_key):
            best_per_symbol.setdefault(out.symbol, out)  # first = best EV for that symbol
        kept = sorted(best_per_symbol.values(), key=_sort_key)[: max(s.ideas_max_per_horizon, 0)]
        if kept:
            ideas.extend(kept)
        else:
            no_edge[h] = no_edge_text(h, s)
    return RankReport(
        ideas=ideas,
        n_candidates=n_candidates,
        n_unpriced=n_unpriced,
        n_cleared=n_cleared,
        n_unsizable=n_unsizable,
        best_p_target=best_p,
        best_ev_pct=best_ev,
        no_edge=no_edge,
    )


def rank_ideas(scored: Iterable[ScoredLike], settings: Settings | None = None) -> list[IdeaOutput]:
    """The ranked ideas only (see `rank_report`); empty means "No edge today"."""
    return rank_report(scored, settings).ideas


# --------------------------------------------------------------------------- sizing


def size_position(
    entry_mid: float, stop: float, settings: Settings | None = None
) -> tuple[int, float, float]:
    """(qty, position_value, risk_amount) for one idea from the paper risk settings.

    qty = floor(risk budget / (entry_mid - stop)) with risk budget = paper_capital x
    risk_per_trade_pct / 100, then capped so position_value = qty x entry_mid stays within
    paper_max_position_pct % of paper_capital. `risk_amount` is what that qty actually
    risks to the stop, qty x (entry_mid - stop): equal to the budget (less a share's
    rounding) when the position cap does not bind, smaller when it does. qty is 0 when
    even one share breaks the caps. Raises ValueError for a missing price or a stop at
    or above the entry.
    """
    s = settings or get_settings()
    e, st = finite(entry_mid), finite(stop)
    if e is None or e <= 0.0:
        raise ValueError(f"entry_mid is missing or not positive: {entry_mid!r}")
    if st is None or st <= 0.0:
        raise ValueError(f"stop is missing or not positive: {stop!r}")
    if st >= e:
        raise ValueError(f"stop {st} must be below the entry {e}")
    per_share_risk = e - st
    budget = s.paper_capital * s.risk_per_trade_pct / 100.0
    max_value = s.paper_capital * s.paper_max_position_pct / 100.0
    qty = min(math.floor(budget / per_share_risk), math.floor(max_value / e))
    qty = max(int(qty), 0)
    return qty, qty * e, qty * per_share_risk


def sized(
    ideas: Sequence[IdeaOutput], settings: Settings | None = None
) -> list[tuple[int, float, float]]:
    """`size_position` for each idea, in order (for the ledger's qty / value / risk)."""
    return [size_position(entry_mid(o), o.stop, settings) for o in ideas]

"""The stock-ideas scorecard (brief §6.3): the live track record beside every idea.

Everything here is derived from the idea ledger and `idea_outcome`; nothing is
estimated. A metric with no graded ideas behind it is None (the UI shows "—"), every
number carries its `n`, the sample-size warning is on below `Settings.min_sample_size`
graded ideas, and the verdict says in plain words whether the ideas are making money
after costs.

Definitions
- graded: outcome status target / stop / expired. Pending, open, invalidated and voided
  ideas are counted in `by_status` but take part in no rate.
- hit rate: share of graded ideas whose status is "target" (target before stop), the event
  `p_target` predicts, so the calibration table compares like with like.
- win rate: share of graded ideas with a positive net return after costs (an expired idea
  can be a winner or a loser).
- average win / loss, expectancy and profit factor use `net_return_pct` (after the
  round-trip cost model), per idea, in %.
- superseded ideas (ones a correction points at) are left out so a corrected idea counts
  once; `ledger.idea_history` still lists every record.

`idea_track` is the small per-(horizon, setup) record shown next to each idea, and the
champion box per horizon comes from `registry.champion_light` (never the artifact).
Aggregates never load `Idea.feature_snapshot` or `Idea.drivers`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, defer

from app.config import Settings, get_settings
from app.ml import registry
from app.ml.core import IDEA_STATUSES, finite, horizons
from app.ml.ideas_models import model_name, model_target
from app.models import Idea, IdeaOutcome
from app.services.ledger import not_superseded_idea

GRADED_STATUSES: tuple[str, ...] = ("target", "stop", "expired")
RECENT_LIMIT = 20
CALIBRATION_BINS = 10
OOS_KEYS: tuple[str, ...] = (
    "n_oos",
    "hit_rate",
    "n_selected",
    "hit_rate_ranked",
    "n_ranked",
    "brier",
    "log_loss",
    "accuracy",
    "expectancy_after_costs_pct",
    "expectancy_ranked_pct",
    "expectancy_all_candidates_pct",
    "hit_rate_top",
    "n_top",
    "expectancy_top_pct",
    "cost_round_trip_pct",
    "n_sessions",
    "embargo_sessions",
)
OOS_BASELINE_NAMES: tuple[str, ...] = ("all_candidates", "random_universe")
BASELINE_LABELS: dict[str, str] = {
    "all_candidates": "every candidate that session",
    "random_universe": "random universe stocks, same barriers",
}


@dataclass(frozen=True)
class IdeaTrack:
    """The live record of one setup at one horizon, for the line next to an idea."""

    horizon: str
    setup: str | None
    n_graded: int
    hit_rate: float | None
    win_rate: float | None
    expectancy_pct: float | None
    sample_warning: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdeasScorecard:
    """The ideas' live track record, for one horizon or all. `n_graded` is the sample size
    behind every rate; `by_status` counts every current idea."""

    horizon: str | None
    n_ideas: int
    n_graded: int
    by_status: dict[str, int]
    hit_rate: float | None
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy_pct: float | None
    profit_factor: float | None
    avg_r_multiple: float | None
    avg_days_held: float | None
    by_setup: list[dict[str, Any]]
    by_horizon: list[dict[str, Any]]
    calibration: list[dict[str, Any]]  # bins of p_target vs the observed hit rate
    sample_warning: bool
    verdict: str
    champions: dict[str, dict[str, Any] | None]  # horizon -> champion model summary
    recent: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- pure math


def _mean(values: Sequence[float]) -> float | None:
    return finite(float(np.mean(values))) if len(values) else None


def group_stats(graded: Iterable[tuple[Idea, IdeaOutcome]]) -> dict[str, Any]:
    """hit rate, win rate, average win / loss, expectancy, profit factor and n of a group
    of graded ideas (None where there is nothing to compute from)."""
    rows = list(graded)
    nets = [float(o.net_return_pct) for _, o in rows if o.net_return_pct is not None]
    hits = [1.0 if o.status == "target" else 0.0 for _, o in rows]
    wins = [x for x in nets if x > 0.0]
    losses = [x for x in nets if x < 0.0]
    gross_loss = -sum(losses)
    rs = [float(o.r_multiple) for _, o in rows if o.r_multiple is not None]
    held = [float(o.days_held) for _, o in rows if o.days_held is not None]
    return {
        "n": len(rows),
        "hit_rate": _mean(hits),
        "win_rate": _mean([1.0 if x > 0.0 else 0.0 for x in nets]),
        "avg_win_pct": _mean(wins),
        "avg_loss_pct": _mean(losses),
        "expectancy_pct": _mean(nets),
        "profit_factor": finite(sum(wins) / gross_loss) if gross_loss > 0.0 else None,
        "avg_r_multiple": _mean(rs),
        "avg_days_held": _mean(held),
    }


def calibration_bins(
    p_target: Sequence[float], hit: Sequence[bool], bins: int = CALIBRATION_BINS
) -> list[dict[str, float | int]]:
    """Reliability table of P(target): equal-width bins over [0, 1], empty ones left out.
    Rows: {"bin_lo", "bin_hi", "n", "predicted_mean", "observed"}."""
    p = np.asarray(list(p_target), dtype=float).reshape(-1)
    ok = np.asarray(list(hit), dtype=bool).reshape(-1)
    if p.size == 0 or bins < 1:
        return []
    # Bin k holds [k / bins, (k + 1) / bins); the last bin is closed so p = 1.0 is counted.
    # floor(p x bins) avoids the float edges of linspace (0.6 must land in the 0.6 bin).
    idx = np.clip(np.floor(p * bins).astype(int), 0, bins - 1)
    out: list[dict[str, float | int]] = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "bin_lo": round(b / bins, 4),
                "bin_hi": round((b + 1) / bins, 4),
                "n": n,
                "predicted_mean": float(p[mask].mean()),
                "observed": float(ok[mask].mean()),
            }
        )
    return out


def verdict_text(
    n: int,
    hit_rate: float | None,
    expectancy_pct: float | None,
    profit_factor: float | None,
    sample_warning: bool,
    min_sample_size: int | None = None,
    horizon: str | None = None,
) -> str:
    """The plain-words verdict: how often the target came first, what an idea made after
    costs, and whether that is proven yet."""
    min_n = get_settings().min_sample_size if min_sample_size is None else min_sample_size
    where = f" ({horizon})" if horizon else ""
    if n == 0 or hit_rate is None:
        return (
            f"No graded ideas{where} yet: the ideas engine is unproven. "
            f"Treat it as unproven until it has {min_n} graded ideas."
        )
    ideas = "idea" if n == 1 else "ideas"
    parts = [
        f"After {n} graded {ideas}{where} the target came before the stop "
        f"{hit_rate:.0%} of the time."
    ]
    if expectancy_pct is not None:
        parts.append(f"The average idea made {expectancy_pct:+.2f}% after costs")
        if profit_factor is not None:
            parts[-1] += f" (profit factor {profit_factor:.2f})."
        else:
            parts[-1] += " (no losing idea yet, so no profit factor)."
        if expectancy_pct > 0.0:
            parts.append("The ideas are making money after costs so far.")
        elif expectancy_pct < 0.0:
            parts.append("The ideas are NOT making money after costs so far.")
        else:
            parts.append("The ideas are breaking even after costs so far.")
    if sample_warning:
        parts.append(f"Fewer than {min_n} graded ideas: treat as unproven.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- champion


def _iso(dt: datetime | date | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def oos_baselines(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The walk-forward baselines stored with an ideas model: {"name", "label", "n",
    "hit_rate", "brier"} per `OOS_BASELINE_NAMES` entry; a missing number is None."""
    raw = metrics.get("baselines")
    if not isinstance(raw, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for name in OOS_BASELINE_NAMES:
        entry = raw.get(name)
        e: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
        n = finite(e.get("n"))
        out.append(
            {
                "name": name,
                "label": BASELINE_LABELS.get(name, name.replace("_", " ")),
                "n": int(n) if n is not None else 0,
                "hit_rate": finite(e.get("hit_rate")),
                "brier": finite(e.get("brier")),
            }
        )
    return out


def oos_verdict_text(oos: Mapping[str, Any], baselines: list[dict[str, Any]]) -> str:
    """Plain words about the walk-forward period: the selected ideas' hit rate against
    every candidate's and the expectancy after costs; when nothing cleared the bar, that;
    and, when stored, the replay that shows the best ideas every session regardless of
    the bar. Says so when a number is missing."""
    hit = finite(oos.get("hit_rate"))
    n_sel = finite(oos.get("n_selected"))
    base = next((b for b in baselines if b["name"] == "all_candidates"), None)
    base_text = (
        f"{base['hit_rate']:.0%} for every candidate (n={base['n']})"
        if base is not None and base["hit_rate"] is not None and base["n"] > 0
        else None
    )
    if n_sel is not None and int(n_sel) == 0:
        text = "Out of sample no candidate cleared the bar, so the bar itself is untested."
    elif hit is None or n_sel is None:
        text = "No out-of-sample hit rate was stored with this model."
    else:
        text = f"Out of sample {int(n_sel)} ideas cleared the bar and hit the target {hit:.0%}"
        if base_text is not None:
            text += f" of the time, against {base_text}."
        else:
            text += " of the time; no all-candidates baseline was stored to compare with."
        exp = finite(oos.get("expectancy_after_costs_pct"))
        if exp is not None:
            text += f" Expectancy after costs {exp:+.2f}% per idea."
        else:
            text += " No expectancy after costs was stored."
    top = finite(oos.get("hit_rate_top"))
    n_top = finite(oos.get("n_top"))
    if top is not None and n_top is not None and int(n_top) > 0:
        text += (
            f" Showing the best ideas every session regardless of the bar would have hit "
            f"{top:.0%} (n={int(n_top)})"
        )
        if base_text is not None:
            text += f" against {base_text}"
        exp_top = finite(oos.get("expectancy_top_pct"))
        if exp_top is not None:
            text += f", {exp_top:+.2f}% per idea after costs"
        text += "."
    return text


def champion_summary(
    db: Session, horizon: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:
    """The champion ideas model for a horizon (`ideas_<horizon>` on the universe target)
    and its out-of-sample metrics, or None if there is none. The artifact is never read."""
    s = settings or get_settings()
    mv = registry.champion_light(db, model_target(s), name=model_name(horizon))
    if mv is None:
        return None
    metrics: dict[str, Any] = dict(mv.metrics or {})
    oos = {k: metrics.get(k) for k in OOS_KEYS}
    baselines = oos_baselines(metrics)
    per_strategy = metrics.get("per_strategy")
    calibration = metrics.get("calibration")
    lift = metrics.get("lift")
    freq = metrics.get("class_frequencies")
    return {
        "id": mv.id,
        "name": mv.name,
        "horizon": horizon,
        "version": mv.version,
        "kind": mv.kind,
        "trained_through": mv.trained_through.isoformat(),
        "train_rows": mv.train_rows,
        "oos": oos,
        "baselines": baselines,
        "per_strategy": dict(per_strategy) if isinstance(per_strategy, Mapping) else {},
        "calibration": list(calibration) if isinstance(calibration, list) else [],
        "lift": list(lift) if isinstance(lift, list) else [],
        "class_frequencies": dict(freq) if isinstance(freq, Mapping) else None,
        "notes": list(metrics.get("notes") or []),
        "oos_verdict": oos_verdict_text(oos, baselines),
        "promoted_at": _iso(mv.promoted_at),
        "created_at": _iso(mv.created_at),
    }


# ----------------------------------------------------------------------------- rows


def row_dict(i: Idea, o: IdeaOutcome | None) -> dict[str, Any]:
    """One idea plus its outcome (outcome fields None while pending), for tables and the API."""
    return {
        "id": i.id,
        "seq": i.seq,
        "session_date": i.session_date.isoformat(),
        "as_of": _iso(i.as_of),
        "is_late": i.is_late,
        "instrument_id": i.instrument_id,
        "symbol": i.symbol,
        "exchange": i.exchange,
        "horizon": i.horizon,
        "horizon_days": i.horizon_days,
        "setup": i.setup,
        "strategy": i.strategy,
        "direction": i.direction,
        "entry_low": i.entry_low,
        "entry_high": i.entry_high,
        "stop": i.stop,
        "target_1": i.target_1,
        "target_2": i.target_2,
        "p_target": i.p_target,
        "expected_return_low_pct": i.expected_return_low_pct,
        "expected_return_high_pct": i.expected_return_high_pct,
        "risk_reward": i.risk_reward,
        "ev_pct_after_costs": i.ev_pct_after_costs,
        "position_qty": i.position_qty,
        "position_value": i.position_value,
        "risk_amount": i.risk_amount,
        "rank": i.rank,
        "surveillance_unknown": i.surveillance_unknown,
        "model_version_id": i.model_version_id,
        "corrects_idea_id": i.corrects_idea_id,
        "note": i.note,
        "status": o.status if o else "pending",
        "entry_date": _iso(o.entry_date) if o else None,
        "entry_price": o.entry_price if o else None,
        "exit_date": _iso(o.exit_date) if o else None,
        "exit_price": o.exit_price if o else None,
        "gross_return_pct": o.gross_return_pct if o else None,
        "costs_pct": o.costs_pct if o else None,
        "net_return_pct": o.net_return_pct if o else None,
        "r_multiple": o.r_multiple if o else None,
        "days_held": o.days_held if o else None,
        "max_favorable_pct": o.max_favorable_pct if o else None,
        "max_adverse_pct": o.max_adverse_pct if o else None,
        "graded_at": _iso(o.graded_at) if o else None,
        "outcome_source": o.source if o else None,
        "outcome_notes": o.notes if o else None,
        "error_category": o.error_category if o else None,
    }


def _base_query(horizon: str | None, setup: str | None = None) -> Any:
    stmt = (
        select(Idea, IdeaOutcome)
        # Nothing here reads the feature snapshot or the drivers; keep the big JSON out.
        .options(defer(Idea.feature_snapshot), defer(Idea.drivers))
        .outerjoin(IdeaOutcome, IdeaOutcome.idea_id == Idea.id)
        .where(not_superseded_idea())
    )
    if horizon is not None:
        stmt = stmt.where(Idea.horizon == horizon)
    if setup is not None:
        stmt = stmt.where(Idea.setup == setup)
    return stmt


def _graded(rows: Iterable[tuple[Idea, IdeaOutcome | None]]) -> list[tuple[Idea, IdeaOutcome]]:
    return [(i, o) for i, o in rows if o is not None and o.status in GRADED_STATUSES]


def idea_track(
    db: Session, horizon: str, setup: str | None = None, *, settings: Settings | None = None
) -> IdeaTrack:
    """The live record of a setup at a horizon (all setups when `setup` is None)."""
    s = settings or get_settings()
    rows = db.execute(_base_query(horizon, setup)).all()
    stats = group_stats(_graded([(i, o) for i, o in rows]))
    return IdeaTrack(
        horizon=horizon,
        setup=setup,
        n_graded=stats["n"],
        hit_rate=stats["hit_rate"],
        win_rate=stats["win_rate"],
        expectancy_pct=stats["expectancy_pct"],
        sample_warning=stats["n"] < s.min_sample_size,
    )


def _grouped(
    graded: list[tuple[Idea, IdeaOutcome]], key: str, order: Sequence[str]
) -> list[dict[str, Any]]:
    """`group_stats` per value of `key` ("setup" / "horizon"); the values in `order` always
    get a row (n = 0 when empty), other values seen follow alphabetically."""
    seen = sorted({getattr(i, key) for i, _ in graded} - set(order))
    out: list[dict[str, Any]] = []
    for value in [*order, *seen]:
        stats = group_stats((i, o) for i, o in graded if getattr(i, key) == value)
        out.append({key: value, **stats})
    return out


def ideas_scorecard(
    db: Session, horizon: str | None = None, *, settings: Settings | None = None
) -> IdeasScorecard:
    """Aggregate every current (non-superseded) idea, for one horizon or all of them."""
    s = settings or get_settings()
    if horizon is not None and horizon not in horizons(s):
        raise ValueError(f"unknown horizon {horizon!r}")
    rows = [(i, o) for i, o in db.execute(_base_query(horizon).order_by(Idea.seq.asc())).all()]
    graded = _graded(rows)
    by_status = dict.fromkeys(IDEA_STATUSES, 0)
    for _, o in rows:
        by_status[o.status if o is not None else "pending"] = (
            by_status.get(o.status if o is not None else "pending", 0) + 1
        )
    stats = group_stats(graded)
    n = stats["n"]
    sample_warning = n < s.min_sample_size
    setups = sorted({i.setup for i, _ in graded})
    known_horizons = [h for h in horizons(s) if horizon is None or h == horizon]
    recent = db.execute(
        _base_query(horizon).order_by(Idea.as_of.desc(), Idea.seq.desc()).limit(RECENT_LIMIT)
    ).all()
    return IdeasScorecard(
        horizon=horizon,
        n_ideas=len(rows),
        n_graded=n,
        by_status=by_status,
        hit_rate=stats["hit_rate"],
        win_rate=stats["win_rate"],
        avg_win_pct=stats["avg_win_pct"],
        avg_loss_pct=stats["avg_loss_pct"],
        expectancy_pct=stats["expectancy_pct"],
        profit_factor=stats["profit_factor"],
        avg_r_multiple=stats["avg_r_multiple"],
        avg_days_held=stats["avg_days_held"],
        by_setup=_grouped(graded, "setup", setups),
        by_horizon=_grouped(graded, "horizon", known_horizons),
        calibration=[
            dict(r)
            for r in calibration_bins(
                [i.p_target for i, _ in graded], [o.status == "target" for _, o in graded]
            )
        ],
        sample_warning=sample_warning,
        verdict=verdict_text(
            n,
            stats["hit_rate"],
            stats["expectancy_pct"],
            stats["profit_factor"],
            sample_warning,
            s.min_sample_size,
            horizon,
        ),
        champions={h: champion_summary(db, h, settings=s) for h in known_horizons},
        recent=[row_dict(i, o) for i, o in recent],
    )

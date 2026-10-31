"""Telegram texts for the market-call loop: the morning brief and the close summary.

Telegram renders the HTML parse mode, so every string that did not originate here
(target names, driver sentences, verdicts, timing guidance) goes through `html.escape`.
Numbers are formatted by hand; a missing number is shown as an em dash, never a guess.
The brief always carries the model's confidence or "No edge today", the live track record
beside the call, and the footer the brief asks for.
"""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Iterable
from datetime import date

from app.config import Settings, get_settings
from app.models import Prediction, PredictionGrade
from app.services.scorecard import Scorecard, baseline_label

FOOTER = "Personal research tool. Not investment advice."
MISSING = "—"  # em dash
TOP_DRIVERS_IN_BRIEF = 3
GAP_LABELS: dict[str, str] = {"gap_up": "gap up", "gap_down": "gap down", "flat": "flat open"}


def _pct(x: float | None, dp: int = 2, *, sign: bool = True) -> str:
    """A signed percentage such as "+0.42%" or "-1.20%", or MISSING for None."""
    if x is None:
        return MISSING
    return f"{x:+.{dp}f}%" if sign else f"{x:.{dp}f}%"


def _prob(x: float | None) -> str:
    return MISSING if x is None else f"{x * 100:.0f}%"


def _day(d: date) -> str:
    return d.strftime("%a %d %b %Y")


def _esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def track_record_line(card: Scorecard | None, settings: Settings | None = None) -> str:
    """One plain line: sample size, accuracy against the best baseline, or "unproven"."""
    s = settings or get_settings()
    if card is None:
        return "Track record: unavailable"
    unproven = f"unproven (&lt; {s.min_sample_size} graded)"
    if card.n_graded == 0 or card.accuracy is None:
        return f"Track record: no graded calls yet, {unproven}"
    with_calls = [b for b in card.baselines if b.get("accuracy") is not None]
    best = max(with_calls, key=lambda b: (b["accuracy"], b["n"])) if with_calls else None
    line = f"Track record: {card.n_graded} graded, {card.accuracy * 100:.0f}% right"
    if best is not None:
        line += (
            f" vs best baseline ({_esc(baseline_label(str(best['name'])))}) "
            f"{float(best['accuracy']) * 100:.0f}%"
        )
    if card.sample_warning:
        line += f"; {unproven}"
    return line


def _call_block(p: Prediction, card: Scorecard | None, settings: Settings) -> list[str]:
    name = f"<b>{_esc(p.target)}</b>"
    if p.no_edge:
        head = f"{name}: <b>No edge today</b> (leans {_esc(p.predicted_class)})"
    else:
        head = f"{name}: <b>{_esc(p.predicted_class)}</b> ({_esc(p.confidence)} confidence)"
    lines = [
        head,
        f"UP {_prob(p.prob_up)} · FLAT {_prob(p.prob_flat)} · DOWN {_prob(p.prob_down)}",
        f"Range (10–90%): {_pct(p.range_p10_pct)} to {_pct(p.range_p90_pct)}",
    ]
    if p.gap_call is None:
        lines.append("Gap: no call (overnight futures missing)")
    else:
        lines.append(
            f"Gap: {_esc(GAP_LABELS.get(p.gap_call, p.gap_call))} "
            f"({_pct(p.gap_expected_pct)} expected)"
        )
    drivers = [d for d in (p.drivers or []) if isinstance(d, dict) and d.get("text")]
    if drivers:
        lines.append("Drivers:")
        lines += [f"• {_esc(d['text'])}" for d in drivers[:TOP_DRIVERS_IN_BRIEF]]
    else:
        lines.append("Drivers: none reported")
    if p.timing_guidance:
        lines.append(f"Timing: {_esc(p.timing_guidance)}")
    lines.append(track_record_line(card, settings))
    if p.is_late:
        lines.append(f"Recorded late (after {settings.ledger_lock_time:%H:%M} IST).")
    return lines


def morning_brief(
    calls: list[tuple[Prediction, Scorecard | None]],
    session: date,
    settings: Settings | None = None,
) -> str:
    """The 07:30 Telegram brief: one block per target, HTML-escaped, with the footer."""
    s = settings or get_settings()
    parts = [f"<b>Dalal market call — {_day(session)}</b>"]
    if not calls:
        parts.append("No market call was recorded for this session.")
    for p, card in calls:
        parts.append("\n".join(_call_block(p, card, s)))
    parts.append(f"<i>{FOOTER}</i>")
    return "\n\n".join(parts)


def _grade_line(p: Prediction, g: PredictionGrade) -> str:
    called = f"called <b>{_esc(p.predicted_class)}</b>"
    called += " (no edge)" if p.no_edge else f" ({_prob(_top_prob(p))})"
    if g.actual_class is None or g.actual_pct is None:
        return f"{_esc(p.target)}: {called}, close not graded yet"
    verdict = "correct" if g.correct else "wrong"
    line = (
        f"{_esc(p.target)}: {called}, actual {_pct(g.actual_pct)} "
        f"<b>{_esc(g.actual_class)}</b> — {verdict}. Brier {_num(g.brier)}"
    )
    if p.gap_call is not None and g.gap_correct is not None:
        gap_verdict = "correct" if g.gap_correct else "wrong"
        line += (
            f". Gap call {_esc(GAP_LABELS.get(p.gap_call, p.gap_call))}: {gap_verdict} "
            f"({_pct(g.gap_actual_pct)} actual)"
        )
    return line


def _top_prob(p: Prediction) -> float | None:
    return {"UP": p.prob_up, "FLAT": p.prob_flat, "DOWN": p.prob_down}.get(p.predicted_class)


def _num(x: float | None, dp: int = 2) -> str:
    return MISSING if x is None else f"{x:.{dp}f}"


def close_summary(grades: Iterable[tuple[Prediction, PredictionGrade]]) -> str:
    """The 15:45 Telegram summary: every call graded in this run, grouped by session."""
    by_session: dict[date, list[tuple[Prediction, PredictionGrade]]] = defaultdict(list)
    for p, g in grades:
        by_session[p.session_date].append((p, g))
    parts: list[str] = []
    for session in sorted(by_session):
        lines = [f"<b>Dalal close grading — {_day(session)}</b>"]
        lines += [_grade_line(p, g) for p, g in by_session[session]]
        parts.append("\n".join(lines))
    if not parts:
        parts.append("<b>Dalal close grading</b>\nNothing was graded.")
    parts.append(f"<i>{FOOTER}</i>")
    return "\n\n".join(parts)

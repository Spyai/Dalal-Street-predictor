"""Telegram texts for the market-call and stock-ideas loops: the morning brief, the
09:20 fill summary and the close summary.

Telegram renders the HTML parse mode, so every string that did not originate here
(target names, symbols, driver sentences, verdicts, timing guidance, outcome notes) goes
through `html.escape`. Numbers are formatted by hand; a missing number is shown as an em
dash, never a guess. The brief always carries the model's confidence or "No edge today",
the live track record beside the call, and the footer the brief asks for. It also names
the ledger heads (seq and the first 12 hex digits of the hash, one per chain) so the
Telegram chat is an independent witness of both chains: a later edit to the database
cannot rewrite what was sent that morning.

The ideas section (`ideas_section`) is one line per idea: symbol, horizon, setup, entry
zone, stop, target with the expected-return range, P(target before stop), EV after costs,
position size, and the live hit rate of that setup at that horizon so far ("unproven"
until there is one). A horizon with no idea gets its honest line ("No edge today", or why
the engine did not run). Surveillance (ASM/GSM) is never checked, and the section says so.
"""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date

from app.config import Settings, get_settings
from app.ml import ranking
from app.ml.core import horizons
from app.models import Idea, IdeaOutcome, PaperEquity, PaperTrade, Prediction, PredictionGrade
from app.services.ideas_scorecard import IdeaTrack
from app.services.scorecard import Scorecard, baseline_label

FOOTER = "Personal research tool. Not investment advice."
MISSING = "—"  # em dash
HASH_PREFIX_LEN = 12
TOP_DRIVERS_IN_BRIEF = 3
GAP_LABELS: dict[str, str] = {"gap_up": "gap up", "gap_down": "gap down", "flat": "flat open"}
IDEAS_HEADER = "<b>Stock ideas</b>"
# Honesty rule (Phase 3 contract): no ASM/GSM source is reachable; the reader must check.
SURVEILLANCE_LINE = "ASM/GSM surveillance status: unknown (not checked; verify before acting)."
PAPER_LINE = "Sizes are paper positions from the paper risk settings, not orders."


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
    # Paired comparison (scorecard.best_baseline): the model and the baseline scored on
    # the same sessions, so the two percentages are comparable.
    best = getattr(card, "best_baseline", None)
    line = f"Track record: {card.n_graded} graded, {card.accuracy * 100:.0f}% right"
    if best is not None and best.get("accuracy") is not None:
        model_acc = best.get("model_accuracy")
        paired = f"{float(model_acc) * 100:.0f}% vs " if model_acc is not None else ""
        line += (
            f"; toughest baseline ({_esc(baseline_label(str(best['name'])))}): "
            f"{paired}{float(best['accuracy']) * 100:.0f}% on the same "
            f"{int(best['n'])} call{'' if int(best['n']) == 1 else 's'}"
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


def ledger_anchor(head: Prediction | None) -> str | None:
    """ "Ledger head: seq N, <12 hex>" for the chain's last record, None for an empty ledger."""
    if head is None:
        return None
    return f"Ledger head: seq {head.seq}, {_esc(str(head.hash)[:HASH_PREFIX_LEN])}"


def idea_ledger_anchor(head: Idea | None) -> str | None:
    """ "Idea ledger head: seq N, <12 hex>" for the idea chain's last record, or None."""
    if head is None:
        return None
    return f"Idea ledger head: seq {head.seq}, {_esc(str(head.hash)[:HASH_PREFIX_LEN])}"


def _price(x: float | None) -> str:
    return MISSING if x is None else f"{x:,.2f}"


def _money(x: float | None) -> str:
    return MISSING if x is None else f"₹{x:,.0f}"


def idea_track_line(track: IdeaTrack | None, settings: Settings | None = None) -> str:
    """The live record of the idea's setup at its horizon, or "unproven"."""
    s = settings or get_settings()
    if track is None:
        return "track record: unavailable"
    label = f"{track.setup}/{track.horizon}" if track.setup else track.horizon
    if track.n_graded == 0 or track.hit_rate is None:
        return f"{_esc(label)} so far: unproven (no graded ideas yet)"
    n = track.n_graded
    line = f"{_esc(label)} so far: {track.hit_rate * 100:.0f}% hit target of {n} graded"
    if track.expectancy_pct is not None:
        line += f", {_pct(track.expectancy_pct)} net per idea"
    if track.sample_warning:
        line += f"; unproven (&lt; {s.min_sample_size} graded)"
    return line


def idea_line(i: Idea, track: IdeaTrack | None, settings: Settings | None = None) -> str:
    """One line per idea (contract): symbol, horizon, setup, entry zone, stop, target with
    the expected-return range, P(target before stop), EV after costs, size, track record."""
    qty = i.position_qty
    size = f"{qty} share{'' if qty == 1 else 's'} ≈ {_money(i.position_value)}"
    if qty <= 0:
        size = "no position (too small for the risk caps)"
    line = (
        f"• <b>{_esc(i.symbol)}</b> {_esc(i.horizon)}/{_esc(i.setup)}: "
        f"entry {_price(i.entry_low)}–{_price(i.entry_high)}, stop {_price(i.stop)}, "
        f"target {_price(i.target_1)} ({_pct(i.expected_return_low_pct, 1)} to "
        f"{_pct(i.expected_return_high_pct, 1)}), P(target before stop) {_prob(i.p_target)}, "
        f"EV {_pct(i.ev_pct_after_costs)} after costs, R:R {_num(i.risk_reward, 1)}, {size}; "
        f"{idea_track_line(track, settings)}"
    )
    if i.is_late:
        line += " [recorded late]"
    return line


def _no_edge_line(horizon: str, text: str) -> str:
    """Bold the verdict and escape the rest (the ranking's text quotes ">=")."""
    if text.startswith(ranking.NO_EDGE_TEXT):
        return f"<b>{ranking.NO_EDGE_TEXT}</b>{_esc(text[len(ranking.NO_EDGE_TEXT) :])}"
    return f"{_esc(horizon)}: {_esc(text)}"


def ideas_section(
    ideas: Iterable[tuple[Idea, IdeaTrack | None]],
    settings: Settings | None = None,
    *,
    notes: Mapping[str, str] | None = None,
) -> list[str]:
    """The ideas block: one line per idea grouped by horizon (`core.horizons` order), the
    horizon's note when it has none (`notes`, default "No edge today"), the surveillance
    and paper-size caveats."""
    s = settings or get_settings()
    by_horizon: dict[str, list[tuple[Idea, IdeaTrack | None]]] = defaultdict(list)
    for i, track in ideas:
        by_horizon[i.horizon].append((i, track))
    lines = [IDEAS_HEADER]
    order = [*horizons(s), *(h for h in by_horizon if h not in horizons(s))]
    for h in order:
        rows = sorted(by_horizon.get(h, []), key=lambda it: (it[0].rank, it[0].seq))
        if rows:
            lines += [idea_line(i, track, s) for i, track in rows]
        else:
            text = (notes or {}).get(h) or ranking.no_edge_text(h, s)
            lines.append(_no_edge_line(h, text))
    if by_horizon:
        lines.append(PAPER_LINE)
    lines.append(SURVEILLANCE_LINE)
    return lines


def morning_brief(
    calls: list[tuple[Prediction, Scorecard | None]],
    session: date,
    settings: Settings | None = None,
    *,
    head: Prediction | None = None,
    ideas: Iterable[tuple[Idea, IdeaTrack | None]] = (),
    idea_head: Idea | None = None,
    idea_notes: Mapping[str, str] | None = None,
) -> str:
    """The 07:30 Telegram brief: one block per target, the ideas section, HTML-escaped,
    with the footer.

    `head` is the prediction ledger's last record (`ledger.head`); without it the
    highest-seq call in `calls` stands in, so the anchor line is always present when
    something was recorded. The ideas section appears when there are `ideas` or when
    `idea_notes` is given (the engine ran: a horizon without an idea shows its note, "No
    edge today" by default); `idea_head` (`ledger.head_idea`) anchors the idea chain the
    same way, with the highest-seq idea standing in.
    """
    s = settings or get_settings()
    parts = [f"<b>Dalal market call — {_day(session)}</b>"]
    if not calls:
        parts.append("No market call was recorded for this session.")
    for p, card in calls:
        parts.append("\n".join(_call_block(p, card, s)))
    anchor = ledger_anchor(head or max((p for p, _ in calls), key=lambda p: p.seq, default=None))
    if anchor is not None:
        parts.append(anchor)
    idea_rows = list(ideas)
    if idea_rows or idea_notes is not None:
        parts.append("\n".join(ideas_section(idea_rows, s, notes=idea_notes)))
        idea_anchor = idea_ledger_anchor(
            idea_head or max((i for i, _ in idea_rows), key=lambda i: i.seq, default=None)
        )
        if idea_anchor is not None:
            parts.append(idea_anchor)
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


def _idea_head(i: Idea) -> str:
    return f"{_esc(i.symbol)} ({_esc(i.horizon)}, {_esc(i.setup)})"


def idea_outcome_line(i: Idea, o: IdeaOutcome) -> str:
    """What happened to one idea: settled (target / stop / expired) with the exit and the
    returns after costs, still open with its excursions, invalidated or void with why."""
    head = _idea_head(i)
    if o.status in ("target", "stop", "expired"):
        when = o.exit_date.isoformat() if o.exit_date else MISSING
        n = o.days_held
        held = MISSING if n is None else f"{n} session{'' if n == 1 else 's'}"
        return (
            f"{head}: <b>{_esc(o.status)}</b> on {when} at {_price(o.exit_price)} — "
            f"{_pct(o.gross_return_pct)} gross, {_pct(o.net_return_pct)} net after costs, "
            f"{_num(o.r_multiple)}R, {held}"
        )
    if o.status == "open":
        since = o.entry_date.isoformat() if o.entry_date else MISSING
        return (
            f"{head}: open since {since} at {_price(o.entry_price)}; best "
            f"{_pct(o.max_favorable_pct)}, worst {_pct(o.max_adverse_pct)}"
            + (f" ({_esc(o.notes)})" if o.notes else "")
        )
    why = f" — {_esc(o.notes)}" if o.notes else ""
    return f"{head}: <b>{_esc(o.status)}</b>{why}"


def equity_line(e: PaperEquity) -> str:
    n = e.open_positions
    line = (
        f"Paper equity {e.trade_date.isoformat()}: {_money(e.equity)} "
        f"(cash {_money(e.cash)}, invested {_money(e.invested)}, "
        f"{n} open position{'' if n == 1 else 's'})"
    )
    if e.nifty_close is not None:
        line += f"; Nifty 50 {_price(e.nifty_close)} (price index)"
    return line


def fill_summary(
    rows: Iterable[tuple[Idea, IdeaOutcome] | tuple[Idea, IdeaOutcome, PaperTrade | None]],
    session: date,
) -> str:
    """The 09:20 Telegram note: which ideas were filled at the open and which the gap
    invalidated (brief §3). A third element carries the paper trade actually opened: its
    quantity is what the note states (sized down to the book's cash when it was), and None
    means the book had no cash left, which the note says instead of inventing a position."""
    lines = [f"<b>Dalal open check — {_day(session)}</b>"]
    n = 0
    for row in rows:
        i, o = row[0], row[1]
        n += 1
        if o.status == "open":
            if len(row) > 2:
                trade = row[2]
                if trade is None:
                    paper_text = "no paper position: no cash left in the paper book"
                elif trade.qty < i.position_qty:
                    paper_text = (
                        f"{trade.qty} share{'' if trade.qty == 1 else 's'} paper "
                        f"(sized down from {i.position_qty}: cash)"
                    )
                else:
                    paper_text = f"{trade.qty} share{'' if trade.qty == 1 else 's'} paper"
            else:
                qty = i.position_qty
                paper_text = f"{qty} share{'' if qty == 1 else 's'} paper"
            lines.append(
                f"{_idea_head(i)}: <b>filled</b> at {_price(o.entry_price)} "
                f"(zone {_price(i.entry_low)}–{_price(i.entry_high)}), {paper_text}"
            )
        else:
            why = f": {_esc(o.notes)}" if o.notes else ""
            lines.append(f"{_idea_head(i)}: <b>{_esc(o.status)}</b>{why}")
    if n == 0:
        lines.append("No idea was waiting for the open.")
    return "\n\n".join(["\n".join(lines), f"<i>{FOOTER}</i>"])


def close_summary(
    grades: Iterable[tuple[Prediction, PredictionGrade]],
    *,
    ideas: Iterable[tuple[Idea, IdeaOutcome]] = (),
    equity: PaperEquity | None = None,
) -> str:
    """The 15:45 Telegram summary: every call graded in this run, grouped by session, then
    every idea whose outcome moved (settled, still open, invalidated) and the paper
    equity snapshot when one was written."""
    by_session: dict[date, list[tuple[Prediction, PredictionGrade]]] = defaultdict(list)
    for p, g in grades:
        by_session[p.session_date].append((p, g))
    parts: list[str] = []
    for session in sorted(by_session):
        lines = [f"<b>Dalal close grading — {_day(session)}</b>"]
        lines += [_grade_line(p, g) for p, g in by_session[session]]
        parts.append("\n".join(lines))
    idea_rows = list(ideas)
    if idea_rows or equity is not None:
        lines = ["<b>Ideas</b>"]
        lines += [idea_outcome_line(i, o) for i, o in idea_rows] or ["No idea outcome moved."]
        if equity is not None:
            lines.append(equity_line(equity))
        parts.append("\n".join(lines))
    if not parts:
        parts.append("<b>Dalal close grading</b>\nNothing was graded.")
    parts.append(f"<i>{FOOTER}</i>")
    return "\n\n".join(parts)

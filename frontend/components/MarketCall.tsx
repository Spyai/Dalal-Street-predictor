"use client";

import Link from "next/link";
import type { CallClass, Comparison, MarketCall, Track } from "@/lib/types";
import { fmtDate, fmtIstDateTime, fmtNum, fmtPct } from "@/lib/format";
import { Notice } from "@/components/Status";

// The market-call card (brief §7.1). Honesty first: "No edge today" is the loudest state,
// every missing number renders "—", and the live track record sits under every call. After
// the close the upcoming session has no call yet, so the card keeps the latest earlier call
// (flagged "previous") with its grade instead of going blank.

const CLASSES: CallClass[] = ["UP", "FLAT", "DOWN"];
const CLASS_TONE: Record<CallClass, string> = {
  UP: "text-up",
  FLAT: "text-muted",
  DOWN: "text-down",
};
const CLASS_BAR: Record<CallClass, string> = { UP: "bg-up", FLAT: "bg-faint", DOWN: "bg-down" };
const GAP_LABEL: Record<string, string> = {
  gap_up: "Gap up",
  gap_down: "Gap down",
  flat: "Flat open",
};
const CONFIDENCE_STYLE: Record<string, string> = {
  high: "border-up/40 text-up",
  medium: "border-accent/40 text-accent",
  low: "border-warn/40 text-warn",
  none: "border-warn/40 text-warn",
};

export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(0)}%`;
}

// Plain words for the paired comparison, shared by the Today card and the Scorecard page.
export function comparisonText(c: Comparison, pairedN: number | null, minPairedN: number): string {
  switch (c) {
    case "ahead":
      return "Ahead of the naive baselines on paired calls.";
    case "behind":
      return "NOT beating the naive baselines on paired calls.";
    case "level":
      return "Level with the naive baselines: no edge shown yet.";
    case "too_early":
      return `Too early to tell whether the model beats the baselines: ${pairedN ?? 0} of the ${minPairedN} paired calls needed.`;
    case "no_baseline":
      return "No baseline has a call to compare with yet.";
    case "no_calls":
      return "No graded calls yet.";
  }
}

export const COMPARISON_TONE: Record<Comparison, string> = {
  ahead: "text-up",
  behind: "text-down",
  level: "text-muted",
  too_early: "text-warn",
  no_baseline: "text-faint",
  no_calls: "text-faint",
};

export function MarketCallCard({
  call,
  upcomingSession,
}: {
  call: MarketCall;
  upcomingSession?: string;
}) {
  const top = call.probs[call.predicted_class] ?? null;
  const hasRange = call.range_p10_pct !== null && call.range_p90_pct !== null;
  return (
    <div className="card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="label">Market call: {call.target}</h2>
        <div className="flex items-center gap-1.5">
          {call.is_previous && (
            <Badge
              cls="border-line text-muted"
              title={`The latest recorded call; no call yet for ${upcomingSession ? fmtDate(upcomingSession) : "the upcoming session"}`}
            >
              previous · for {fmtDate(call.session_date)}
            </Badge>
          )}
          {call.is_late && (
            <Badge cls="border-warn/40 text-warn" title="Recorded after the 09:15 IST ledger lock">
              late
            </Badge>
          )}
          <Badge cls={CONFIDENCE_STYLE[call.confidence] ?? "border-line text-muted"}>
            {call.confidence === "none" ? "no edge" : `${call.confidence} confidence`}
          </Badge>
        </div>
      </div>

      {call.no_edge ? (
        <div className="rounded-md border border-warn/40 bg-warn/5 px-3 py-2">
          <p className="text-3xl font-semibold text-warn">No edge today</p>
          <p className="mt-1 text-sm text-muted">
            The class probabilities are too close to call. The model&apos;s lean (
            <span className={CLASS_TONE[call.predicted_class]}>{call.predicted_class}</span>,{" "}
            {fmtProb(top)}) is in the ledger and will be graded, but it is not a signal.
          </p>
        </div>
      ) : (
        <div className="flex flex-wrap items-baseline gap-x-3">
          <p className={`text-3xl font-semibold ${CLASS_TONE[call.predicted_class]}`}>
            {call.predicted_class}
          </p>
          <span className="num text-sm text-muted">{fmtProb(top)} probability</span>
        </div>
      )}
      <p className="mt-1 text-xs text-faint">
        Session {fmtDate(call.session_date)} · vs close {fmtNum(call.reference_close)} on{" "}
        {fmtDate(call.reference_date)} · recorded {fmtIstDateTime(call.as_of)} IST
        {call.model_version ? ` · ${call.model_version}` : ""}
      </p>
      {call.is_previous && (
        <p className="mt-1 text-xs text-muted">
          No call yet for {upcomingSession ? fmtDate(upcomingSession) : "the upcoming session"}:
          the 07:30 IST job records it before the open. Shown is the latest call, with its
          grade.
        </p>
      )}

      <div className="mt-3 space-y-1.5">
        {CLASSES.map((c) => {
          const p = call.probs[c] ?? null;
          return (
            <div key={c} className="flex items-center gap-2 text-xs">
              <span className={`w-12 ${CLASS_TONE[c]}`}>{c}</span>
              <div className="h-2 flex-1 rounded bg-panel-2">
                <div
                  className={`h-2 rounded ${CLASS_BAR[c]}`}
                  style={{ width: `${Math.round(Math.min(1, Math.max(0, p ?? 0)) * 100)}%` }}
                />
              </div>
              <span className="num w-10 text-right">{fmtProb(p)}</span>
            </div>
          );
        })}
      </div>

      <dl className="num mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <Row
          label="10–90% range"
          value={hasRange ? `${fmtPct(call.range_p10_pct)} to ${fmtPct(call.range_p90_pct)}` : "—"}
        />
        <Row
          label="Gap call"
          value={
            call.gap_call
              ? `${GAP_LABEL[call.gap_call] ?? call.gap_call} (${fmtPct(call.gap_expected_pct)} expected)`
              : "— (no overnight futures)"
          }
        />
      </dl>

      {call.drivers.length > 0 ? (
        <div className="mt-3">
          <div className="label mb-1">Top drivers</div>
          <ul className="space-y-0.5 text-sm">
            {call.drivers.map((d) => (
              <li key={d.feature} className="flex gap-2">
                <span
                  className={`num w-3 shrink-0 ${
                    d.contribution === null
                      ? "text-faint"
                      : d.contribution >= 0
                        ? "text-up"
                        : "text-down"
                  }`}
                  title={
                    d.contribution === null
                      ? "contribution unknown"
                      : `contribution ${d.contribution.toFixed(3)} to ${call.predicted_class}`
                  }
                >
                  {d.contribution === null ? "·" : d.contribution >= 0 ? "+" : "−"}
                </span>
                <span>{d.text}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="mt-3 text-xs text-faint">No driver breakdown recorded.</p>
      )}

      {call.timing_guidance && (
        <p className="mt-2 text-sm text-muted">
          <span className="label mr-1">Timing</span>
          {call.timing_guidance}
          <span className="text-faint"> (rule-based; not graded yet)</span>
        </p>
      )}

      <GradeLine call={call} />
      <TrackLine track={call.track} />
    </div>
  );
}

function GradeLine({ call }: { call: MarketCall }) {
  const g = call.grade;
  const closed = g !== null && g.actual_class !== null;
  const gapGraded = g !== null && g.gap_correct !== null;
  if (!closed && !gapGraded) {
    // A previous session's call that the 15:45 job has not graded yet: say so, don't hide it.
    if (call.is_previous) {
      return (
        <p className="mt-3 border-t border-line pt-2 text-xs text-muted">
          Not graded yet: the 15:45 IST close-grading job writes the outcome.
        </p>
      );
    }
    return null;
  }
  return (
    <div className="num mt-3 border-t border-line pt-2 text-sm">
      {closed && g && (
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-muted">Graded:</span>
          <span className={CLASS_TONE[g.actual_class ?? "FLAT"]}>
            {g.actual_class} {fmtPct(g.actual_pct)}
          </span>
          <Verdict ok={g.correct} />
          <span className="text-xs text-muted">
            Brier {fmtNum(g.brier, 3)} · log loss {fmtNum(g.log_loss, 3)}
          </span>
        </div>
      )}
      {gapGraded && g && (
        <div className="mt-0.5 flex flex-wrap items-baseline gap-x-2 text-xs">
          <span className="text-muted">Open gap</span>
          <span>{fmtPct(g.gap_actual_pct)}</span>
          <Verdict ok={g.gap_correct} />
        </div>
      )}
    </div>
  );
}

function Verdict({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="text-faint">—</span>;
  return ok ? <span className="text-up">correct</span> : <span className="text-down">wrong</span>;
}

function TrackLine({ track }: { track: Track | null }) {
  if (!track) {
    return <p className="mt-3 border-t border-line pt-2 text-xs text-faint">Track record: —</p>;
  }
  const n = track.n_graded;
  const paired =
    track.best_baseline_label !== null &&
    track.best_baseline_accuracy !== null &&
    track.paired_n !== null;
  return (
    <div className="mt-3 border-t border-line pt-2">
      <p className="num text-xs text-muted">
        Track record: {n} graded
        {n > 0 && track.accuracy !== null ? ` · ${fmtProb(track.accuracy)} right` : ""}
        {" · "}
        <Link href="/scorecard" className="underline hover:text-fg">
          Scorecard
        </Link>
      </p>
      {paired && (
        <p className="num mt-0.5 text-xs text-muted">
          Toughest baseline ({track.best_baseline_label}) {fmtProb(track.best_baseline_accuracy)}{" "}
          vs model {fmtProb(track.model_paired_accuracy)} on the same {track.paired_n} call
          {track.paired_n === 1 ? "" : "s"}
        </p>
      )}
      {n > 0 && (
        <p className={`mt-0.5 text-xs ${COMPARISON_TONE[track.comparison]}`}>
          {comparisonText(track.comparison, track.paired_n, track.min_paired_n)}
        </p>
      )}
      {track.sample_warning && (
        <div className="mt-2">
          <Notice tone="warn">
            Sample size: {n} of {track.min_sample_size} graded calls. Treat the model as unproven
            until it has {track.min_sample_size}.
          </Notice>
        </div>
      )}
    </div>
  );
}

export function MarketCallEmpty({ note }: { note: string }) {
  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="label">Market call</h2>
        <Badge cls="border-line text-muted">no call</Badge>
      </div>
      <p className="text-2xl font-semibold text-faint">No call recorded</p>
      <p className="mt-2 text-sm text-muted">{note}</p>
      <p className="mt-2 text-xs text-faint">
        A call shows UP / FLAT / DOWN probabilities, the 10–90% range, the gap call, its drivers,
        and the live track record against four naive baselines. Nothing is shown that was not
        written to the ledger.
      </p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="text-right">{value}</dd>
    </>
  );
}

function Badge({ cls, title, children }: { cls: string; title?: string; children: React.ReactNode }) {
  return (
    <span
      title={title}
      className={`inline-block rounded border px-1.5 py-px text-[10px] uppercase tracking-wide ${cls}`}
    >
      {children}
    </span>
  );
}

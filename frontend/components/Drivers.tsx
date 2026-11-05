"use client";

import Link from "next/link";
import { useState } from "react";
import type { Driver, Factors, IdeaStatus, IdeaTrack } from "@/lib/types";
import { fmtNum } from "@/lib/format";
import { Notice } from "@/components/Status";

// The reasoning behind a market call or a stock idea (brief §4.1, §4.2, §7.1, §7.3): the
// top `drivers_top_n` drivers as recorded in the ledger, and one click away the full
// factor list, every feature of the recorded snapshot with the model's contribution.
// A factor the ledger holds no contribution for shows "—": nothing is estimated after the
// fact. The idea atoms below (status pill, surveillance badge, track line) are shared by
// the Today, Ideas and Scorecard pages so an idea reads the same everywhere.

export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(0)}%`;
}

// A feature value as it reads in the factor table: integers as they are, large numbers
// with one decimal, small ones with up to three.
export function fmtFactorValue(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return v.toString();
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function fmtContribution(c: number | null): string {
  if (c === null || Number.isNaN(c)) return "—";
  return `${c >= 0 ? "+" : "−"}${Math.abs(c).toFixed(3)}`;
}

function Sign({ contribution, target }: { contribution: number | null; target: string }) {
  return (
    <span
      className={`num w-3 shrink-0 ${
        contribution === null ? "text-faint" : contribution >= 0 ? "text-up" : "text-down"
      }`}
      title={
        contribution === null
          ? "contribution unknown"
          : `contribution ${contribution.toFixed(3)} towards ${target}`
      }
    >
      {contribution === null ? "·" : contribution >= 0 ? "+" : "−"}
    </span>
  );
}

export function Drivers({
  drivers,
  target,
  factors,
  loadFactors,
  topN,
  title = "Top drivers",
}: {
  drivers: Driver[]; // the recorded drivers (the card shows these)
  target: string; // what a positive contribution pushes towards ("UP", "target before stop")
  factors?: Factors | null; // the full list, when the page already has it (idea detail)
  loadFactors?: () => Promise<Factors>; // or how to fetch it on demand (market call)
  topN?: number; // settings.drivers_top_n, for the heading
  title?: string;
}) {
  const [open, setOpen] = useState(false);
  const [loaded, setLoaded] = useState<Factors | null>(factors ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const full = factors ?? loaded;
  const canExpand = Boolean(factors) || Boolean(loadFactors);

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && !full && loadFactors) {
      setLoading(true);
      setError(null);
      try {
        setLoaded(await loadFactors());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load the factors");
      } finally {
        setLoading(false);
      }
    }
  }

  const shown = full ? full.drivers : drivers;
  const n = full?.n_features ?? null;
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <div className="label">
          {title}
          {topN !== undefined && shown.length > 0 ? ` (top ${Math.min(topN, shown.length)})` : ""}
        </div>
        {canExpand && (
          <button
            onClick={toggle}
            className="text-xs text-muted underline-offset-2 hover:text-fg hover:underline"
            aria-expanded={open}
          >
            {open ? "Hide all factors" : n !== null ? `All factors (${n})` : "All factors"}
          </button>
        )}
      </div>
      {shown.length > 0 ? (
        <ul className="space-y-0.5 text-sm">
          {shown.map((d) => (
            <li key={d.feature} className="flex gap-2">
              <Sign contribution={d.contribution} target={target} />
              <span>{d.text}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-faint">No driver breakdown recorded.</p>
      )}
      {open && (
        <div className="mt-2 rounded-md border border-line bg-panel-2/40 p-2">
          {loading && <p className="text-xs text-muted">Loading the factor list…</p>}
          {error && <p className="text-xs text-down">{error}</p>}
          {full && <FactorTable factors={full} target={target} />}
        </div>
      )}
    </div>
  );
}

function FactorTable({ factors, target }: { factors: Factors; target: string }) {
  const max = Math.max(
    1e-9,
    ...factors.all_factors.map((f) => (f.contribution === null ? 0 : Math.abs(f.contribution))),
  );
  return (
    <div>
      <p className="mb-1 text-[11px] text-faint">
        {factors.n_features} feature{factors.n_features === 1 ? "" : "s"} in the recorded snapshot;{" "}
        {factors.n_with_contribution} with a recorded contribution towards {target}. A factor
        without one was in the model&apos;s input but the ledger holds no contribution for it, so
        none is shown.
      </p>
      <div className="max-h-80 overflow-y-auto">
        <table className="num w-full text-xs">
          <thead className="sticky top-0 bg-panel text-left text-faint">
            <tr>
              <th className="py-1 font-normal">Factor</th>
              <th className="py-1 text-right font-normal">Value</th>
              <th className="py-1 pl-2 text-right font-normal">Contribution</th>
              <th className="w-1/4 pl-2" />
            </tr>
          </thead>
          <tbody>
            {factors.all_factors.map((f) => {
              const c = f.contribution;
              const w = c === null ? 0 : Math.round((Math.abs(c) / max) * 100);
              return (
                <tr key={f.feature} className="border-t border-line align-top">
                  <td className="py-1 pr-2" title={f.text ?? f.feature}>
                    <span>{f.label}</span>
                    <span className="ml-1 text-[10px] text-faint">{f.feature}</span>
                  </td>
                  <td className="py-1 text-right">{fmtFactorValue(f.value)}</td>
                  <td
                    className={`py-1 pl-2 text-right ${
                      c === null ? "text-faint" : c >= 0 ? "text-up" : "text-down"
                    }`}
                  >
                    {fmtContribution(c)}
                  </td>
                  <td className="py-1 pl-2">
                    {c !== null && (
                      <div className="flex h-1.5 w-full items-center">
                        <div className="h-1.5 w-1/2">
                          {c < 0 && (
                            <div
                              className="ml-auto h-1.5 rounded-l bg-down"
                              style={{ width: `${w}%` }}
                            />
                          )}
                        </div>
                        <div className="h-1.5 w-1/2">
                          {c >= 0 && <div className="h-1.5 rounded-r bg-up" style={{ width: `${w}%` }} />}
                        </div>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- idea atoms

export const HORIZON_LABEL: Record<string, string> = {
  swing: "Swing",
  positional: "Positional",
};

const STATUS_STYLE: Record<IdeaStatus, string> = {
  pending: "border-line text-faint",
  open: "border-accent/40 text-accent",
  target: "border-up/40 text-up",
  stop: "border-down/40 text-down",
  expired: "border-line text-muted",
  invalidated: "border-warn/40 text-warn",
  void: "border-line text-faint",
};

const STATUS_LABEL: Record<IdeaStatus, string> = {
  pending: "pending entry",
  open: "open",
  target: "target hit",
  stop: "stopped",
  expired: "expired",
  invalidated: "invalidated",
  void: "void",
};

export function IdeaStatusPill({ status, title }: { status: IdeaStatus; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-block whitespace-nowrap rounded border px-1.5 py-px text-[10px] uppercase tracking-wide ${
        STATUS_STYLE[status] ?? "border-line text-muted"
      }`}
    >
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

// Surveillance (ASM / GSM / trade-to-trade) is never checked: no source is reachable from
// the job host. The badge says so instead of pretending the stock was screened.
export function SurveillanceBadge({ note }: { note: string }) {
  return (
    <span
      title={note}
      className="inline-block whitespace-nowrap rounded border border-warn/40 px-1.5 py-px text-[10px] uppercase tracking-wide text-warn"
    >
      surveillance not checked (NSE unreachable)
    </span>
  );
}

export function fmtRupees(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v < 0 ? "−" : "";
  return `${sign}₹${Math.abs(v).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

// The live record of the idea's setup at its horizon, beside its p_target (honesty rule:
// every probability with its track record, and the sample-size warning below the minimum).
export function IdeaTrackLine({ track, compact = false }: { track: IdeaTrack; compact?: boolean }) {
  const where = `${track.setup ? `${track.setup.replace(/_/g, " ")} ` : ""}${track.horizon}`;
  const n = track.n_graded;
  return (
    <div className={compact ? "" : "mt-2 border-t border-line pt-2"}>
      <p className="num text-xs text-muted">
        Track record ({where}): {n} graded
        {n > 0 ? ` · target first ${fmtProb(track.hit_rate)}` : ""}
        {n > 0 && track.expectancy_pct !== null
          ? ` · ${track.expectancy_pct >= 0 ? "+" : ""}${track.expectancy_pct.toFixed(2)}% per idea after costs`
          : ""}
        {" · "}
        <Link href="/scorecard#ideas" className="underline hover:text-fg">
          Scorecard
        </Link>
      </p>
      {track.sample_warning && !compact && (
        <div className="mt-1">
          <Notice tone="warn">
            Sample size: {n} of {track.min_sample_size} graded ideas for this setup and horizon.
            Treat the probability as unproven until it has {track.min_sample_size}.
          </Notice>
        </div>
      )}
      {track.sample_warning && compact && (
        <p className="text-[11px] text-warn">
          Unproven: {n} of {track.min_sample_size} graded ideas.
        </p>
      )}
    </div>
  );
}

export function fmtLevel(v: number | null | undefined): string {
  return fmtNum(v, 2);
}

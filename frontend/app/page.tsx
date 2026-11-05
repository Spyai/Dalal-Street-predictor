"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Cue, Flows, HorizonIdeas, Idea, KiteStatus, MarketClock, Today, TodayIdeas } from "@/lib/types";
import { ago, fmtCr, fmtDate, fmtIstDateTime, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice, StatusPill } from "@/components/Status";
import { MarketCallCard, MarketCallEmpty } from "@/components/MarketCall";
import {
  HORIZON_LABEL,
  IdeaStatusPill,
  IdeaTrackLine,
  SurveillanceBadge,
  fmtLevel,
  fmtProb,
  fmtRupees,
} from "@/components/Drivers";

const REFRESH_MS = 60_000;

export default function TodayPage() {
  const [data, setData] = useState<Today | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(() => {
    api<Today>("/api/today")
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  async function refreshCues() {
    setRefreshing(true);
    try {
      await api("/api/today/refresh-cues", { method: "POST" });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
      load();
    }
  }

  if (!data) {
    return <p className="text-sm text-muted">{error ?? "Loading…"}</p>;
  }

  const india = data.cues.filter((c) => c.group === "India");
  const global = data.cues.filter((c) => c.group !== "India");

  return (
    <div className="space-y-5">
      <ClockBar clock={data.clock} />
      {error && <Notice tone="down">{error}</Notice>}
      <KiteBanner kite={data.kite} />

      <section className="space-y-2">
        <div className="grid gap-4 md:grid-cols-2">
          {data.market_call.length === 0 ? (
            <MarketCallEmpty note={data.market_call_note} />
          ) : (
            data.market_call.map((c) => (
              <MarketCallCard key={c.id} call={c} upcomingSession={data.market_call_session} />
            ))
          )}
        </div>
        {data.market_call.length > 0 && data.market_call_note && (
          <p className="text-xs text-warn">{data.market_call_note}</p>
        )}
      </section>

      <IdeasSection ideas={data.ideas} upcomingSession={data.market_call_session} />

      <FlowsCard flows={data.flows} />

      <section className="space-y-2">
        <div className="flex items-baseline justify-between">
          <h2 className="label">India</h2>
        </div>
        <CueGrid cues={india} />
      </section>

      <section className="space-y-2">
        <div className="flex items-baseline justify-between">
          <h2 className="label">Overnight global cues</h2>
          <button
            onClick={refreshCues}
            disabled={refreshing}
            className="text-xs text-muted hover:text-fg disabled:opacity-40"
          >
            {refreshing ? "Refreshing…" : "Refresh now"}
          </button>
        </div>
        <CueGrid cues={global} />
      </section>

      <section className="card p-4">
        <h2 className="label mb-1">Daily brief</h2>
        <p className="text-sm text-muted">
          The one-paragraph AI brief is not built yet. The 07:30 Telegram brief carries the market
          call above; the written summary will cover the numbers on this page and the model&apos;s
          call. It will not invent a view of its own.
        </p>
      </section>
    </div>
  );
}

// The day's stock ideas per horizon (brief §4.2, §7.1). "No edge today" is the loudest
// state and is only claimed when the engine ran and recorded nothing; every probability
// carries its live track record; surveillance is marked as not checked on every idea.
function IdeasSection({ ideas, upcomingSession }: { ideas: TodayIdeas; upcomingSession: string }) {
  return (
    <section className="card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="label">
          Stock ideas · session {fmtDate(ideas.session)}
          {ideas.is_previous ? " (previous)" : ""}
        </h2>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span>
            bar: P(target) ≥ {fmtProb(ideas.min_p_target)} · EV ≥ {ideas.min_ev_pct}% after costs · at
            most {ideas.max_per_horizon} per horizon
          </span>
          <Link href="/ideas" className="underline hover:text-fg">
            All ideas
          </Link>
        </div>
      </div>
      {ideas.is_previous && (
        <p className="mb-2 text-xs text-muted">
          No idea yet for {fmtDate(upcomingSession)}: the 07:30 IST job records them right after
          the market calls. Shown are the latest session&apos;s ideas with their outcomes.
        </p>
      )}
      {ideas.n_ideas === 0 && ideas.note && (
        <div className="mb-3">
          {ideas.note.startsWith("No edge today") ? (
            <div className="rounded-md border border-warn/40 bg-warn/5 px-3 py-2">
              <p className="text-2xl font-semibold text-warn">No edge today</p>
              <p className="mt-1 text-sm text-muted">{ideas.note}</p>
            </div>
          ) : (
            <Notice tone="info">{ideas.note}</Notice>
          )}
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        {ideas.by_horizon.map((g) => (
          <HorizonGroup key={g.horizon} group={g} />
        ))}
      </div>
      <p className="mt-3 text-[11px] text-warn">{ideas.surveillance_note}</p>
    </section>
  );
}

function HorizonGroup({ group }: { group: HorizonIdeas }) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <h3 className="text-sm font-medium">
          {HORIZON_LABEL[group.horizon] ?? group.horizon}
          <span className="ml-2 text-xs text-muted">
            {group.horizon_days} sessions · target {group.target_atr}× ATR · stop {group.stop_atr}× ATR
          </span>
        </h3>
        <span className="text-[11px] text-faint">
          {group.champion_version ? `model ${group.champion_version}` : "no model"}
        </span>
      </div>
      {group.ideas.length === 0 ? (
        group.no_edge ? (
          <div className="rounded-md border border-warn/40 bg-warn/5 px-3 py-2">
            <p className="font-semibold text-warn">No edge today</p>
            <p className="mt-0.5 text-xs text-muted">{group.note}</p>
          </div>
        ) : (
          <p className="rounded-md border border-line px-3 py-2 text-xs text-muted">{group.note}</p>
        )
      ) : (
        <div className="space-y-2">
          {group.ideas.map((i) => (
            <IdeaCard key={i.id} idea={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function IdeaCard({ idea }: { idea: Idea }) {
  const o = idea.outcome;
  return (
    <div className="rounded-md border border-line p-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <Link href={`/ideas?seq=${idea.seq}`} className="text-lg font-semibold hover:underline">
            {idea.symbol}
          </Link>
          <span className="text-xs text-muted">
            #{idea.rank} · {idea.setup_label} · {idea.strategy_label}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {idea.is_late && (
            <span className="rounded border border-warn/40 px-1.5 py-px text-[10px] uppercase text-warn">
              late
            </span>
          )}
          <IdeaStatusPill status={o.status} title={o.notes ?? undefined} />
        </div>
      </div>
      <dl className="num mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs sm:grid-cols-3">
        <IdeaRow label="Entry zone" value={`${fmtLevel(idea.entry_low)}–${fmtLevel(idea.entry_high)}`} />
        <IdeaRow label="Stop" value={fmtLevel(idea.stop)} tone="text-down" />
        <IdeaRow
          label="Targets"
          value={`${fmtLevel(idea.target_1)}${idea.target_2 !== null ? ` / ${fmtLevel(idea.target_2)}` : ""}`}
          tone="text-up"
        />
        <IdeaRow
          label="Expected return"
          value={`${fmtPct(idea.expected_return_low_pct)} to ${fmtPct(idea.expected_return_high_pct)}`}
        />
        <IdeaRow label="P(target before stop)" value={fmtProb(idea.p_target)} />
        <IdeaRow label="EV after costs" value={fmtPct(idea.ev_pct_after_costs)} tone={tone(idea.ev_pct_after_costs)} />
        <IdeaRow label="Risk / reward" value={`${fmtNum(idea.risk_reward, 2)} : 1`} />
        <IdeaRow label="Position" value={`${idea.position_qty} sh · ${fmtRupees(idea.position_value)}`} />
        <IdeaRow label="Risk" value={fmtRupees(idea.risk_amount)} />
      </dl>
      {o.status !== "pending" && (
        <p className="num mt-1 text-xs text-muted">
          {o.entry_price !== null ? `Entered ${fmtLevel(o.entry_price)}` : "Not entered"}
          {o.exit_price !== null ? ` · exit ${fmtLevel(o.exit_price)} on ${fmtDate(o.exit_date)}` : ""}
          {o.net_return_pct !== null ? (
            <>
              {" · net "}
              <span className={tone(o.net_return_pct)}>{fmtPct(o.net_return_pct)}</span>
            </>
          ) : (
            ""
          )}
          {o.notes && o.status === "invalidated" ? ` · ${o.notes}` : ""}
        </p>
      )}
      <IdeaTrackLine track={idea.track} compact />
      <div className="mt-1.5">
        <SurveillanceBadge note={idea.surveillance_note} />
      </div>
    </div>
  );
}

function IdeaRow({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted">{label}</dt>
      <dd className={`text-right ${tone}`}>{value}</dd>
    </div>
  );
}

const SESSION_LABEL: Record<MarketClock["session"], string> = {
  pre_open: "Pre-open",
  open: "Market open",
  closed: "Market closed",
  holiday: "Holiday",
  weekend: "Weekend",
};

function ClockBar({ clock }: { clock: MarketClock }) {
  const open = clock.session === "open" || clock.session === "pre_open";
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <h1 className="text-xl font-semibold">{fmtDate(clock.now_ist)}</h1>
        <span className={`text-sm ${open ? "text-up" : "text-muted"}`}>
          ● {SESSION_LABEL[clock.session]}
          {clock.day_reason && clock.session === "holiday" ? ` (${clock.day_reason})` : ""}
        </span>
        <span className="text-sm text-muted">Next session {fmtDate(clock.next_trading_day)}</span>
      </div>
      {!clock.calendar_verified && (
        <Notice>
          NSE holiday calendar for this year isn&apos;t loaded, so every weekday counts as a trading
          day. Check <a className="underline" href="/health">Health</a>.
        </Notice>
      )}
    </div>
  );
}

function KiteBanner({ kite }: { kite: KiteStatus }) {
  if (!kite.configured) {
    return (
      <Notice tone="info">
        Kite Connect isn&apos;t configured. Live quotes, intraday data and holdings are off. Set{" "}
        <code>KITE_API_KEY</code> and <code>KITE_API_SECRET</code> (see SETUP.md). Redirect URL:{" "}
        <code className="break-all">{kite.redirect_url}</code>
      </Notice>
    );
  }
  if (!kite.connected) {
    return (
      <Notice>
        <span className="mr-2">
          No Kite session today. Broker data is off until you log in (tokens expire ~06:00 IST).
        </span>
        <a href="/api/kite/login" className="font-medium text-accent underline">
          Log in to Kite
        </a>
      </Notice>
    );
  }
  return (
    <p className="text-xs text-muted">
      Kite connected ({kite.kite_user_id}) until {fmtIstDateTime(kite.expires_at)} IST
    </p>
  );
}

function FlowsCard({ flows }: { flows: Flows }) {
  const latest = flows.days[0];
  const max = Math.max(
    1,
    ...flows.days.flatMap((d) => [Math.abs(d.fii_net_cr ?? 0), Math.abs(d.dii_net_cr ?? 0)]),
  );
  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="label">FII / DII cash flows</h2>
        <StatusPill status={flows.status} />
      </div>
      {!latest ? (
        <p className="text-sm text-muted">
          Missing. {flows.note} Fills in after the 19:30 evening pull (source: NSE).
        </p>
      ) : (
        <>
          <div className="mb-3 grid grid-cols-2 gap-4">
            <Stat label={`FII net · ${fmtDate(latest.trade_date)}`} value={latest.fii_net_cr} />
            <Stat label={`DII net · ${fmtDate(latest.trade_date)}`} value={latest.dii_net_cr} />
          </div>
          {flows.note && <p className="mb-2 text-xs text-warn">{flows.note}</p>}
          <table className="num w-full text-xs">
            <thead className="text-faint">
              <tr>
                <th className="py-1 text-left font-normal">Date</th>
                <th className="py-1 text-right font-normal">FII</th>
                <th className="w-1/4" />
                <th className="py-1 text-right font-normal">DII</th>
                <th className="w-1/4" />
              </tr>
            </thead>
            <tbody>
              {flows.days.map((d) => (
                <tr key={d.trade_date} className="border-t border-line">
                  <td className="py-1 text-muted">{fmtDate(d.trade_date)}</td>
                  <td className={`py-1 text-right ${tone(d.fii_net_cr)}`}>{fmtCr(d.fii_net_cr)}</td>
                  <td className="px-2">
                    <Bar v={d.fii_net_cr} max={max} />
                  </td>
                  <td className={`py-1 text-right ${tone(d.dii_net_cr)}`}>{fmtCr(d.dii_net_cr)}</td>
                  <td className="px-2">
                    <Bar v={d.dii_net_cr} max={max} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-faint">
            NSE provisional figures, ₹ crore. Source: {latest.source}.
          </p>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div>
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`num text-xl font-semibold ${tone(value)}`}>{fmtCr(value)}</div>
    </div>
  );
}

function Bar({ v, max }: { v: number | null; max: number }) {
  if (v === null) return null;
  const w = `${Math.round((Math.abs(v) / max) * 100)}%`;
  return (
    <div className="h-1.5 w-full rounded bg-panel-2">
      <div className={`h-1.5 rounded ${v >= 0 ? "bg-up" : "bg-down"}`} style={{ width: w }} />
    </div>
  );
}

function CueGrid({ cues }: { cues: Cue[] }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
      {cues.map((c) => (
        <CueTile key={c.key} cue={c} />
      ))}
    </div>
  );
}

function CueTile({ cue }: { cue: Cue }) {
  const unavailable = cue.status === "unavailable" || cue.last === null;
  const digits = cue.unit === "fx" ? 3 : 2;
  const change =
    cue.unit === "yield_pct" && cue.change_bps !== null
      ? `${cue.change_bps > 0 ? "+" : ""}${cue.change_bps.toFixed(1)} bp`
      : fmtPct(cue.change_pct);
  const changeVal = cue.unit === "yield_pct" ? cue.change_bps : cue.change_pct;
  return (
    <div
      className="card p-3"
      title={cue.note ?? (cue.as_of ? `As of ${fmtIstDateTime(cue.as_of)} IST · ${cue.source}` : "")}
    >
      <div className="truncate text-xs text-muted">{cue.label}</div>
      {unavailable ? (
        <>
          <div className="num mt-1 text-lg text-faint">—</div>
          <div className="mt-1">
            <StatusPill status="unavailable" />
          </div>
          <div className="mt-1 line-clamp-2 text-[11px] text-faint">{cue.note ?? "no data"}</div>
        </>
      ) : (
        <>
          <div className="num mt-1 text-lg font-semibold">
            {fmtNum(cue.last, digits)}
            {cue.unit === "yield_pct" ? "%" : ""}
          </div>
          <div className="flex items-baseline justify-between">
            <span className={`num text-xs ${tone(changeVal)}`}>{change}</span>
            <span className="text-[10px] text-faint">{ago(cue.as_of)}</span>
          </div>
          {cue.status === "stale" && (
            <div className="mt-1 flex items-center gap-1">
              <StatusPill status="stale" />
              <span className="text-[10px] text-faint">as of {fmtIstDateTime(cue.as_of)}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

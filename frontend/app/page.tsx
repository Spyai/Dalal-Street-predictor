"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Cue, Flows, KiteStatus, MarketClock, Today } from "@/lib/types";
import { ago, fmtCr, fmtDate, fmtIstDateTime, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice, StatusPill } from "@/components/Status";
import { MarketCallCard, MarketCallEmpty } from "@/components/MarketCall";

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
            data.market_call.map((c) => <MarketCallCard key={c.id} call={c} />)
          )}
        </div>
        {data.market_call.length > 0 && data.market_call_note && (
          <p className="text-xs text-warn">{data.market_call_note}</p>
        )}
      </section>

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

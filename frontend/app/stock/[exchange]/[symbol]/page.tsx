"use client";

import { use, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, ApiError } from "@/lib/api";
import type { Stock } from "@/lib/types";
import { fmtDate, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice, StatusPill } from "@/components/Status";
import SymbolSearch from "@/components/SymbolSearch";
import type { Overlays } from "@/components/PriceChart";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false });

const RANGES = ["6m", "1y", "3y", "5y"] as const;
const OVERLAY_LABELS: [keyof Overlays, string][] = [
  ["sma20", "SMA 20"],
  ["sma50", "SMA 50"],
  ["sma200", "SMA 200"],
  ["bollinger", "Bollinger"],
  ["levels", "S/R levels"],
  ["rsi", "RSI 14"],
  ["macd", "MACD"],
];

export default function StockPage({
  params,
}: {
  params: Promise<{ exchange: string; symbol: string }>;
}) {
  const { exchange, symbol: rawSymbol } = use(params);
  const symbol = decodeURIComponent(rawSymbol);
  const [range, setRange] = useState<(typeof RANGES)[number]>("1y");
  const [data, setData] = useState<Stock | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [overlays, setOverlays] = useState<Overlays>({
    sma20: false,
    sma50: true,
    sma200: true,
    bollinger: false,
    levels: true,
    rsi: true,
    macd: false,
  });

  useEffect(() => {
    setLoading(true);
    setError(null);
    api<Stock>(`/api/stocks/${exchange}/${encodeURIComponent(symbol)}?range=${range}`)
      .then(setData)
      .catch((e: unknown) => {
        setData(null);
        setError(e instanceof ApiError ? e.message : "Failed to load");
      })
      .finally(() => setLoading(false));
  }, [exchange, symbol, range]);

  const last = data?.candles.at(-1);
  const stableOverlays = useMemo(() => overlays, [overlays]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex flex-wrap items-baseline gap-x-3">
            <h1 className="text-xl font-semibold">{symbol}</h1>
            <span className="text-sm text-muted">
              {exchange}
              {data?.instrument.name ? ` · ${data.instrument.name}` : ""}
              {data?.instrument.industry ? ` · ${data.instrument.industry}` : ""}
            </span>
          </div>
          {data && (
            <div className="num mt-1 flex flex-wrap items-baseline gap-x-3">
              <span className="text-2xl font-semibold">{fmtNum(data.last_close)}</span>
              <span className={`text-sm ${tone(data.change_pct)}`}>{fmtPct(data.change_pct)}</span>
              <span className="text-xs text-muted">
                close {fmtDate(data.data_as_of)} · {data.sources.join(", ")}
              </span>
              {data.data_status !== "ok" && <StatusPill status={data.data_status} />}
            </div>
          )}
        </div>
        <SymbolSearch />
      </div>

      {error && <Notice tone="down">{error}</Notice>}
      {data?.notes.map((n) => (
        <Notice key={n}>{n}</Notice>
      ))}

      <div className="card p-3">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
          <div className="flex overflow-hidden rounded border border-line">
            {RANGES.map((r) => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className={`px-2.5 py-1 uppercase ${r === range ? "bg-panel-2 text-fg" : "text-muted hover:text-fg"}`}
              >
                {r}
              </button>
            ))}
          </div>
          {OVERLAY_LABELS.map(([key, label]) => (
            <label key={key} className="flex cursor-pointer items-center gap-1 text-muted">
              <input
                type="checkbox"
                checked={overlays[key]}
                onChange={() => setOverlays((o) => ({ ...o, [key]: !o[key] }))}
                className="accent-[var(--accent)]"
              />
              {label}
            </label>
          ))}
          {loading && <span className="ml-auto text-faint">Loading…</span>}
        </div>
        {data && data.candles.length > 0 ? (
          <PriceChart
            candles={data.candles}
            indicators={data.indicators}
            levels={data.levels}
            overlays={stableOverlays}
          />
        ) : (
          <div className="flex h-[300px] items-center justify-center text-sm text-muted">
            {loading ? "Loading…" : "No price data."}
          </div>
        )}
      </div>

      {data && last && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Mini label="Open" value={fmtNum(last.open)} />
          <Mini label="High / Low" value={`${fmtNum(last.high)} / ${fmtNum(last.low)}`} />
          <Mini
            label="Volume"
            value={last.volume === null ? "—" : last.volume.toLocaleString("en-IN")}
          />
          <Mini
            label="Delivery %"
            value={last.delivery_pct === null ? "missing" : `${last.delivery_pct.toFixed(1)}%`}
          />
          <Mini
            label="Supports"
            value={
              data.levels
                .filter((l) => l.kind === "support")
                .map((l) => fmtNum(l.price))
                .join(", ") || "—"
            }
          />
          <Mini
            label="Resistances"
            value={
              data.levels
                .filter((l) => l.kind === "resistance")
                .map((l) => fmtNum(l.price))
                .join(", ") || "—"
            }
          />
        </div>
      )}

      <div className="card p-4 text-sm text-muted">
        <h2 className="label mb-1">Coming in later phases</h2>
        Fundamentals (PE, PB, ROE, ROCE, D/E, growth, promoter holding and pledge), quarterly
        results, shareholding trend, corporate actions, news with sentiment and F&amp;O data arrive
        in Phase 5. This stock&apos;s prediction history arrives with the ledger in Phase 2/3.
      </div>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="card px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num truncate text-sm">{value}</div>
    </div>
  );
}

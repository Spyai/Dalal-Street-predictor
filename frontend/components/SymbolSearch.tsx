"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { InstrumentInfo } from "@/lib/types";

export default function SymbolSearch() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<InstrumentInfo[]>([]);
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!q.trim()) {
      setResults([]);
      return;
    }
    timer.current = setTimeout(() => {
      api<InstrumentInfo[]>(`/api/stocks/search?q=${encodeURIComponent(q.trim())}`)
        .then(setResults)
        .catch(() => setResults([]));
    }, 200);
  }, [q]);

  const go = (exchange: string, symbol: string) => {
    setOpen(false);
    setQ("");
    router.push(`/stock/${exchange}/${encodeURIComponent(symbol)}`);
  };

  return (
    <div className="relative w-full sm:w-80">
      <input
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && q.trim()) {
            const first = results[0];
            if (first) go(first.exchange, first.symbol);
            else go("NSE", q.trim().toUpperCase());
          }
        }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Any NSE/BSE symbol, e.g. TATAMOTORS"
        className="w-full rounded-md border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent"
        aria-label="Search symbol"
      />
      {open && results.length > 0 && (
        <ul className="absolute z-30 mt-1 max-h-72 w-full overflow-auto rounded-md border border-line bg-panel text-sm shadow-lg">
          {results.map((r) => (
            <li key={`${r.exchange}:${r.symbol}`}>
              <button
                className="flex w-full items-baseline justify-between gap-2 px-3 py-1.5 text-left hover:bg-panel-2"
                onMouseDown={() => go(r.exchange, r.symbol)}
              >
                <span className="font-medium">{r.symbol}</span>
                <span className="truncate text-xs text-muted">
                  {r.name ?? ""} · {r.exchange}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

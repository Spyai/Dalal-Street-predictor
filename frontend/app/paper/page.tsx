"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { EquityPoint, Paper, PaperTrade } from "@/lib/types";
import { fmtDate, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice } from "@/components/Status";
import { fmtProb, fmtRupees } from "@/components/Drivers";

// The paper portfolio (brief §7.10, §6.3): every filled idea as a simulated trade with
// costs, plus manual paper trades. Every statistic carries its sample size; the equity
// curve is shown beside the Nifty 50 price index rebased to the paper capital (the
// total-return index is not available, and the page says so). Never a real order.

export default function PaperPage() {
  const [data, setData] = useState<Paper | null>(null);
  const [closed, setClosed] = useState<PaperTrade[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Paper>("/api/paper")
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
    api<PaperTrade[]>("/api/paper/trades?status=closed&limit=500")
      .then(setClosed)
      .catch(() => setClosed(null));
  }, []);

  if (!data) return <p className="text-sm text-muted">{error ?? "Loading…"}</p>;

  const last = data.equity_curve.at(-1) ?? null;
  const equity = last ? last.equity : null;
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold">Paper portfolio</h1>
        <p className="text-sm text-muted">
          Simulated book, {fmtRupees(data.capital)} starting capital, {data.risk_per_trade_pct}%
          risk per idea, at most {data.max_position_pct}% of capital per position, round-trip
          costs {data.cost_round_trip_pct.toFixed(3)}% of the entry value. Paper everywhere: no
          real orders.
        </p>
      </div>

      {error && <Notice tone="down">{error}</Notice>}

      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Paper equity"
          value={fmtRupees(equity)}
          sub={last ? `as of ${fmtDate(last.trade_date)}` : "no snapshot yet"}
          toneCls={tone(data.total_return_pct)}
        />
        <Stat
          label="Total return (paper)"
          value={fmtPct(data.total_return_pct)}
          sub={`Nifty 50 price index ${fmtPct(data.nifty_return_pct)} over the same ${data.n_days} snapshot${data.n_days === 1 ? "" : "s"}`}
          toneCls={tone(data.total_return_pct)}
        />
        <Stat
          label="Net P&L (closed)"
          value={fmtRupees(data.total_net_pnl)}
          sub={`n = ${data.n_trades} closed · ${data.n_open} open`}
          toneCls={tone(data.total_net_pnl)}
        />
        <Stat label="Max drawdown (paper)" value={fmtPct(data.max_drawdown_pct === null ? null : -data.max_drawdown_pct)} sub={`n = ${data.n_days} snapshots`} toneCls="text-down" />
      </section>

      {data.sample_warning && (
        <Notice tone="warn">
          Sample size: {data.n_trades} of {data.min_sample_size} closed paper trades. Treat every
          statistic here as unproven until there are {data.min_sample_size}.
        </Notice>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Trade statistics (closed paper trades)">
          <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <Row label="Hit rate" value={`${fmtProb(data.hit_rate)} (${data.n_wins} wins, ${data.n_losses} losses, n=${data.n_trades})`} />
            <Row label="Average win" value={`${fmtRupees(data.avg_win)} · ${fmtPct(data.avg_win_pct)}`} tone="text-up" />
            <Row label="Average loss" value={`${fmtRupees(data.avg_loss)} · ${fmtPct(data.avg_loss_pct)}`} tone="text-down" />
            <Row label="Expectancy per trade" value={`${fmtRupees(data.expectancy)} · ${fmtPct(data.expectancy_pct)}`} tone={tone(data.expectancy)} />
            <Row
              label="Profit factor"
              value={data.profit_factor === null ? (data.n_wins > 0 && data.n_losses === 0 ? "— (no losing trade yet)" : "—") : fmtNum(data.profit_factor, 2)}
            />
          </dl>
        </Section>
        <Section title="Curve statistics (daily paper equity)">
          <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <Row label="Sharpe (paper, annualised)" value={`${fmtNum(data.sharpe, 2)} (n=${data.n_returns} daily returns)`} />
            <Row label="Sortino (paper, annualised)" value={`${fmtNum(data.sortino, 2)} (n=${data.n_returns})`} />
            <Row label="Max drawdown" value={data.max_drawdown_pct === null ? "—" : `${data.max_drawdown_pct.toFixed(2)}%`} tone="text-down" />
            <Row label="Snapshots" value={String(data.n_days)} />
          </dl>
          <p className="mt-2 text-[11px] text-faint">{data.note}</p>
        </Section>
      </div>

      <Section title="Equity curve: paper vs Nifty 50 price index (rebased)">
        {data.equity_curve.length === 0 ? (
          <p className="text-sm text-muted">
            No equity snapshot yet. The 15:45 IST close-grading job writes one per session once
            the first idea is filled.
          </p>
        ) : (
          <EquityChart points={data.equity_curve} capital={data.capital} />
        )}
      </Section>

      <Section title={`Open positions (${data.open_trades.length})`}>
        {data.open_trades.length === 0 ? (
          <p className="text-sm text-muted">No open paper position.</p>
        ) : (
          <div className="overflow-x-auto">
            <TradeTable trades={data.open_trades} />
          </div>
        )}
      </Section>

      <Section title={`Closed trades (${closed ? closed.length : data.recent_closed.length})`}>
        {(closed ?? data.recent_closed).length === 0 ? (
          <p className="text-sm text-muted">No closed paper trade yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <TradeTable trades={closed ?? data.recent_closed} />
          </div>
        )}
      </Section>
    </div>
  );
}

function EquityChart({ points, capital }: { points: EquityPoint[]; capital: number }) {
  const W = 720;
  const H = 220;
  const padL = 56;
  const padR = 12;
  const padT = 10;
  const padB = 24;
  const n = points.length;
  const eq = points.map((p) => p.equity);
  const nifty = points.map((p) => p.nifty_rebased);
  const values = [...eq, ...nifty.filter((v): v is number => v !== null), capital];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (k: number) => padL + (n === 1 ? (W - padL - padR) / 2 : (k / (n - 1)) * (W - padL - padR));
  const y = (v: number) => padT + (1 - (v - lo) / span) * (H - padT - padB);
  const path = (vals: (number | null)[]) => {
    let d = "";
    let pen = false;
    vals.forEach((v, k) => {
      if (v === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)} `;
      pen = true;
    });
    return d.trim();
  };
  const ticks = [lo, (lo + hi) / 2, hi];
  const gaps = nifty.filter((v) => v === null).length;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-56 w-full" role="img" aria-label="Paper equity vs Nifty 50">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--muted)">
              {fmtRupees(t / 1000, 0)}k
            </text>
          </g>
        ))}
        <line x1={padL} x2={W - padR} y1={y(capital)} y2={y(capital)} stroke="var(--faint)" strokeDasharray="3 3" strokeWidth={1} />
        <path d={path(nifty)} fill="none" stroke="var(--muted)" strokeWidth={1.5} />
        <path d={path(eq)} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {n === 1 && <circle cx={x(0)} cy={y(eq[0] ?? capital)} r={3} fill="var(--accent)" />}
        <text x={padL} y={H - 6} fontSize={10} fill="var(--muted)">
          {fmtDate(points[0]?.trade_date)}
        </text>
        <text x={W - padR} y={H - 6} textAnchor="end" fontSize={10} fill="var(--muted)">
          {fmtDate(points[n - 1]?.trade_date)}
        </text>
      </svg>
      <div className="mt-1 flex flex-wrap gap-4 text-[11px] text-muted">
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-accent align-middle" /> Paper equity
        </span>
        <span>
          <span className="mr-1 inline-block h-0.5 w-4 bg-muted align-middle" /> Nifty 50 price
          index rebased to {fmtRupees(capital)} (not total return)
        </span>
        <span>
          <span className="mr-1 inline-block w-4 border-t border-dashed border-faint align-middle" />{" "}
          starting capital
        </span>
        {gaps > 0 && (
          <span className="text-warn">
            Nifty close missing on {gaps} snapshot{gaps === 1 ? "" : "s"} (gap left, not filled).
          </span>
        )}
      </div>
    </div>
  );
}

function TradeTable({ trades }: { trades: PaperTrade[] }) {
  return (
    <table className="num w-full text-xs">
      <thead className="text-left text-faint">
        <tr>
          <th className="py-1 font-normal">Entered</th>
          <th className="py-1 font-normal">Symbol</th>
          <th className="py-1 text-right font-normal">Qty</th>
          <th className="py-1 text-right font-normal">Entry</th>
          <th className="py-1 text-right font-normal">Stop / target</th>
          <th className="py-1 text-right font-normal">Exit</th>
          <th className="py-1 text-right font-normal">Gross</th>
          <th className="py-1 text-right font-normal">Costs</th>
          <th className="py-1 text-right font-normal">Net</th>
          <th className="py-1 text-right font-normal">R</th>
          <th className="py-1 font-normal">Idea</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((t) => (
          <tr key={t.id} className="border-t border-line">
            <td className="py-1.5 pr-2 whitespace-nowrap text-muted">{fmtDate(t.entry_date)}</td>
            <td className="py-1.5 pr-2 font-medium">
              <Link href={`/stock/${t.exchange}/${encodeURIComponent(t.symbol)}`} className="hover:underline">
                {t.symbol}
              </Link>
              {t.idea_id === null && <span className="ml-1 text-[10px] uppercase text-faint">manual</span>}
            </td>
            <td className="py-1.5 text-right">{t.qty}</td>
            <td className="py-1.5 text-right">{fmtNum(t.entry_price)}</td>
            <td className="py-1.5 text-right whitespace-nowrap">
              <span className="text-down">{fmtNum(t.stop)}</span> /{" "}
              <span className="text-up">{fmtNum(t.target)}</span>
            </td>
            <td className="py-1.5 text-right whitespace-nowrap">
              {t.exit_price === null ? (
                <span className="text-faint">open</span>
              ) : (
                `${fmtNum(t.exit_price)} · ${fmtDate(t.exit_date)}`
              )}
            </td>
            <td className={`py-1.5 text-right ${tone(t.gross_pnl)}`}>{fmtRupees(t.gross_pnl)}</td>
            <td className="py-1.5 text-right text-muted">{t.costs === null ? "—" : `−${fmtRupees(t.costs)}`}</td>
            <td className={`py-1.5 text-right whitespace-nowrap ${tone(t.net_pnl)}`}>
              {fmtRupees(t.net_pnl)}
              {t.net_return_pct !== null && <span className="ml-1 text-[10px]">{fmtPct(t.net_return_pct)}</span>}
            </td>
            <td className="py-1.5 text-right">{t.r_multiple === null ? "—" : fmtNum(t.r_multiple, 2)}</td>
            <td className="py-1.5 pl-2">
              {t.idea_id !== null ? (
                <Link
                  href={ideaSeqFromNote(t.note) !== null ? `/ideas?seq=${ideaSeqFromNote(t.note)}` : "/ideas"}
                  className="text-muted underline hover:text-fg"
                  title={t.note ?? undefined}
                >
                  {ideaSeqFromNote(t.note) !== null ? `#${ideaSeqFromNote(t.note)}` : "idea"}
                </Link>
              ) : (
                <span className="text-faint" title={t.note ?? undefined}>—</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Auto-opened trades carry "auto: idea seq N (...)" in their note; the seq links the idea.
function ideaSeqFromNote(note: string | null): number | null {
  const m = note?.match(/idea seq (\d+)/);
  return m && m[1] ? Number(m[1]) : null;
}

function Stat({ label, value, sub, toneCls = "" }: { label: string; value: string; sub?: string; toneCls?: string }) {
  return (
    <div className="card p-3">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`num mt-1 text-xl font-semibold ${toneCls}`}>{value}</div>
      {sub && <div className="num text-[11px] text-faint">{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card p-4">
      <h2 className="label mb-2">{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className={`text-right ${tone}`}>{value}</dd>
    </>
  );
}

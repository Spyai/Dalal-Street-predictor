"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Idea, IdeaDetail, IdeasList, LedgerVerify, PaperTrade } from "@/lib/types";
import { fmtDate, fmtIstDateTime, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice } from "@/components/Status";
import {
  Drivers,
  HORIZON_LABEL,
  IdeaStatusPill,
  IdeaTrackLine,
  SurveillanceBadge,
  fmtLevel,
  fmtProb,
  fmtRupees,
} from "@/components/Drivers";

// The Ideas page (brief §7.3): every current idea in the ledger with filters by outcome
// status, horizon and setup; clicking one opens its full reasoning (reasons, invalidation,
// levels, the probability with its live track record, every factor, the outcome and the
// paper trade). Nothing is shown that was not written to the ledger; a missing number is
// "—"; surveillance is marked as not checked on every idea.

const STATUS_LABELS: Record<string, string> = {
  active: "active (pending + open)",
  graded: "graded (target / stop / expired)",
  pending: "pending entry",
  open: "open",
  target: "target hit",
  stop: "stopped",
  expired: "expired",
  invalidated: "invalidated",
  void: "void",
};

export default function IdeasPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <IdeasView />
    </Suspense>
  );
}

function IdeasView() {
  const router = useRouter();
  const params = useSearchParams();
  const seqParam = params.get("seq");
  const selectedSeq = seqParam ? Number(seqParam) : null;
  const [status, setStatus] = useState(params.get("status") ?? "");
  const [horizon, setHorizon] = useState(params.get("horizon") ?? "");
  const [setup, setSetup] = useState(params.get("setup") ?? "");
  const [data, setData] = useState<IdeasList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<IdeaDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const load = useCallback(() => {
    const q = new URLSearchParams();
    if (status) q.set("status", status);
    if (horizon) q.set("horizon", horizon);
    if (setup) q.set("setup", setup);
    q.set("limit", "200");
    api<IdeasList>(`/api/ideas?${q.toString()}`)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
  }, [status, horizon, setup]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (selectedSeq === null || Number.isNaN(selectedSeq)) {
      setDetail(null);
      setDetailError(null);
      return;
    }
    setDetail(null);
    api<IdeaDetail>(`/api/ideas/${selectedSeq}`)
      .then((d) => {
        setDetail(d);
        setDetailError(null);
      })
      .catch((e: unknown) =>
        setDetailError(e instanceof ApiError ? e.message : "Failed to load the idea"),
      );
  }, [selectedSeq]);

  function select(seq: number | null) {
    const q = new URLSearchParams();
    if (status) q.set("status", status);
    if (horizon) q.set("horizon", horizon);
    if (setup) q.set("setup", setup);
    if (seq !== null) q.set("seq", String(seq));
    const qs = q.toString();
    router.replace(qs ? `/ideas?${qs}` : "/ideas");
  }

  if (!data) return <p className="text-sm text-muted">{error ?? "Loading…"}</p>;

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold">Ideas</h1>
          <p className="text-sm text-muted">
            Every stock idea in the ledger with its outcome. Long only; entry at the next open
            inside the zone, else invalidated; same-day target and stop count as the stop.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Select
            label="Status"
            value={status}
            onChange={setStatus}
            options={[["", "all"], ...data.statuses.map((s) => [s, STATUS_LABELS[s] ?? s] as [string, string])]}
          />
          <Select
            label="Horizon"
            value={horizon}
            onChange={setHorizon}
            options={[["", "both"], ...data.horizons.map((h) => [h, HORIZON_LABEL[h] ?? h] as [string, string])]}
          />
          <Select
            label="Setup"
            value={setup}
            onChange={setSetup}
            options={[["", "all"], ...data.setups.map((s) => [s, s.replace(/_/g, " ")] as [string, string])]}
          />
        </div>
      </div>

      {error && <Notice tone="down">{error}</Notice>}
      <Notice tone="warn">{data.surveillance_note}</Notice>

      <div className={`grid gap-4 ${selectedSeq !== null ? "lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" : ""}`}>
        <section className="card p-4">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="label">
              {data.n} idea{data.n === 1 ? "" : "s"}
              {data.n >= data.limit ? ` (showing the latest ${data.limit})` : ""}
            </h2>
          </div>
          {data.ideas.length === 0 ? (
            <p className="text-sm text-muted">
              No idea matches these filters.{" "}
              {!status && !horizon && !setup
                ? "The ledger has no ideas yet: the 07:30 job records them once the ideas models are trained, and only when a candidate clears the bar."
                : ""}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <IdeaTable ideas={data.ideas} selected={selectedSeq} onSelect={select} />
            </div>
          )}
        </section>

        {selectedSeq !== null && (
          <section className="card p-4">
            <div className="mb-2 flex items-start justify-between gap-2">
              <h2 className="label">Idea #{selectedSeq}</h2>
              <button onClick={() => select(null)} className="text-xs text-muted hover:text-fg">
                Close
              </button>
            </div>
            {detailError && <Notice tone="down">{detailError}</Notice>}
            {!detail && !detailError && <p className="text-sm text-muted">Loading…</p>}
            {detail && <IdeaDetailView idea={detail} />}
          </section>
        )}
      </div>

      <LedgerBox />
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <label className="flex items-center gap-1 text-muted">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-line bg-panel px-1.5 py-1 text-fg"
      >
        {options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

function IdeaTable({
  ideas,
  selected,
  onSelect,
}: {
  ideas: Idea[];
  selected: number | null;
  onSelect: (seq: number) => void;
}) {
  return (
    <table className="num w-full text-xs">
      <thead className="text-left text-faint">
        <tr>
          <th className="py-1 font-normal">Session</th>
          <th className="py-1 font-normal">Symbol</th>
          <th className="py-1 font-normal">Horizon · setup</th>
          <th className="py-1 text-right font-normal">Entry zone</th>
          <th className="py-1 text-right font-normal">Stop</th>
          <th className="py-1 text-right font-normal">Targets</th>
          <th className="py-1 text-right font-normal" title="P(target before stop) · live hit rate of this setup/horizon (n graded)">
            P(target) · record
          </th>
          <th className="py-1 text-right font-normal" title="Expected value after costs, % of entry">
            EV
          </th>
          <th className="py-1 font-normal">Outcome</th>
          <th className="py-1 text-right font-normal">Net</th>
        </tr>
      </thead>
      <tbody>
        {ideas.map((i) => {
          const o = i.outcome;
          const active = i.seq === selected;
          return (
            <tr
              key={i.id}
              onClick={() => onSelect(i.seq)}
              className={`cursor-pointer border-t border-line hover:bg-panel-2/60 ${active ? "bg-panel-2" : ""}`}
            >
              <td className="py-1.5 pr-2 whitespace-nowrap text-muted">
                {fmtDate(i.session_date)}
                {i.is_late && (
                  <span className="ml-1 text-[10px] uppercase text-warn" title="recorded after 09:15 IST">
                    late
                  </span>
                )}
                {i.corrects_idea_id !== null && (
                  <span className="ml-1 text-[10px] uppercase text-faint" title={`corrects idea #${i.corrects_idea_id}`}>
                    fix
                  </span>
                )}
              </td>
              <td className="py-1.5 pr-2 font-medium">
                {i.symbol}
                <span className="ml-1 text-[10px] text-faint">#{i.rank}</span>
              </td>
              <td className="py-1.5 pr-2 whitespace-nowrap text-muted">
                {HORIZON_LABEL[i.horizon] ?? i.horizon} · {i.setup_label}
              </td>
              <td className="py-1.5 text-right whitespace-nowrap">
                {fmtLevel(i.entry_low)}–{fmtLevel(i.entry_high)}
              </td>
              <td className="py-1.5 text-right text-down">{fmtLevel(i.stop)}</td>
              <td className="py-1.5 text-right whitespace-nowrap text-up">
                {fmtLevel(i.target_1)}
                {i.target_2 !== null ? ` / ${fmtLevel(i.target_2)}` : ""}
              </td>
              <td className="py-1.5 text-right whitespace-nowrap">
                {fmtProb(i.p_target)}
                <span className={`ml-1 ${i.track.sample_warning ? "text-warn" : "text-muted"}`}>
                  · {i.track.n_graded > 0 ? fmtProb(i.track.hit_rate) : "—"} (n={i.track.n_graded})
                </span>
              </td>
              <td className={`py-1.5 text-right ${tone(i.ev_pct_after_costs)}`}>
                {fmtPct(i.ev_pct_after_costs)}
              </td>
              <td className="py-1.5 pl-2">
                <IdeaStatusPill status={o.status} title={o.notes ?? undefined} />
              </td>
              <td className={`py-1.5 text-right ${tone(o.net_return_pct)}`}>{fmtPct(o.net_return_pct)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function levelsText(i: Idea): string {
  return `${fmtLevel(i.entry_low)}–${fmtLevel(i.entry_high)}`;
}

function IdeaDetailView({ idea }: { idea: IdeaDetail }) {
  const o = idea.outcome;
  return (
    <div className="space-y-3 text-sm">
      <div>
        <div className="flex flex-wrap items-baseline gap-x-2">
          <Link
            href={`/stock/${idea.exchange}/${encodeURIComponent(idea.symbol)}`}
            className="text-2xl font-semibold hover:underline"
          >
            {idea.symbol}
          </Link>
          <span className="text-muted">
            {HORIZON_LABEL[idea.horizon] ?? idea.horizon} · {idea.horizon_days} sessions ·{" "}
            {idea.strategy_label}
          </span>
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          <IdeaStatusPill status={o.status} />
          <SurveillanceBadge note={idea.surveillance_note} />
          {idea.is_late && (
            <span className="rounded border border-warn/40 px-1.5 py-px text-[10px] uppercase text-warn">
              late
            </span>
          )}
          {idea.corrected_by_seq !== null && (
            <span className="rounded border border-warn/40 px-1.5 py-px text-[10px] uppercase text-warn">
              superseded by #{idea.corrected_by_seq}
            </span>
          )}
        </div>
        <p className="mt-1 text-xs text-faint">
          Session {fmtDate(idea.session_date)} · rank {idea.rank} · recorded {fmtIstDateTime(idea.as_of)}{" "}
          IST{idea.model_version ? ` · ${idea.model_version}` : " · no model version recorded"}
          {idea.corrects_idea_id !== null ? ` · corrects idea #${idea.corrects_idea_id}` : ""}
        </p>
      </div>

      <dl className="num grid grid-cols-2 gap-x-4 gap-y-1">
        <Row label="Entry zone" value={levelsText(idea)} />
        <Row label="Stop" value={fmtLevel(idea.stop)} tone="text-down" />
        <Row label="Target 1" value={fmtLevel(idea.target_1)} tone="text-up" />
        <Row label="Target 2" value={fmtLevel(idea.target_2)} tone="text-up" />
        <Row
          label="Expected return"
          value={`${fmtPct(idea.expected_return_low_pct)} to ${fmtPct(idea.expected_return_high_pct)}`}
        />
        <Row label="P(target before stop)" value={fmtProb(idea.p_target)} />
        <Row label="Risk / reward" value={`${fmtNum(idea.risk_reward, 2)} : 1`} />
        <Row
          label="EV after costs"
          value={fmtPct(idea.ev_pct_after_costs)}
          tone={tone(idea.ev_pct_after_costs)}
        />
        <Row
          label="Position"
          value={`${idea.position_qty} shares · ${fmtRupees(idea.position_value)}`}
        />
        <Row label="Risk amount" value={fmtRupees(idea.risk_amount)} />
      </dl>
      <IdeaTrackLine track={idea.track} />

      <div>
        <div className="label mb-1">Why ({idea.reasons.length} reasons)</div>
        {idea.reasons.length === 0 ? (
          <p className="text-xs text-faint">No reasons recorded.</p>
        ) : (
          <ol className="list-decimal space-y-0.5 pl-5">
            {idea.reasons.map((r, k) => (
              <li key={k}>{r}</li>
            ))}
          </ol>
        )}
      </div>
      <div>
        <div className="label mb-1">What invalidates it</div>
        <p>{idea.invalidation || <span className="text-faint">—</span>}</p>
      </div>

      <Drivers
        drivers={idea.factors.drivers}
        target="target before stop"
        factors={idea.factors}
        topN={idea.factors.drivers_top_n}
        title="Model drivers"
      />

      <OutcomeBox idea={idea} />
      <PaperTradeBox trade={idea.paper_trade} />

      {idea.note && <p className="text-xs text-muted">Note: {idea.note}</p>}
      <p className="break-all text-[11px] text-faint">
        Ledger #{idea.seq} · hash {idea.hash.slice(0, 16)}… · prev {idea.prev_hash.slice(0, 16)}…
      </p>
    </div>
  );
}

function OutcomeBox({ idea }: { idea: IdeaDetail }) {
  const o = idea.outcome;
  return (
    <div className="rounded-md border border-line p-3">
      <div className="mb-1 flex items-center justify-between">
        <div className="label">Outcome</div>
        <IdeaStatusPill status={o.status} />
      </div>
      {o.status === "pending" ? (
        <p className="text-xs text-muted">
          Not entered yet. The 09:20 IST open check fills the idea at {fmtDate(idea.session_date)}
          &apos;s open if it lies inside the entry zone, otherwise marks it invalidated.
        </p>
      ) : (
        <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <Row label="Entry" value={`${fmtLevel(o.entry_price)} on ${fmtDate(o.entry_date)}`} />
          <Row
            label="Exit"
            value={o.exit_price === null ? "—" : `${fmtLevel(o.exit_price)} on ${fmtDate(o.exit_date)}`}
          />
          <Row label="Gross return" value={fmtPct(o.gross_return_pct)} tone={tone(o.gross_return_pct)} />
          <Row label="Costs" value={o.costs_pct === null ? "—" : `−${o.costs_pct.toFixed(3)}%`} />
          <Row label="Net return" value={fmtPct(o.net_return_pct)} tone={tone(o.net_return_pct)} />
          <Row label="R multiple" value={o.r_multiple === null ? "—" : `${fmtNum(o.r_multiple, 2)} R`} />
          <Row label="Sessions held" value={o.days_held === null ? "—" : String(o.days_held)} />
          <Row
            label="Best / worst excursion"
            value={`${fmtPct(o.max_favorable_pct)} / ${fmtPct(o.max_adverse_pct)}`}
          />
        </dl>
      )}
      {o.notes && <p className="mt-1 text-xs text-muted">{o.notes}</p>}
      {o.graded_at && (
        <p className="mt-1 text-[11px] text-faint">
          Graded {fmtIstDateTime(o.graded_at)} IST{o.source ? ` · ${o.source}` : ""}
          {o.error_category ? ` · ${o.error_category}` : ""}
        </p>
      )}
    </div>
  );
}

function PaperTradeBox({ trade }: { trade: PaperTrade | null }) {
  return (
    <div className="rounded-md border border-line p-3">
      <div className="mb-1 flex items-center justify-between">
        <div className="label">Paper trade</div>
        <Link href="/paper" className="text-xs text-muted underline hover:text-fg">
          Paper portfolio
        </Link>
      </div>
      {!trade ? (
        <p className="text-xs text-muted">
          No paper trade: one is opened when the idea is filled at the open (never a real order).
        </p>
      ) : (
        <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <Row label="Status" value={trade.status} />
          <Row label="Quantity" value={`${trade.qty} × ${fmtLevel(trade.entry_price)}`} />
          <Row label="Entry value" value={fmtRupees(trade.entry_value)} />
          <Row label="Entered" value={fmtDate(trade.entry_date)} />
          <Row
            label="Exit"
            value={trade.exit_price === null ? "—" : `${fmtLevel(trade.exit_price)} on ${fmtDate(trade.exit_date)}`}
          />
          <Row label="Gross P&L" value={fmtRupees(trade.gross_pnl)} tone={tone(trade.gross_pnl)} />
          <Row label="Costs" value={trade.costs === null ? "—" : `−${fmtRupees(trade.costs)}`} />
          <Row label="Net P&L" value={fmtRupees(trade.net_pnl)} tone={tone(trade.net_pnl)} />
          <Row label="Net return" value={fmtPct(trade.net_return_pct)} tone={tone(trade.net_return_pct)} />
          <Row label="R multiple" value={trade.r_multiple === null ? "—" : `${fmtNum(trade.r_multiple, 2)} R`} />
        </dl>
      )}
    </div>
  );
}

function LedgerBox() {
  const [verify, setVerify] = useState<LedgerVerify | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);

  async function run() {
    setVerifying(true);
    setVerifyError(null);
    try {
      setVerify(await api<LedgerVerify>("/api/scorecard/verify-ledger?chain=idea", { method: "POST" }));
    } catch (e) {
      setVerify(null);
      setVerifyError(e instanceof ApiError ? e.message : "Verification failed");
    } finally {
      setVerifying(false);
    }
  }

  return (
    <section className="card p-4">
      <h2 className="label mb-2">Idea ledger</h2>
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={run}
          disabled={verifying}
          className="rounded border border-line px-3 py-1 text-sm hover:bg-panel-2 disabled:opacity-40"
        >
          {verifying ? "Verifying…" : "Verify idea ledger"}
        </button>
        {verify && verify.ok && <span className="text-sm text-up">Intact: {verify.message}.</span>}
        {verify && !verify.ok && (
          <span className="text-sm text-down">
            Broken at seq {verify.first_bad_seq ?? "?"}: {verify.message} ({verify.checked} verified
            before it).
          </span>
        )}
        {verifyError && <span className="text-sm text-down">{verifyError}</span>}
      </div>
      <p className="mt-2 text-[11px] text-faint">
        Ideas have their own hash chain, separate from the market calls: every record carries a
        SHA-256 over its whole content and the previous idea&apos;s hash. Corrections are new
        records that point at the original; nothing is edited or deleted.
      </p>
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

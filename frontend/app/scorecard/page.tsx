"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  CalibrationBin,
  CallClass,
  Champion,
  IdeaChampion,
  IdeaGroupStats,
  IdeaLiftRow,
  IdeaRow,
  IdeasScorecard,
  LedgerVerify,
  PredictionRow,
  Scorecard,
} from "@/lib/types";
import { fmtDate, fmtIstDateTime, fmtNum, fmtPct, tone } from "@/lib/format";
import { Notice } from "@/components/Status";
import { COMPARISON_TONE, comparisonText, fmtProb } from "@/components/MarketCall";
import { HORIZON_LABEL, IdeaStatusPill } from "@/components/Drivers";

// The scorecard (brief §6.3): the market call's track record (every metric with its sample
// size, the naive baselines beside the model, calibration, the champion model), the stock
// ideas' track record (hit rate, average win / loss, expectancy, profit factor, by setup
// and horizon, calibration of p_target, the ideas champions) and the ledger checks.

const CLASS_TONE: Record<string, string> = { UP: "text-up", FLAT: "text-muted", DOWN: "text-down" };

export default function ScorecardPage() {
  const [target, setTarget] = useState<string | null>(null);
  const [data, setData] = useState<Scorecard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [verify, setVerify] = useState<LedgerVerify | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);

  const load = useCallback((t: string | null) => {
    const path = t ? `/api/scorecard?target=${encodeURIComponent(t)}` : "/api/scorecard";
    api<Scorecard>(path)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
  }, []);

  useEffect(() => {
    load(target);
  }, [load, target]);

  const [chain, setChain] = useState<"prediction" | "idea">("prediction");

  async function verifyLedger(which: "prediction" | "idea") {
    setVerifying(true);
    setVerifyError(null);
    setChain(which);
    try {
      setVerify(
        await api<LedgerVerify>(`/api/scorecard/verify-ledger?chain=${which}`, { method: "POST" }),
      );
    } catch (e) {
      setVerify(null);
      setVerifyError(e instanceof ApiError ? e.message : "Verification failed");
    } finally {
      setVerifying(false);
    }
  }

  if (!data) return <p className="text-sm text-muted">{error ?? "Loading…"}</p>;

  const modelRow = { label: "Model", n: data.n_graded, accuracy: data.accuracy };

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold">Scorecard</h1>
          <p className="text-sm text-muted">
            Market call track record for {data.target}, from the ledger. Every metric shows its
            sample size.
          </p>
        </div>
        <div className="flex overflow-hidden rounded border border-line text-xs">
          {data.targets.map((t) => (
            <button
              key={t}
              onClick={() => setTarget(t)}
              className={`px-2.5 py-1 ${t === data.target ? "bg-panel-2 text-fg" : "text-muted hover:text-fg"}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {error && <Notice tone="down">{error}</Notice>}

      <section className="card p-4">
        <h2 className="label mb-1">Verdict</h2>
        <p className="text-sm">{data.verdict}</p>
        {data.n_graded > 0 && (
          <p className={`mt-1 text-sm ${COMPARISON_TONE[data.comparison]}`}>
            {comparisonText(data.comparison, data.best_baseline?.n ?? null, data.min_paired_n)}
          </p>
        )}
        {data.sample_warning && (
          <div className="mt-2">
            <Notice tone="warn">
              Sample size: {data.n_graded} of {data.min_sample_size} graded calls. Treat the model
              as unproven until it has {data.min_sample_size}.
            </Notice>
          </div>
        )}
      </section>

      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Directional accuracy" value={fmtProb(data.accuracy)} n={data.n_graded} />
        <Stat label="Brier score" value={fmtNum(data.brier, 3)} n={data.n_graded} hint="0 = perfect, 0.667 = uniform" />
        <Stat label="Log loss" value={fmtNum(data.log_loss, 3)} n={data.n_graded} hint="1.099 = uniform" />
        <Stat label="Gap-call accuracy" value={fmtProb(data.gap_accuracy)} n={data.gap_n} />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Accuracy vs naive baselines (paired)">
          <table className="num w-full text-sm">
            <thead className="text-left text-xs text-faint">
              <tr>
                <th className="py-1 font-normal">Strategy</th>
                <th className="py-1 text-right font-normal">n</th>
                <th className="py-1 text-right font-normal">Baseline</th>
                <th className="py-1 text-right font-normal" title="The model on the same n sessions">
                  Model
                </th>
                <th className="py-1 text-right font-normal">Model − baseline</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-line font-medium">
                <td className="py-1.5">{modelRow.label} (all graded)</td>
                <td className="py-1.5 text-right">{modelRow.n}</td>
                <td className="py-1.5 text-right text-faint">—</td>
                <td className="py-1.5 text-right">{fmtProb(modelRow.accuracy)}</td>
                <td className="py-1.5 text-right text-faint">—</td>
              </tr>
              {data.baselines.map((b) => {
                const delta =
                  b.model_accuracy !== null && b.accuracy !== null
                    ? (b.model_accuracy - b.accuracy) * 100
                    : null;
                const toughest = data.best_baseline?.name === b.name;
                return (
                  <tr key={b.name} className={`border-t border-line ${toughest ? "bg-panel-2/60" : ""}`}>
                    <td className="py-1.5 text-muted">
                      {b.label}
                      {toughest && (
                        <span className="ml-1 text-[10px] uppercase text-faint" title="The baseline the model leads by the least">
                          toughest
                        </span>
                      )}
                    </td>
                    <td className="py-1.5 text-right">{b.n}</td>
                    <td className="py-1.5 text-right">{fmtProb(b.accuracy)}</td>
                    <td className="py-1.5 text-right">{fmtProb(b.model_accuracy)}</td>
                    <td
                      className={`py-1.5 text-right ${
                        delta === null ? "text-faint" : delta > 0 ? "text-up" : delta < 0 ? "text-down" : "text-muted"
                      }`}
                    >
                      {delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(0)} pp`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-faint">
            Paired: each baseline is compared with the model on exactly the n graded sessions where
            that baseline had a call (the overnight-gap rule needs US futures; random needs class
            frequencies), so both columns cover the same sessions. The verdict names the toughest
            baseline and only claims ahead/behind with at least {data.min_paired_n} paired calls.
          </p>
        </Section>

        <Section title="Accuracy by confidence">
          {data.by_confidence.length === 0 ? (
            <Empty>No graded calls yet.</Empty>
          ) : (
            <table className="num w-full text-sm">
              <thead className="text-left text-xs text-faint">
                <tr>
                  <th className="py-1 font-normal">Confidence</th>
                  <th className="py-1 text-right font-normal">n</th>
                  <th className="py-1 text-right font-normal">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {data.by_confidence.map((r) => (
                  <tr key={r.confidence} className="border-t border-line">
                    <td className="py-1.5 capitalize text-muted">
                      {r.confidence === "none" ? "none (no edge)" : r.confidence}
                    </td>
                    <td className="py-1.5 text-right">{r.n}</td>
                    <td className="py-1.5 text-right">{fmtProb(r.accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>

        <Section title="Calibration (top-class probability vs hit rate)">
          {data.calibration.length === 0 ? (
            <Empty>No graded calls yet.</Empty>
          ) : (
            <div className="space-y-2">
              {data.calibration.map((b) => (
                <div key={`${b.bin_lo}-${b.bin_hi}`} className="text-xs">
                  <div className="num flex justify-between text-muted">
                    <span>
                      said {fmtProb(b.bin_lo)}–{fmtProb(b.bin_hi)}
                    </span>
                    <span>n = {b.n}</span>
                  </div>
                  <CalBar label="predicted" value={b.predicted_mean} cls="bg-accent" />
                  <CalBar label="observed" value={b.observed} cls="bg-up" />
                </div>
              ))}
              <p className="text-[11px] text-faint">
                Well calibrated means the two bars match in every row: when it says 70%, it is right
                about 70% of the time.
              </p>
            </div>
          )}
        </Section>

        <Section title="Champion model">
          {data.champion ? (
            <ChampionBox champion={data.champion} />
          ) : (
            <Empty>
              No champion model for {data.target}. Train one with <code>python -m app.cli train</code>;
              until then the 07:30 job records no call.
            </Empty>
          )}
          <p className="mt-2 text-[11px] text-faint">
            OOS = strict walk-forward, out-of-sample, with the naive baselines scored on the same
            sessions. A challenger replaces the champion only when it beats it on these numbers.
          </p>
        </Section>
      </div>

      <IdeasScorecardSection />

      <Section title="Ledger">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => verifyLedger("prediction")}
            disabled={verifying}
            className="rounded border border-line px-3 py-1 text-sm hover:bg-panel-2 disabled:opacity-40"
          >
            {verifying && chain === "prediction" ? "Verifying…" : "Verify market-call ledger"}
          </button>
          <button
            onClick={() => verifyLedger("idea")}
            disabled={verifying}
            className="rounded border border-line px-3 py-1 text-sm hover:bg-panel-2 disabled:opacity-40"
          >
            {verifying && chain === "idea" ? "Verifying…" : "Verify idea ledger"}
          </button>
          {verify && verify.ok && (
            <span className="text-sm text-up">
              {chain === "idea" ? "Idea ledger" : "Market-call ledger"} intact: {verify.message}.
            </span>
          )}
          {verify && !verify.ok && (
            <span className="text-sm text-down">
              Broken at seq {verify.first_bad_seq ?? "?"}: {verify.message} ({verify.checked} verified
              before it).
            </span>
          )}
          {verifyError && <span className="text-sm text-down">{verifyError}</span>}
        </div>
        <p className="mt-2 text-[11px] text-faint">
          Two chains, one per ledger: every record carries a SHA-256 over its content and the
          previous record&apos;s hash. Verification recomputes the whole chain from the database;
          any edit, gap or removal shows up as the first bad sequence number.
        </p>
      </Section>

      <Section title={`Recent predictions (${data.target})`}>
        {data.recent.length === 0 ? (
          <Empty>No predictions recorded for {data.target} yet.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <RecentTable rows={data.recent} />
          </div>
        )}
      </Section>
    </div>
  );
}

// ---------------------------------------------------------------------------- stock ideas

const IDEA_STATUSES = ["pending", "open", "target", "stop", "expired", "invalidated", "void"] as const;

function IdeasScorecardSection() {
  const [horizon, setHorizon] = useState<string>("");
  const [data, setData] = useState<IdeasScorecard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const path = horizon ? `/api/scorecard/ideas?horizon=${encodeURIComponent(horizon)}` : "/api/scorecard/ideas";
    api<IdeasScorecard>(path)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
  }, [horizon]);

  return (
    <div id="ideas" className="space-y-4 border-t border-line pt-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Stock ideas</h2>
          <p className="text-sm text-muted">
            Track record of the ideas in the ledger after costs. Hit rate = the target came before
            the stop (what P(target) predicts); win rate = positive net return. Every metric shows
            its sample size.{" "}
            <Link href="/ideas" className="underline hover:text-fg">
              All ideas
            </Link>{" "}
            ·{" "}
            <Link href="/paper" className="underline hover:text-fg">
              Paper portfolio
            </Link>
          </p>
        </div>
        <div className="flex overflow-hidden rounded border border-line text-xs">
          {[["", "Both horizons"], ...(data?.horizons ?? ["swing", "positional"]).map((h) => [h, HORIZON_LABEL[h] ?? h])].map(
            ([value, label]) => (
              <button
                key={value}
                onClick={() => setHorizon(value ?? "")}
                className={`px-2.5 py-1 ${value === horizon ? "bg-panel-2 text-fg" : "text-muted hover:text-fg"}`}
              >
                {label}
              </button>
            ),
          )}
        </div>
      </div>

      {error && <Notice tone="down">{error}</Notice>}
      {!data && !error && <p className="text-sm text-muted">Loading…</p>}
      {data && <IdeasScorecardBody data={data} />}
    </div>
  );
}

function IdeasScorecardBody({ data }: { data: IdeasScorecard }) {
  const n = data.n_graded;
  return (
    <>
      <section className="card p-4">
        <h3 className="label mb-1">Verdict</h3>
        <p className="text-sm">{data.verdict}</p>
        {data.sample_warning && (
          <div className="mt-2">
            <Notice tone="warn">
              Sample size: {n} of {data.min_sample_size} graded ideas. Treat the ideas engine as
              unproven until it has {data.min_sample_size}.
            </Notice>
          </div>
        )}
        <p className="num mt-2 text-xs text-muted">
          {data.n_ideas} idea{data.n_ideas === 1 ? "" : "s"} in the ledger:{" "}
          {IDEA_STATUSES.map((st, k) => (
            <span key={st}>
              {k > 0 && " · "}
              {st} {data.by_status[st] ?? 0}
            </span>
          ))}
        </p>
      </section>

      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Hit rate (target before stop)" value={fmtProb(data.hit_rate)} n={n} />
        <Stat label="Win rate (net > 0)" value={fmtProb(data.win_rate)} n={n} />
        <Stat label="Average win / loss" value={`${fmtPct(data.avg_win_pct)} / ${fmtPct(data.avg_loss_pct)}`} n={n} hint="Mean net return of the winners / of the losers, % of entry" />
        <Stat label="Expectancy per idea" value={fmtPct(data.expectancy_pct)} n={n} hint="Mean net return after costs" />
        <Stat
          label="Profit factor"
          value={data.profit_factor === null ? (n > 0 ? "— (no losing idea yet)" : "—") : fmtNum(data.profit_factor, 2)}
          n={n}
          hint="Sum of winning returns over the sum of losing returns"
        />
        <Stat label="Average R multiple" value={data.avg_r_multiple === null ? "—" : `${fmtNum(data.avg_r_multiple, 2)} R`} n={n} />
        <Stat label="Average sessions held" value={fmtNum(data.avg_days_held, 1)} n={n} />
        <Stat label="Graded ideas" value={String(n)} n={data.n_ideas} hint="n below = ideas in the ledger (any status)" />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="By setup">
          {data.by_setup.length === 0 ? (
            <Empty>No graded ideas yet.</Empty>
          ) : (
            <GroupTable rows={data.by_setup} keyName="setup" />
          )}
        </Section>
        <Section title="By horizon">
          <GroupTable rows={data.by_horizon} keyName="horizon" />
        </Section>
        <Section title="Calibration (P(target) vs observed hit rate)">
          {data.calibration.length === 0 ? (
            <Empty>No graded ideas yet.</Empty>
          ) : (
            <Calibration bins={data.calibration} />
          )}
        </Section>
        <Section title="Champion ideas models">
          <div className="space-y-4">
            {Object.entries(data.champions).map(([h, c]) => (
              <div key={h}>
                <div className="mb-1 text-sm font-medium">{HORIZON_LABEL[h] ?? h}</div>
                {c ? (
                  <IdeaChampionBox champion={c} />
                ) : (
                  <Empty>
                    No champion for {h}. Train one with <code>python -m app.cli train-ideas</code>;
                    until then no {h} idea is generated.
                  </Empty>
                )}
              </div>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-faint">
            OOS = strict walk-forward over signal sessions with an embargo of the horizon&apos;s
            length, scored against every candidate&apos;s hit rate and random universe stocks with
            the same barriers.
          </p>
        </Section>
      </div>

      <Section title="Recent ideas">
        {data.recent.length === 0 ? (
          <Empty>No ideas recorded yet.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <RecentIdeasTable rows={data.recent} />
          </div>
        )}
      </Section>
    </>
  );
}

function GroupTable({ rows, keyName }: { rows: IdeaGroupStats[]; keyName: "setup" | "horizon" }) {
  return (
    <table className="num w-full text-xs">
      <thead className="text-left text-faint">
        <tr>
          <th className="py-1 font-normal">{keyName === "setup" ? "Setup" : "Horizon"}</th>
          <th className="py-1 text-right font-normal">n</th>
          <th className="py-1 text-right font-normal">Hit</th>
          <th className="py-1 text-right font-normal">Win</th>
          <th className="py-1 text-right font-normal">Avg win</th>
          <th className="py-1 text-right font-normal">Avg loss</th>
          <th className="py-1 text-right font-normal">Expect.</th>
          <th className="py-1 text-right font-normal">PF</th>
          <th className="py-1 text-right font-normal">Avg R</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const key = (keyName === "setup" ? r.setup : r.horizon) ?? "?";
          return (
            <tr key={key} className="border-t border-line">
              <td className="py-1 text-muted">{keyName === "horizon" ? HORIZON_LABEL[key] ?? key : key.replace(/_/g, " ")}</td>
              <td className="py-1 text-right">{r.n}</td>
              <td className="py-1 text-right">{fmtProb(r.hit_rate)}</td>
              <td className="py-1 text-right">{fmtProb(r.win_rate)}</td>
              <td className="py-1 text-right text-up">{fmtPct(r.avg_win_pct)}</td>
              <td className="py-1 text-right text-down">{fmtPct(r.avg_loss_pct)}</td>
              <td className={`py-1 text-right ${tone(r.expectancy_pct)}`}>{fmtPct(r.expectancy_pct)}</td>
              <td className="py-1 text-right">{fmtNum(r.profit_factor, 2)}</td>
              <td className="py-1 text-right">{fmtNum(r.avg_r_multiple, 2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Calibration({ bins }: { bins: CalibrationBin[] }) {
  return (
    <div className="space-y-2">
      {bins.map((b) => (
        <div key={`${b.bin_lo}-${b.bin_hi}`} className="text-xs">
          <div className="num flex justify-between text-muted">
            <span>
              said {fmtProb(b.bin_lo)}–{fmtProb(b.bin_hi)}
            </span>
            <span>n = {b.n}</span>
          </div>
          <CalBar label="predicted" value={b.predicted_mean} cls="bg-accent" />
          <CalBar label="observed" value={b.observed} cls="bg-up" />
        </div>
      ))}
      <p className="text-[11px] text-faint">
        Well calibrated means the two bars match in every row: when it says 60%, the target comes
        before the stop about 60% of the time.
      </p>
    </div>
  );
}

function IdeaChampionBox({ champion }: { champion: IdeaChampion }) {
  const oos = champion.oos;
  const all = champion.baselines.find((b) => b.name === "all_candidates");
  const worse = oos.hit_rate !== null && all?.hit_rate !== null && all !== undefined && oos.hit_rate <= all.hit_rate;
  return (
    <div className="space-y-2">
      <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <Row label="Version" value={champion.version} />
        <Row label="Trained through" value={fmtDate(champion.trained_through)} />
        <Row label="Training candidates" value={String(champion.train_rows)} />
        <Row label="OOS candidates" value={oos.n_oos === null ? "—" : String(oos.n_oos)} />
        <Row
          label="OOS hit rate (cleared the bar)"
          value={`${fmtProb(oos.hit_rate)}${oos.n_selected !== null ? ` (n=${oos.n_selected})` : ""}`}
          tone={worse ? "text-down" : ""}
        />
        <Row
          label="OOS hit rate (ranked, capped)"
          value={`${fmtProb(oos.hit_rate_ranked)}${oos.n_ranked !== null ? ` (n=${oos.n_ranked})` : ""}`}
        />
        <Row label="OOS Brier" value={fmtNum(oos.brier, 3)} />
        <Row label="OOS log loss" value={fmtNum(oos.log_loss, 3)} />
        <Row label="OOS expectancy after costs" value={fmtPct(oos.expectancy_after_costs_pct)} tone={tone(oos.expectancy_after_costs_pct)} />
        <Row label="All candidates' expectancy" value={fmtPct(oos.expectancy_all_candidates_pct)} tone={tone(oos.expectancy_all_candidates_pct)} />
        <Row
          label="Best ideas every session, bar ignored: hit rate"
          value={`${fmtProb(oos.hit_rate_top)}${oos.n_top !== null && oos.n_top !== undefined ? ` (n=${oos.n_top})` : ""}`}
        />
        <Row label="Best ideas every session, bar ignored: expectancy" value={fmtPct(oos.expectancy_top_pct)} tone={tone(oos.expectancy_top_pct)} />
        <Row label="Promoted" value={champion.promoted_at ? `${fmtIstDateTime(champion.promoted_at)} IST` : "—"} />
      </dl>
      <p className={`text-sm ${worse ? "text-down" : "text-muted"}`}>{champion.oos_verdict}</p>
      <table className="num w-full text-xs">
        <thead className="text-left text-faint">
          <tr>
            <th className="py-1 font-normal">Out of sample</th>
            <th className="py-1 text-right font-normal">n</th>
            <th className="py-1 text-right font-normal">Hit rate</th>
            <th className="py-1 text-right font-normal">Brier</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-line font-medium">
            <td className="py-1">Model (ideas that cleared the bar)</td>
            <td className="py-1 text-right">{oos.n_selected ?? "—"}</td>
            <td className="py-1 text-right">{fmtProb(oos.hit_rate)}</td>
            <td className="py-1 text-right">{fmtNum(oos.brier, 3)}</td>
          </tr>
          {champion.baselines.map((b) => (
            <tr key={b.name} className="border-t border-line">
              <td className="py-1 text-muted">{b.label}</td>
              <td className="py-1 text-right">{b.n}</td>
              <td className="py-1 text-right">{fmtProb(b.hit_rate)}</td>
              <td className="py-1 text-right">{fmtNum(b.brier, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {champion.lift && champion.lift.length > 0 && <LiftTable rows={champion.lift} />}
      {Object.keys(champion.per_strategy).length > 0 && (
        <p className="num text-xs text-muted">
          Per strategy (OOS hit rate):{" "}
          {Object.entries(champion.per_strategy).map(([name, m], k) => (
            <span key={name}>
              {k > 0 && " · "}
              {name.replace(/_/g, " ")} {fmtProb(m.hit_rate ?? null)} (n={m.n ?? "—"})
            </span>
          ))}
        </p>
      )}
      {champion.notes.length > 0 && (
        <ul className="list-disc pl-5 text-[11px] text-warn">
          {champion.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function LiftTable({ rows }: { rows: IdeaLiftRow[] }) {
  const first = rows[0];
  const last = rows[rows.length - 1];
  const flat =
    first !== undefined &&
    last !== undefined &&
    first.hit_rate !== null &&
    last.hit_rate !== null &&
    last.hit_rate - first.hit_rate < 0.03;
  return (
    <div className="space-y-1">
      <table className="num w-full text-xs">
        <thead className="text-left text-faint">
          <tr>
            <th className="py-1 font-normal">P(target) decile within the session (OOS)</th>
            <th className="py-1 text-right font-normal">n</th>
            <th className="py-1 text-right font-normal">Said</th>
            <th className="py-1 text-right font-normal">Hit rate</th>
            <th className="py-1 text-right font-normal">Expectancy</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.decile} className="border-t border-line">
              <td className="py-1 text-muted">
                {r.decile === 1 ? "1 (lowest)" : r.decile === rows.length ? `${r.decile} (highest)` : String(r.decile)}
              </td>
              <td className="py-1 text-right">{r.n}</td>
              <td className="py-1 text-right">{fmtProb(r.p_mean)}</td>
              <td className="py-1 text-right">{fmtProb(r.hit_rate)}</td>
              <td className={`py-1 text-right ${tone(r.expectancy_pct)}`}>{fmtPct(r.expectancy_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className={`text-[11px] ${flat ? "text-down" : "text-faint"}`}>
        {flat
          ? "Flat: on a given day the candidates the model likes most hit about as often as the ones it likes least. The ranking carries little beyond the strategies' base rates; the ideas shown are those base rates plus payoff geometry."
          : "A ranking that carries information climbs from decile 1 to 10: on a given day the candidates the model likes most hit more often."}
      </p>
    </div>
  );
}

function RecentIdeasTable({ rows }: { rows: IdeaRow[] }) {
  return (
    <table className="num w-full text-xs">
      <thead className="text-left text-faint">
        <tr>
          <th className="py-1 font-normal">Session</th>
          <th className="py-1 font-normal">Symbol</th>
          <th className="py-1 font-normal">Horizon · setup</th>
          <th className="py-1 text-right font-normal">P(target)</th>
          <th className="py-1 text-right font-normal">EV</th>
          <th className="py-1 font-normal">Outcome</th>
          <th className="py-1 text-right font-normal">Net</th>
          <th className="py-1 text-right font-normal">R</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-line">
            <td className="py-1.5 pr-2 whitespace-nowrap text-muted">
              {fmtDate(r.session_date)}
              {r.is_late && <span className="ml-1 text-[10px] uppercase text-warn">late</span>}
              {r.corrects_idea_id !== null && <span className="ml-1 text-[10px] uppercase text-faint">fix</span>}
            </td>
            <td className="py-1.5 pr-2 font-medium">
              <Link href={`/ideas?seq=${r.seq}`} className="hover:underline">
                {r.symbol}
              </Link>
            </td>
            <td className="py-1.5 pr-2 whitespace-nowrap text-muted">
              {HORIZON_LABEL[r.horizon] ?? r.horizon} · {r.setup.replace(/_/g, " ")}
            </td>
            <td className="py-1.5 text-right">{fmtProb(r.p_target)}</td>
            <td className={`py-1.5 text-right ${tone(r.ev_pct_after_costs)}`}>{fmtPct(r.ev_pct_after_costs)}</td>
            <td className="py-1.5 pl-2">
              <IdeaStatusPill status={r.status} title={r.outcome_notes ?? undefined} />
            </td>
            <td className={`py-1.5 text-right ${tone(r.net_return_pct)}`}>{fmtPct(r.net_return_pct)}</td>
            <td className="py-1.5 text-right">{r.r_multiple === null ? "—" : fmtNum(r.r_multiple, 2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const CLASS_ORDER: CallClass[] = ["UP", "FLAT", "DOWN"];

function ChampionBox({ champion }: { champion: Champion }) {
  const oos = champion.oos;
  const worse =
    oos.brier !== null && champion.climatology_brier !== null && oos.brier > champion.climatology_brier;
  const freq = champion.class_frequencies;
  return (
    <div className="space-y-3">
      <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <Row label="Version" value={champion.version} />
        <Row label="Kind" value={champion.kind} />
        <Row label="Trained through" value={fmtDate(champion.trained_through)} />
        <Row label="Training rows" value={String(champion.train_rows)} />
        <Row label="OOS sessions" value={oos.n_oos === null ? "—" : String(oos.n_oos)} />
        <Row label="OOS accuracy" value={fmtProb(oos.accuracy)} />
        <Row label="OOS Brier" value={fmtNum(oos.brier, 3)} tone={worse ? "text-down" : ""} />
        <Row label="Climatology Brier" value={fmtNum(champion.climatology_brier, 3)} />
        <Row label="OOS log loss" value={fmtNum(oos.log_loss, 3)} />
        {champion.range_coverage !== null && (
          <Row
            label="10–90% range coverage"
            value={`${fmtProb(champion.range_coverage)} (nominal 80%)`}
            tone={Math.abs(champion.range_coverage - 0.8) > 0.1 ? "text-warn" : ""}
          />
        )}
        <Row
          label="Promoted"
          value={champion.promoted_at ? `${fmtIstDateTime(champion.promoted_at)} IST` : "—"}
        />
      </dl>

      <p className={`text-sm ${worse ? "text-down" : "text-muted"}`}>{champion.oos_verdict}</p>

      <div>
        <div className="label mb-1">Out of sample: model vs baselines</div>
        {champion.baselines.length === 0 ? (
          <Empty>This model was stored without walk-forward baselines, so none are shown.</Empty>
        ) : (
          <table className="num w-full text-xs">
            <thead className="text-left text-faint">
              <tr>
                <th className="py-1 font-normal">Strategy</th>
                <th className="py-1 text-right font-normal">n</th>
                <th className="py-1 text-right font-normal">Accuracy</th>
                <th className="py-1 text-right font-normal">Brier</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-line font-medium">
                <td className="py-1">Model</td>
                <td className="py-1 text-right">{oos.n_oos ?? "—"}</td>
                <td className="py-1 text-right">{fmtProb(oos.accuracy)}</td>
                <td className={`py-1 text-right ${worse ? "text-down" : ""}`}>{fmtNum(oos.brier, 3)}</td>
              </tr>
              {champion.baselines.map((b) => (
                <tr key={b.name} className="border-t border-line">
                  <td className="py-1 text-muted">
                    {b.label}
                    {b.name === "random" && (
                      <span className="ml-1 text-[10px] text-faint" title="Its Brier is that of the class-frequency (climatology) forecast">
                        · Brier = climatology
                      </span>
                    )}
                  </td>
                  <td className="py-1 text-right">{b.n}</td>
                  <td className="py-1 text-right">{fmtProb(b.accuracy)}</td>
                  <td className="py-1 text-right">{fmtNum(b.brier, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-1 text-[11px] text-faint">
          A baseline&apos;s n below the model&apos;s means it had no call on some sessions; its
          accuracy then covers a different set and is not directly comparable.
        </p>
      </div>

      {freq && (
        <p className="num text-xs text-muted">
          Training class frequencies:{" "}
          {CLASS_ORDER.map((c, i) => (
            <span key={c}>
              {i > 0 && " · "}
              <span className={CLASS_TONE[c]}>{c}</span> {fmtProb(freq[c] ?? null)}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

function RecentTable({ rows }: { rows: PredictionRow[] }) {
  return (
    <table className="num w-full text-xs">
      <thead className="text-left text-faint">
        <tr>
          <th className="py-1 font-normal">Session</th>
          <th className="py-1 font-normal">Call</th>
          <th className="py-1 text-right font-normal">UP</th>
          <th className="py-1 text-right font-normal">FLAT</th>
          <th className="py-1 text-right font-normal">DOWN</th>
          <th className="py-1 font-normal">Confidence</th>
          <th className="py-1 text-right font-normal">Actual</th>
          <th className="py-1 font-normal">Result</th>
          <th className="py-1 text-right font-normal">Brier</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-line">
            <td className="py-1.5 pr-2 whitespace-nowrap text-muted">
              {fmtDate(r.session_date)}
              {r.is_late && (
                <span className="ml-1 text-[10px] uppercase text-warn" title="recorded after 09:15 IST">
                  late
                </span>
              )}
              {r.corrects_prediction_id !== null && (
                <span className="ml-1 text-[10px] uppercase text-faint" title={`corrects #${r.corrects_prediction_id}`}>
                  fix
                </span>
              )}
            </td>
            <td className="py-1.5 pr-2 whitespace-nowrap">
              {r.no_edge ? (
                <span className="text-warn">No edge</span>
              ) : (
                <span className={CLASS_TONE[r.predicted_class] ?? ""}>{r.predicted_class}</span>
              )}
            </td>
            <td className="py-1.5 text-right">{fmtProb(r.probs.UP)}</td>
            <td className="py-1.5 text-right">{fmtProb(r.probs.FLAT)}</td>
            <td className="py-1.5 text-right">{fmtProb(r.probs.DOWN)}</td>
            <td className="py-1.5 pl-2 text-muted">{r.confidence}</td>
            <td className="py-1.5 text-right whitespace-nowrap">
              {r.actual_class === null ? (
                <span className="text-faint">—</span>
              ) : (
                <span className={CLASS_TONE[r.actual_class] ?? ""}>
                  {fmtPct(r.actual_pct)} {r.actual_class}
                </span>
              )}
            </td>
            <td className="py-1.5 pl-2">
              {r.correct === null ? (
                <span className="text-faint">pending</span>
              ) : r.correct ? (
                <span className="text-up">correct</span>
              ) : (
                <span className="text-down">wrong</span>
              )}
            </td>
            <td className="py-1.5 text-right">{fmtNum(r.brier, 3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CalBar({ label, value, cls }: { label: string; value: number | null; cls: string }) {
  const w = value === null ? 0 : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div className="mt-0.5 flex items-center gap-2">
      <span className="w-14 text-faint">{label}</span>
      <div className="h-1.5 flex-1 rounded bg-panel-2">
        <div className={`h-1.5 rounded ${cls}`} style={{ width: `${w}%` }} />
      </div>
      <span className="num w-10 text-right">{fmtProb(value)}</span>
    </div>
  );
}

function Stat({ label, value, n, hint }: { label: string; value: string; n: number; hint?: string }) {
  return (
    <div className="card p-3" title={hint}>
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num mt-1 text-xl font-semibold">{value}</div>
      <div className="num text-[11px] text-faint">n = {n}</div>
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
      <dd className={`text-right break-all ${tone}`}>{value}</dd>
    </>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}

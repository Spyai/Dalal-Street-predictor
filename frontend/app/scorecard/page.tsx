"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { CallClass, Champion, LedgerVerify, PredictionRow, Scorecard } from "@/lib/types";
import { fmtDate, fmtIstDateTime, fmtNum, fmtPct } from "@/lib/format";
import { Notice } from "@/components/Status";
import { COMPARISON_TONE, comparisonText, fmtProb } from "@/components/MarketCall";

// The market-call scorecard (brief §6.3): every metric with its sample size, the naive
// baselines beside the model, calibration, the champion model, and the ledger check.

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

  async function verifyLedger() {
    setVerifying(true);
    setVerifyError(null);
    try {
      setVerify(await api<LedgerVerify>("/api/scorecard/verify-ledger", { method: "POST" }));
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

      <Section title="Ledger">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={verifyLedger}
            disabled={verifying}
            className="rounded border border-line px-3 py-1 text-sm hover:bg-panel-2 disabled:opacity-40"
          >
            {verifying ? "Verifying…" : "Verify ledger"}
          </button>
          {verify && verify.ok && (
            <span className="text-sm text-up">Intact: {verify.message}.</span>
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
          Every record carries a SHA-256 over its content and the previous record&apos;s hash.
          Verification recomputes the whole chain from the database; any edit, gap or removal
          shows up as the first bad sequence number.
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

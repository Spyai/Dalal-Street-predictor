"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Health } from "@/lib/types";
import { ago, fmtIstDateTime } from "@/lib/format";
import { Notice, StatusPill } from "@/components/Status";

export default function HealthPage() {
  const [data, setData] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      api<Health>("/api/health")
        .then(setData)
        .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Failed to load"));
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data) return <p className="text-sm text-muted">{error ?? "Loading…"}</p>;

  const built = data.jobs.filter((j) => j.built);
  const planned = data.jobs.filter((j) => !j.built);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Health</h1>
        <Chip
          label={data.scheduler_mode === "external" ? "Scheduler (GitHub Actions)" : "Scheduler"}
          ok={data.scheduler_alive}
          detail={
            data.scheduler_heartbeat
              ? `${data.scheduler_mode === "external" ? "last job" : "heartbeat"} ${ago(data.scheduler_heartbeat)}`
              : data.scheduler_mode === "external"
                ? "no job has run yet"
                : "no heartbeat"
          }
        />
        <Chip label="Database" ok={data.database === "ok"} detail={data.database} />
        <Chip
          label={data.cache_backend === "postgres" ? "Cache (Postgres)" : "Redis"}
          ok={data.redis === "ok"}
          detail={data.redis}
        />
      </div>
      {!data.scheduler_alive && (
        <Notice tone="down">
          {data.scheduler_mode === "external" ? (
            <>
              No scheduled job has run recently. Check the &quot;Dalal jobs&quot; workflow in
              your GitHub repo&apos;s Actions tab, and that the <code>DALAL_JOBS_ENABLED</code>{" "}
              variable is <code>true</code>.
            </>
          ) : (
            <>
              The scheduler isn&apos;t running, so no scheduled job will fire. Check{" "}
              <code>docker compose logs scheduler</code>.
            </>
          )}
        </Notice>
      )}

      <Section title="Data freshness">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-faint">
            <tr>
              <th className="py-1 font-normal">Dataset</th>
              <th className="py-1 font-normal">Status</th>
              <th className="py-1 font-normal">Latest</th>
              <th className="hidden py-1 font-normal sm:table-cell">Expected</th>
            </tr>
          </thead>
          <tbody>
            {data.freshness.map((f) => (
              <tr key={f.dataset} className="border-t border-line align-top">
                <td className="py-1.5 pr-2">
                  {f.dataset}
                  {f.detail && <div className="text-xs text-muted">{f.detail}</div>}
                </td>
                <td className="py-1.5 pr-2"><StatusPill status={f.status} /></td>
                <td className="num py-1.5 pr-2 text-muted">
                  {f.latest ? (f.latest.length > 10 ? fmtIstDateTime(f.latest) : f.latest) : "—"}
                </td>
                <td className="num hidden py-1.5 text-muted sm:table-cell">{f.expected ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Sources">
        <div className="grid gap-2 sm:grid-cols-2">
          {data.sources.map((s) => (
            <div key={s.source} className="card p-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="font-medium">{s.label}</span>
                <StatusPill status={s.status} />
              </div>
              <div className="mt-1 text-xs text-muted">
                Last success: {s.last_success_at ? `${fmtIstDateTime(s.last_success_at)} IST` : "never"}
                {s.data_as_of && ` · data as of ${fmtIstDateTime(s.data_as_of)}`}
              </div>
              {s.last_error && s.status !== "ok" && (
                <div className="mt-1 break-words text-xs text-down">{s.last_error}</div>
              )}
              {s.status !== "ok" && s.hint && <div className="mt-1 text-xs text-faint">{s.hint}</div>}
            </div>
          ))}
        </div>
      </Section>

      <Section title="Scheduled jobs (IST)">
        <JobTable jobs={built} />
      </Section>

      <Section title="Planned jobs (not built yet)">
        <ul className="space-y-1 text-sm text-muted">
          {planned.map((j) => (
            <li key={j.name} className="flex justify-between gap-2 border-t border-line pt-1">
              <span>{j.label}</span>
              <span className="text-xs text-faint">{j.schedule} · Phase {j.phase}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Models in production">
        <p className="text-sm text-muted">{data.models_note}</p>
      </Section>
    </div>
  );
}

function JobTable({ jobs }: { jobs: Health["jobs"] }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs text-faint">
        <tr>
          <th className="py-1 font-normal">Job</th>
          <th className="py-1 font-normal">Last run</th>
          <th className="hidden py-1 font-normal md:table-cell">Last success</th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((j) => (
          <tr key={j.name} className="border-t border-line align-top">
            <td className="py-1.5 pr-2">
              {j.label}
              <div className="text-xs text-faint">{j.schedule}</div>
            </td>
            <td className="py-1.5 pr-2">
              <div className="flex items-center gap-2">
                <StatusPill status={j.last_status} />
                <span className="text-xs text-muted">{j.last_run_at ? ago(j.last_run_at) : "never"}</span>
              </div>
              {j.last_message && (
                <div className="mt-0.5 line-clamp-3 break-words text-xs text-muted">{j.last_message}</div>
              )}
            </td>
            <td className="num hidden py-1.5 text-xs text-muted md:table-cell">
              {j.last_success_at ? `${fmtIstDateTime(j.last_success_at)} IST` : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
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

function Chip({ label, ok, detail }: { label: string; ok: boolean; detail: string }) {
  return (
    <span className={`rounded border px-2 py-0.5 text-xs ${ok ? "border-up/40 text-up" : "border-down/40 text-down"}`}>
      {label}: {ok ? "ok" : "down"} <span className="text-faint">({detail})</span>
    </span>
  );
}

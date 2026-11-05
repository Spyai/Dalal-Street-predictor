const STYLES: Record<string, string> = {
  ok: "border-up/40 text-up",
  success: "border-up/40 text-up",
  partial: "border-warn/40 text-warn",
  stale: "border-warn/40 text-warn",
  auth_required: "border-warn/40 text-warn",
  skipped: "border-line text-muted",
  running: "border-accent/40 text-accent",
  failed: "border-down/40 text-down",
  unavailable: "border-down/40 text-down",
  missing: "border-down/40 text-down",
  not_configured: "border-line text-muted",
  never_used: "border-line text-faint",
};

const LABELS: Record<string, string> = {
  auth_required: "login needed",
  not_configured: "not configured",
  never_used: "not used yet",
};

export function StatusPill({ status }: { status: string | null }) {
  const s = status ?? "never_used";
  return (
    <span
      className={`inline-block rounded border px-1.5 py-px text-[10px] uppercase tracking-wide ${STYLES[s] ?? "border-line text-muted"}`}
    >
      {LABELS[s] ?? s.replace(/_/g, " ")}
    </span>
  );
}

export function Notice({
  tone = "warn",
  children,
}: {
  tone?: "warn" | "info" | "down";
  children: React.ReactNode;
}) {
  const cls =
    tone === "down"
      ? "border-down/40 bg-down/5"
      : tone === "info"
        ? "border-line bg-panel-2"
        : "border-warn/40 bg-warn/5";
  return <div className={`rounded-md border px-3 py-2 text-sm ${cls}`}>{children}</div>;
}

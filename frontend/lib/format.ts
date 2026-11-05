const IST = "Asia/Kolkata";

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-IN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}%`;
}

export function fmtCr(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}₹${Math.abs(v).toLocaleString("en-IN", { maximumFractionDigits: 0 })} cr`;
}

export function tone(v: number | null | undefined): string {
  if (v === null || v === undefined || v === 0) return "text-muted";
  return v > 0 ? "text-up" : "text-down";
}

export function fmtIstDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: IST,
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(iso));
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00+05:30` : iso);
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: IST,
    weekday: "short",
    day: "2-digit",
    month: "short",
  }).format(d);
}

export function ago(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "never";
  const mins = Math.round((now.getTime() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

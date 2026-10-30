"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, ApiError } from "@/lib/api";

function LoginForm() {
  const params = useSearchParams();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [totp, setTotp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password, totp }),
      });
      const next = params.get("next");
      window.location.replace(next && next.startsWith("/") && !next.startsWith("//") ? next : "/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed");
      setTotp("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="card mx-auto mt-16 w-full max-w-sm space-y-4 p-6">
      <div>
        <h1 className="text-lg font-semibold">Dalal</h1>
        <p className="text-sm text-muted">Password and authenticator code.</p>
      </div>
      <Field label="Username" value={username} onChange={setUsername} autoComplete="username" />
      <Field
        label="Password"
        type="password"
        value={password}
        onChange={setPassword}
        autoComplete="current-password"
      />
      <Field
        label="6-digit code"
        value={totp}
        onChange={(v) => setTotp(v.replace(/\D/g, "").slice(0, 6))}
        autoComplete="one-time-code"
        inputMode="numeric"
      />
      {error && <p className="text-sm text-down">{error}</p>}
      <button
        type="submit"
        disabled={busy || !username || !password || totp.length !== 6}
        className="w-full rounded-md bg-accent py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        {busy ? "Checking…" : "Log in"}
      </button>
    </form>
  );
}

function Field(props: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  autoComplete?: string;
  inputMode?: "numeric";
}) {
  return (
    <label className="block space-y-1">
      <span className="label">{props.label}</span>
      <input
        className="w-full rounded-md border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
        type={props.type ?? "text"}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        autoComplete={props.autoComplete}
        inputMode={props.inputMode}
        required
      />
    </label>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}

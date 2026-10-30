"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import type { SetupResult, SetupStatus } from "@/lib/types";

function SetupForm() {
  const params = useSearchParams();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [token, setToken] = useState(params.get("token") ?? "");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SetupResult | null>(null);

  useEffect(() => {
    api<SetupStatus>("/api/auth/setup-status")
      .then(setStatus)
      .catch(() => setError("Can't reach the API. Check BACKEND_URL and /healthz."));
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords don't match.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api<SetupResult>("/api/auth/setup", {
          method: "POST",
          body: JSON.stringify({ token, username, password }),
        }),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Setup failed");
    } finally {
      setBusy(false);
    }
  }

  if (result) {
    return (
      <div className="card mx-auto mt-10 w-full max-w-sm space-y-4 p-6">
        <h1 className="text-lg font-semibold">Scan this once</h1>
        <p className="text-sm text-muted">
          Add it to an authenticator app (Google Authenticator, 1Password, Aegis). This code is
          shown only now.
        </p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={result.qr_svg_data_url}
          alt="TOTP QR code"
          className="mx-auto w-56 rounded bg-white p-2"
        />
        <div>
          <div className="label">Or enter this key manually</div>
          <code className="block break-all text-sm">{result.totp_secret}</code>
        </div>
        <a
          href="/login"
          className="block w-full rounded-md bg-accent py-2 text-center text-sm font-medium text-white"
        >
          I&apos;ve saved it. Log in
        </a>
      </div>
    );
  }

  if (status && !status.needs_setup) {
    return (
      <div className="card mx-auto mt-16 w-full max-w-sm space-y-3 p-6 text-sm">
        <p>Dalal is already set up.</p>
        <a href="/login" className="text-accent underline">
          Go to login
        </a>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="card mx-auto mt-16 w-full max-w-sm space-y-4 p-6">
      <div>
        <h1 className="text-lg font-semibold">Set up Dalal</h1>
        <p className="text-sm text-muted">
          One-time: create the only login. Needs the <code>SETUP_TOKEN</code> you set on the
          server.
        </p>
      </div>
      {status && !status.setup_enabled && (
        <p className="text-sm text-warn">
          Web setup is off: <code>SETUP_TOKEN</code> isn&apos;t set (min 16 characters) on the API.
        </p>
      )}
      <Field label="Setup token" value={token} onChange={setToken} type="password" />
      <Field label="Username" value={username} onChange={setUsername} autoComplete="username" />
      <Field
        label="Password (min 12 characters)"
        type="password"
        value={password}
        onChange={setPassword}
        autoComplete="new-password"
      />
      <Field
        label="Repeat password"
        type="password"
        value={confirm}
        onChange={setConfirm}
        autoComplete="new-password"
      />
      {error && <p className="text-sm text-down">{error}</p>}
      <button
        type="submit"
        disabled={busy || !token || !username || password.length < 12}
        className="w-full rounded-md bg-accent py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        {busy ? "Creating…" : "Create login"}
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
        required
      />
    </label>
  );
}

export default function SetupPage() {
  return (
    <Suspense>
      <SetupForm />
    </Suspense>
  );
}

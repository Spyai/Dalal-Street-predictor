// Thin fetch wrapper. Same-origin (/api is proxied to FastAPI), cookie-based session.

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function toLogin(): void {
  if (typeof window === "undefined") return;
  const here = window.location.pathname;
  if (here.startsWith("/login") || here.startsWith("/setup")) return;
  const next = encodeURIComponent(window.location.pathname + window.location.search);
  window.location.replace(`/login?next=${next}`);
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (res.status === 401 && !path.startsWith("/api/auth/login")) {
    toLogin();
    throw new ApiError(401, "Not logged in");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

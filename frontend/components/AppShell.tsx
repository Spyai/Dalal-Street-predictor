"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const NAV = [
  { href: "/", label: "Today" },
  { href: "/stock/INDEX/NIFTY%2050", label: "Stocks", match: "/stock" },
  { href: "/health", label: "Health" },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLogin = pathname.startsWith("/login") || pathname.startsWith("/setup");
  const [user, setUser] = useState<string | null>(null);

  useEffect(() => {
    if (isLogin) return;
    api<{ username: string }>("/api/auth/me")
      .then((me) => setUser(me.username))
      .catch(() => setUser(null));
  }, [isLogin]);

  return (
    <div className="flex min-h-screen flex-col">
      {!isLogin && (
        <header className="sticky top-0 z-20 border-b border-line bg-bg/90 backdrop-blur">
          <div className="mx-auto flex h-12 max-w-6xl items-center gap-4 px-4">
            <Link href="/" className="font-semibold tracking-tight">
              Dalal
            </Link>
            <nav className="flex gap-1 text-sm">
              {NAV.map((n) => {
                const active =
                  n.match ? pathname.startsWith(n.match) : pathname === n.href;
                return (
                  <Link
                    key={n.href}
                    href={n.href}
                    className={`rounded px-2 py-1 ${active ? "bg-panel-2 text-fg" : "text-muted hover:text-fg"}`}
                  >
                    {n.label}
                  </Link>
                );
              })}
            </nav>
            <div className="ml-auto flex items-center gap-3 text-xs text-muted">
              <ThemeToggle />
              {user && (
                <button
                  className="hover:text-fg"
                  onClick={async () => {
                    await api("/api/auth/logout", { method: "POST" }).catch(() => undefined);
                    window.location.replace("/login");
                  }}
                >
                  Log out
                </button>
              )}
            </div>
          </div>
        </header>
      )}
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-5">{children}</main>
      <footer className="border-t border-line py-4 text-center text-xs text-faint">
        Personal research tool. Not investment advice.
      </footer>
    </div>
  );
}

function ThemeToggle() {
  const [light, setLight] = useState(false);
  useEffect(() => setLight(document.documentElement.dataset.theme === "light"), []);
  return (
    <button
      className="hover:text-fg"
      aria-label="Toggle light/dark theme"
      onClick={() => {
        const next = !light;
        setLight(next);
        if (next) document.documentElement.dataset.theme = "light";
        else delete document.documentElement.dataset.theme;
        try {
          localStorage.setItem("dalal-theme", next ? "light" : "dark");
        } catch {
          /* storage unavailable */
        }
        window.dispatchEvent(new Event("dalal-theme"));
      }}
    >
      {light ? "Dark" : "Light"}
    </button>
  );
}

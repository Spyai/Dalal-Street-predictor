import type { NextConfig } from "next";

// The browser only ever talks to this Next.js server; /api/* is proxied to FastAPI so
// the session cookie stays same-origin. BACKEND_URL is read at build time.
//   Docker Compose: http://backend:8000 (default)
//   Vercel: the URL of the dalal-api project, e.g. https://dalal-api.vercel.app
const onVercel = Boolean(process.env.VERCEL);
if (onVercel && !process.env.BACKEND_URL) {
  throw new Error("Set BACKEND_URL in this Vercel project to the dalal-api URL (see DEPLOY_VERCEL.md).");
}
const backend = (process.env.BACKEND_URL ?? "http://backend:8000").replace(/\/$/, "");

const securityHeaders = [
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "same-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const config: NextConfig = {
  // Standalone output is for the Docker image; Vercel builds its own output.
  output: onVercel ? undefined : "standalone",
  poweredByHeader: false,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/healthz", destination: `${backend}/healthz` },
    ];
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default config;

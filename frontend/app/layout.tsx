import type { Metadata, Viewport } from "next";
import AppShell from "@/components/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dalal",
  description: "Private NSE/BSE research desk",
  robots: { index: false, follow: false },
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0a0e13",
};

// Applies a saved light-theme preference before paint; dark is the default.
const themeScript = `try{if(localStorage.getItem("dalal-theme")==="light")document.documentElement.dataset.theme="light"}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}

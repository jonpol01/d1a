// Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
// Changes for D1A Copyright 2026 John Soliva: rebranded the UI to D1A (title and header text, the /d1a API proxy, the d1a-latest model name).
import type { NextConfig } from "next";

// FastAPI (d1a.serve) is proxied under /d1a so the browser never deals with CORS or ports.
const D1A_API = process.env.D1A_API ?? "http://127.0.0.1:8009";

const nextConfig: NextConfig = {
  reactCompiler: true,
  devIndicators: false,
  // Next dev only trusts the hostname it was started with (localhost); without this,
  // opening the app via 127.0.0.1 renders the SSR HTML but never hydrates (no errors, buttons dead).
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    return [{ source: "/d1a/:path*", destination: `${D1A_API}/:path*` }];
  },
};

export default nextConfig;

import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";

/** Hash of the UI sources the build is made from. The backend recomputes it (backend/buildinfo.py, same algorithm)
 *  and reports `stale: true` in /api/health when the served web/out was built from older sources - the audit found
 *  the server shipping a month-old UI without anyone noticing. */
const SOURCE_DIRS = ["app", "components", "hooks", "lib"];
function sourceHash(root) {
  const files = [];
  const walk = (d) => { for (const n of readdirSync(d)) { const p = join(d, n); statSync(p).isDirectory() ? walk(p) : files.push(p); } };
  for (const d of SOURCE_DIRS) walk(join(root, d));
  files.push(join(root, "package.json"));
  const rel = files.map((f) => [relative(root, f).split(sep).join("/"), f]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const h = createHash("sha256");
  for (const [r, f] of rel) { h.update(r + "\0"); h.update(readFileSync(f)); h.update("\0"); }
  return h.digest("hex").slice(0, 16);
}

if (process.argv.includes("build")) {
  writeFileSync(join(process.cwd(), "public", "build-info.json"),
    JSON.stringify({ source_hash: sourceHash(process.cwd()), built_at: new Date().toISOString() }) + "\n");
}

/** Static export: FastAPI serves web/out at "/", so API, WebSocket and UI share one origin (no CORS, no proxy). */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};
// In `next dev`, call the API directly: NEXT_PUBLIC_API_URL=http://localhost:8000 (CORS allows localhost:3000).
export default nextConfig;

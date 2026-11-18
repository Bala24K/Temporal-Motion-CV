import type { RepDetail } from "@/lib/types";

/** Per-rep strip: bar height = how far through the range the rep went; amber = a form check failed;
 *  dashed = partial (not counted). Used in the live set summary and in History. */
export function RepStrip({ reps, compact = false }: { reps: RepDetail[]; compact?: boolean }) {
  if (!reps.length) return null;
  return (
    <div className="mt-2">
      <div className={`flex items-end gap-1 ${compact ? "h-12" : "h-16"}`} role="img"
        aria-label={`Depth per rep: ${reps.map((r) => Math.round(r.peak_progress * 100) + "%").join(", ")}`}>
        {reps.map((r, i) => {
          const fails = r.evidence.filter((e) => e.evaluated !== false && !e.passed && e.rule_id !== "partial");
          return <div key={i} title={`Rep ${i + 1}: ${Math.round(r.peak_progress * 100)}% of range, ${r.duration.toFixed(1)} s${fails.length ? " · " + fails.map((f) => f.message).join("; ") : ""}`}
            className={`w-5 rounded-t ${r.counted ? (fails.length ? "bg-warn" : "bg-brand") : "border-2 border-dashed border-mute bg-transparent"}`}
            style={{ height: `${Math.max(8, Math.min(100, r.peak_progress * 100))}%` }} />;
        })}
      </div>
      <p className="mt-1 text-xs text-mute">Bar = depth of each rep · amber = a form check failed · dashed = partial, not counted</p>
    </div>);
}

/** Most frequent failed checks across reps, e.g. [["Keep your elbows by your sides", 3], ...]. */
export function topIssues(reps: RepDetail[]): [string, number][] {
  const n: Record<string, number> = {};
  for (const r of reps) for (const e of r.evidence) if (e.evaluated !== false && !e.passed && e.rule_id !== "partial") n[e.message] = (n[e.message] ?? 0) + 1;
  return Object.entries(n).sort((a, b) => b[1] - a[1]);
}

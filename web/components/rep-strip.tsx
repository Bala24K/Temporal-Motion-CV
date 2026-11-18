import type { Evidence, RepDetail } from "@/lib/types";

const isWarning = (e: Evidence) => e.evaluated !== false && !e.passed && e.rule_id !== "partial" && e.severity !== "info";

/** Per-rep strip: bar height = how far through the range the rep went; amber = a form check failed;
 *  dashed = partial (not counted). Used in the live set summary and in History. */
export function RepStrip({ reps, compact = false }: { reps: RepDetail[]; compact?: boolean }) {
  if (!reps.length) return null;
  return (
    <div className="mt-2">
      <div className={`flex items-end gap-1 ${compact ? "h-12" : "h-16"}`} role="img"
        aria-label={`Depth per rep: ${reps.map((r) => Math.round(r.peak_progress * 100) + "%").join(", ")}`}>
        {reps.map((r, i) => {
          const fails = r.evidence.filter(isWarning);
          return <div key={i} title={`Rep ${i + 1}: ${Math.round(r.peak_progress * 100)}% of range, ${r.duration.toFixed(1)} s${fails.length ? " · " + fails.map((f) => f.message).join("; ") : ""}`}
            className={`w-5 rounded-t ${r.counted ? (fails.length ? "bg-warn" : "bg-brand") : "border-2 border-dashed border-mute bg-transparent"}`}
            style={{ height: `${Math.max(8, Math.min(100, r.peak_progress * 100))}%` }} />;
        })}
      </div>
      <p className="mt-1 text-xs text-mute">Bar = depth of each rep · amber = a form check failed · dashed = partial, not counted</p>
    </div>);
}

/** Most frequent failed checks across reps, e.g. [["Keep your elbows by your sides", 3, worstEvidence], ...].
 *  Info-level tips are left out. The third element is the rep that missed the threshold by the most. */
export function topIssues(reps: RepDetail[]): [string, number, Evidence][] {
  const n: Record<string, [number, Evidence]> = {};
  const miss = (e: Evidence) => (e.measured != null && e.threshold != null ? Math.abs(e.measured - e.threshold) : 0);
  for (const r of reps) for (const e of r.evidence) if (isWarning(e)) {
    const cur = n[e.message];
    n[e.message] = cur ? [cur[0] + 1, miss(e) > miss(cur[1]) ? e : cur[1]] : [1, e];
  }
  return Object.entries(n).map(([m, [k, e]]) => [m, k, e] as [string, number, Evidence]).sort((a, b) => b[1] - a[1]);
}

const unit = (signal?: string | null) => (!signal ? "" : /angle|lean/.test(signal) ? "°" : /duration/.test(signal) ? " s" : /progress/.test(signal) ? "" : "");
const num = (v: number, signal?: string | null) => (signal && /progress/.test(signal) ? `${Math.round(v * 100)}%` : `${Math.round(v * 10) / 10}${unit(signal)}`);

/** "measured 18° · limit 12° · front view · rule not yet validated" - the evidence behind a warning. */
export function EvidenceLine({ e }: { e: Evidence }) {
  if (e.measured == null || e.threshold == null) return null;
  const bits = [`measured ${num(e.measured, e.signal)}`, `limit ${num(e.threshold, e.signal)}`];
  if (e.view) bits.push(`${e.view === "45deg" ? "45°" : e.view} view`);
  if (e.status === "provisional") bits.push("check not yet validated");
  return <span className="block text-xs text-mute" data-testid="evidence-line">{bits.join(" · ")}</span>;
}

"use client";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { SessionOut, Workout } from "@/lib/types";
import { Shell } from "@/components/shell";
import { LiveWorkout, type QueueItem } from "@/components/live-workout";
import { Button } from "@/components/ui/button";
import { Card, Skeleton, Stat } from "@/components/ui/primitives";
import { Badge } from "@/components/ui/primitives";
import { toastError } from "@/components/toast";

function Runner() {
  const q = useSearchParams(), router = useRouter(), qc = useQueryClient();
  const [state, setState] = useState<{ session: { id: number }; queue: QueueItem[] } | null>(null);
  const [done, setDone] = useState<SessionOut | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const wid = q.get("workout");
        const w = wid ? (await api<Workout[]>("/api/workouts")).find((x) => x.id === +wid) : null;
        if (wid && !w) throw new Error("Workout not found");
        const session = await api<{ id: number }>("/api/sessions", { body: { workout_id: w?.id ?? null, name: w?.name ?? "Quick set" } });
        const queue: QueueItem[] = w ? w.items.flatMap((item) => Array.from({ length: item.sets }, (_, s) => ({ item, set: s + 1, of: item.sets })))
          : [{ item: { id: 0, exercise_id: "auto", exercise_name: "Auto-detect", sets: 1, reps: 10, weight_kg: 0, rest_seconds: 0, ai_analysis: true, target_muscles: [] }, set: 1, of: 1 }];
        if (!cancelled) setState({ session, queue });
      } catch (e) { toastError(e); router.replace("/workout/"); }
    })();
    return () => { cancelled = true; };
  }, [q, router]);
  const finish = async () => { if (!state) return; const s = await api<SessionOut>(`/api/sessions/${state.session.id}/finish`, { method: "POST" }); qc.invalidateQueries(); setDone(s); setState(null); };
  if (done) return (
    <Shell><h1 className="font-display text-4xl font-extrabold">Session complete.</h1>
      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4"><Stat value={done.sets.length} label="sets" /><Stat value={done.total_reps} label="reps" /><Stat value={Math.round(done.volume_kg)} label="kg volume" /><Stat value={done.avg_form_score ?? "–"} label="avg form" /></div>
      <Card className="mt-4"><table className="w-full text-sm"><thead><tr className="text-left text-mute"><th className="py-1">Exercise</th><th>Reps</th><th>kg</th><th>Form</th></tr></thead><tbody>
        {done.sets.map((s) => <tr key={s.id} className="border-t border-line"><td className="py-2">{s.exercise_name} {s.source === "manual" && <Badge tone="warn">manual</Badge>}</td><td>{s.reps}{s.partial_reps ? <span className="text-mute"> (+{s.partial_reps} partial)</span> : null}</td><td>{s.weight_kg}</td><td>{s.avg_form_score ?? "–"}</td></tr>)}</tbody></table></Card>
      <Button variant="primary" className="mt-4" onClick={() => router.push("/progress/")}>See progress</Button></Shell>);
  if (!state) return <Shell><Skeleton className="h-64" /></Shell>;
  return <Shell><Skeleton className="h-64" /><LiveWorkout sessionId={state.session.id} queue={state.queue} onDone={finish} /></Shell>;
}

export default function LivePage() { return <Suspense fallback={null}><Runner /></Suspense>; }

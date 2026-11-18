"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { SessionOut, SetOut } from "@/lib/types";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge, Card, Notice, Skeleton } from "@/components/ui/primitives";
import { fmt } from "@/lib/utils";
import { EvidenceLine, RepStrip, topIssues } from "@/components/rep-strip";

const when = (iso: string) => new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function SetRow({ s }: { s: SetOut }) {
  const [open, setOpen] = useState(false);
  const detail = s.reps_detail ?? [];
  const issues = topIssues(detail);
  return (
    <li className="border-t border-line py-2">
      <button className="flex w-full items-center gap-2 text-left" onClick={() => setOpen(!open)} aria-expanded={open} disabled={!detail.length}>
        {detail.length ? (open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />) : <span className="w-4" />}
        <span className="flex-1">{s.exercise_name} <span className="text-mute">set {s.set_number}</span> {s.source === "manual" && <Badge tone="warn">manual</Badge>}</span>
        <span className="tabular-nums">{s.reps}{s.partial_reps ? <span className="text-mute"> +{s.partial_reps}p</span> : null} × {s.weight_kg}kg</span>
        <span className="w-14 text-right tabular-nums text-mute">{s.avg_form_score ?? "–"}</span>
      </button>
      {open && <div className="pl-6"><RepStrip reps={detail} />
        {issues.length > 0 && <ul className="mt-2 text-sm">{issues.map(([m, n, e]) => <li key={m} className="text-warn">{m} <span className="text-mute">({n} rep{n > 1 ? "s" : ""})</span><EvidenceLine e={e} /></li>)}</ul>}
        {s.avg_rep_seconds != null && <p className="mt-1 text-sm text-mute">Average rep {s.avg_rep_seconds.toFixed(1)} s{s.avg_rom != null ? ` · average depth ${Math.round(s.avg_rom * 100)}% of range` : ""}</p>}</div>}
    </li>);
}

function SessionDetail({ id }: { id: number }) {
  const q = useQuery({ queryKey: ["session", id], queryFn: () => api<SessionOut>(`/api/sessions/${id}`) });
  if (q.isLoading) return <Skeleton className="mt-3 h-24" />;
  if (q.error) return <Notice tone="bad">Couldn&apos;t load this session. <Button size="sm" onClick={() => q.refetch()}>Retry</Button></Notice>;
  const d = q.data!;
  if (!d.sets.length) return <p className="mt-2 text-sm text-mute">No sets were logged in this session.</p>;
  return (<ul className="mt-2 text-sm" aria-label="Sets">{d.sets.map((s) => <SetRow key={s.id} s={s} />)}</ul>);
}

export default function History() {
  const q = useQuery({ queryKey: ["sessions"], queryFn: () => api<SessionOut[]>("/api/sessions?limit=100") });
  const [open, setOpen] = useState<number | null>(null);
  const [showEmpty, setShowEmpty] = useState(false);
  const list = (q.data ?? []).filter((s) => showEmpty || s.sets.length > 0);
  return (
    <Shell>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-3xl font-extrabold">History</h1>
        <label className="flex items-center gap-2 text-sm text-mute"><input type="checkbox" checked={showEmpty} onChange={(e) => setShowEmpty(e.target.checked)} />Show empty sessions</label>
      </div>
      {q.isLoading && <Skeleton className="mt-4 h-48" />}
      {q.error && <div className="mt-4"><Notice tone="bad">Couldn&apos;t load your history. <Button size="sm" onClick={() => q.refetch()}>Retry</Button></Notice></div>}
      {q.data && list.length === 0 && <p className="mt-4 text-mute">No workouts yet. Finished sessions will appear here with every set and rep.</p>}
      <div className="mt-4 space-y-3">
        {list.map((s) => (
          <Card key={s.id} className="p-4">
            <button className="flex w-full flex-wrap items-baseline gap-x-4 gap-y-1 text-left" onClick={() => setOpen(open === s.id ? null : s.id)} aria-expanded={open === s.id} data-testid="history-session">
              <span className="text-lg font-bold">{s.name}</span>
              <span className="text-sm text-mute">{when(s.started_at)}{s.finished_at ? ` · ${Math.round(s.duration_minutes)} min` : " · not finished"}</span>
              <span className="ml-auto text-sm tabular-nums">{s.sets.length} sets · {s.total_reps} reps · {fmt(s.volume_kg)} kg{s.avg_form_score != null ? ` · form ${s.avg_form_score}` : ""}</span>
            </button>
            {open === s.id && <SessionDetail id={s.id} />}
          </Card>))}
      </div>
    </Shell>);
}

"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ExerciseProgress, GoalProgress, Progress, Trend } from "@/lib/types";
import { Shell } from "@/components/shell";
import { BarsChart, TrendChart } from "@/components/charts";
import { Tabs } from "@/components/ui/tabs";
import { Badge, Card, Meter, Select, Skeleton, Stat } from "@/components/ui/primitives";
import { fmt, plural, titleCase } from "@/lib/utils";
import Link from "next/link";
import { ErrorState } from "@/components/ui/states";

/** AI-derived metrics are shown, but marked: they are not validated against human-labelled data yet. */
const Experimental = () => <span className="ml-2 align-middle text-[10px] font-semibold uppercase tracking-wider text-warn" title="Measured by the camera engine; not yet validated against human-reviewed sets">experimental</span>;

type R = "today" | "week" | "month";
const tone = (t: Trend) => (t.direction === "improving" ? "good" : t.direction === "declining" ? "warn" : "neutral");

export default function ProgressPage() {
  const [range, setRange] = useState<R>("week"), [ex, setEx] = useState("");
  const p = useQuery({ queryKey: ["progress", range], queryFn: () => api<Progress>(`/api/progress?range=${range}`) });
  const exs = useQuery({ queryKey: ["prog-ex"], queryFn: () => api<{ name: string; sets: number; best_e1rm: number | null }[]>("/api/progress/exercises") });
  const pe = useQuery({ queryKey: ["prog-ex", ex], queryFn: () => api<ExerciseProgress>(`/api/progress/exercise/${encodeURIComponent(ex)}`), enabled: !!ex });
  const goal = useQuery({ queryKey: ["goal"], queryFn: () => api<GoalProgress>("/api/progress/goal") });
  const d = p.data;
  return (
    <Shell>
      <div className="flex flex-wrap items-center justify-between gap-2"><h1 className="font-display text-3xl font-extrabold">Progress</h1>
        <Link href="/history/" className="text-sm font-semibold text-brand hover:underline">See every session →</Link></div>
      {p.error && <div className="mt-4"><ErrorState what="your progress" onRetry={() => p.refetch()} /></div>}
      <div className="mt-3"><Tabs label="Range" value={range} onChange={setRange} items={[{ id: "today", label: "Today" }, { id: "week", label: "Week" }, { id: "month", label: "Month" }]} /></div>
      {!d ? (p.error ? null : <Skeleton className="mt-4 h-48" />) : (<>
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3">
          <Stat value={d.workouts} label={`${d.workouts === 1 ? "workout" : "workouts"} · ${plural(d.days_trained, "training day")}`} /><Stat value={d.reps} label={`reps in ${plural(d.sets, "set")}`} /><Stat value={fmt(d.volume_kg)} label="kg volume" />
          <Stat value={<>{fmt(d.avg_form_score)}<Experimental /></>} label="form checks passed" /><Stat value={<>{d.avg_rom != null ? `${Math.round(d.avg_rom * 100)}%` : "–"}<Experimental /></>} label="average depth (of your range)" /><Stat value={`${Math.round(d.consistency * 100)}%`} label="of weekly target" /></div>
        {range !== "today" && <div className="mt-4 grid gap-3 md:grid-cols-2"><Card><h2 className="font-bold">Volume</h2><BarsChart data={d.series} x="date" y="volume_kg" label="Volume by day" /></Card><Card><h2 className="font-bold">Reps</h2><BarsChart data={d.series} x="date" y="reps" label="Reps by day" /></Card></div>}
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <Card><h2 className="font-bold">Muscle distribution</h2><div className="mt-2 space-y-2">{Object.entries(d.muscle_share).map(([m, s]) => <div key={m}><div className="flex justify-between text-sm"><span>{titleCase(m)}</span><span className="text-mute">{s}% · {plural(d.muscle_sets[m], "set")}</span></div><Meter value={s} label={`${titleCase(m)} share of sets`} /></div>)}{!Object.keys(d.muscle_share).length && <p className="text-mute">No analysed sets yet.</p>}</div></Card>
          <Card><h2 className="font-bold">Personal bests</h2><table className="mt-2 w-full text-sm"><tbody>{Object.entries(d.personal_bests).map(([k, v]) => <tr key={k} className="border-t border-line"><td className="py-2">{k}</td><td>{v.max_weight_kg}kg</td><td className="text-mute">{v.max_reps} reps</td></tr>)}</tbody></table></Card></div>
        {d.unsupported_sets > 0 && <p className="mt-3 text-sm text-mute">{d.unsupported_sets} set(s) were tracked manually, without AI analysis.</p>}</>)}
      <Card className="mt-6"><div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-xl font-bold">By exercise</h2>
        <Select aria-label="Exercise" className="w-56" value={ex} onChange={(e) => setEx(e.target.value)}><option value="">Choose…</option>{exs.data?.map((e) => <option key={e.name} value={e.name}>{e.name} ({e.sets} sets)</option>)}</Select></div>
        {pe.data && (<div className="mt-3"><div className="flex flex-wrap gap-2">{Object.entries(pe.data.trends).map(([k, t]) => <Badge key={k} tone={tone(t)}>{k}: {t.direction.replace("_", " ")}</Badge>)}</div>
          <div className="mt-3 grid gap-3 md:grid-cols-2"><div><h3 className="text-sm font-semibold text-mute">Estimated 1RM (kg)</h3><TrendChart data={pe.data.series} x="date" y="best_e1rm" label="Estimated one-rep max over time" /></div>
            <div><h3 className="text-sm font-semibold text-mute">Range of motion</h3><TrendChart data={pe.data.series.map((s) => ({ ...s, avg_rom: s.avg_rom && Math.round(s.avg_rom * 100) }))} x="date" y="avg_rom" label="Range of motion over time" /></div></div>
          {pe.data.personal_records.length > 0 && <p className="mt-2 text-sm text-mute">PRs: {pe.data.personal_records.map((r) => `${r.date.slice(5)} → ${r.e1rm}kg`).join(" · ")}</p>}</div>)}
        {!ex && <p className="mt-2 text-mute">Pick an exercise to see strength, range and tempo trends.</p>}</Card>
      {goal.data && <Card className="mt-4"><h2 className="text-xl font-bold">Goal</h2><p className="text-mute">{goal.data.headline}</p>
        <div className="mt-3 flex items-end gap-1.5" role="img" aria-label="Sessions per week">{goal.data.weeks.map((w) => <div key={w.week_start} className="flex flex-1 flex-col items-center gap-1" title={`${w.week_start}: ${w.sessions} sessions`}>
          <div className={`w-full rounded-t ${w.target_met ? "bg-good" : "bg-brand/60"}`} style={{ height: Math.max(4, w.sessions * 18) }} /><span className="text-[10px] text-mute">{w.week_start.slice(5)}</span></div>)}</div>
        <p className="mt-2 text-xs text-mute">Green = weekly target ({goal.data.weekly_target}) met.</p></Card>}
    </Shell>);
}

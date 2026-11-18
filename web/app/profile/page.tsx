"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, useAuth } from "@/lib/api";
import type { ProfileOut, User } from "@/lib/types";
import { Shell } from "@/components/shell";
import { CalibrationWizard } from "@/components/calibration";
import dynamic from "next/dynamic";
import type { Segments } from "@/components/skeleton-3d";
import { Button } from "@/components/ui/button";
import { Badge, Card, Input, Label, Select, Skeleton } from "@/components/ui/primitives";
import { toast, toastError } from "@/components/toast";
import { titleCase } from "@/lib/utils";

const SkeletonViewer3D = dynamic(() => import("@/components/skeleton-3d").then((m) => m.SkeletonViewer3D), { ssr: false, loading: () => <Skeleton className="h-72" /> });
const GOALS: Record<string, string> = { build_muscle: "Build muscle", get_stronger: "Get stronger", improve_fitness: "Improve fitness", lose_weight: "Lose weight", consistency: "Improve consistency", movement_quality: "Improve movement quality" };

function CalibrationStatus({ p }: { p: ProfileOut }) {
  const c = p.calibration;
  const tone = c?.status === "good" ? "good" : c?.status === "usable" ? "warn" : "bad";
  return (<div className="mt-2 space-y-1">
    <div className="flex flex-wrap items-center gap-2"><Badge tone={tone}>{c?.status === "good" ? "Calibrated" : c?.status === "usable" ? "Calibrated (usable)" : "Recalibrate"}</Badge>
      <span className="text-sm text-mute">{p.stages.map(titleCase).join(" · ")}{c ? ` · quality ${Math.round(c.quality * 100)}%` : ""}{c?.setup?.label ? ` · ${c.setup.label}` : ""}</span></div>
    {c?.warnings.map((w) => <p key={w} className="text-sm text-warn">{w}</p>)}
    {p.missing_stages.length > 0 && <p className="text-xs text-mute">Not captured: {p.missing_stages.map(titleCase).join(", ")}</p>}</div>);
}

export default function ProfilePage() {
  const qc = useQueryClient(), router = useRouter(), { user, setSession, token, clear } = useAuth();
  const p = useQuery({ queryKey: ["profile"], queryFn: () => api<ProfileOut>("/api/profile") });
  const [goal, setGoal] = useState<string | null>(null), [target, setTarget] = useState<number | null>(null);
  const save = useMutation({ mutationFn: () => api<User>("/api/auth/me", { method: "PATCH", body: { goal: (goal ?? user?.goal) || null, weekly_target: target ?? user?.weekly_target } }),
    onSuccess: (u) => { setSession(token!, u); qc.invalidateQueries({ queryKey: ["goal"] }); toast("Saved"); }, onError: toastError });
  const del = useMutation({ mutationFn: () => api("/api/auth/me", { method: "DELETE" }), onSuccess: () => { clear(); router.replace("/login/"); }, onError: toastError });
  const seg = p.data?.segments_cm, h = p.data?.profile.height_cm;
  return (
    <Shell>
      <h1 className="font-display text-3xl font-extrabold">Profile</h1>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Card><h2 className="text-xl font-bold">Goal</h2>
          <Label htmlFor="goal">What are you training for?</Label>
          <Select id="goal" value={goal ?? user?.goal ?? ""} onChange={(e) => setGoal(e.target.value)}><option value="">Not set</option>{Object.entries(GOALS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</Select>
          <Label htmlFor="wt">Workouts per week</Label><Input id="wt" type="number" min={1} max={14} value={target ?? user?.weekly_target ?? 3} onChange={(e) => setTarget(+e.target.value)} />
          <Button variant="primary" className="mt-4" onClick={() => save.mutate()}>Save</Button></Card>
        <Card><h2 className="text-xl font-bold">Your body model</h2>
          {p.isLoading ? <Skeleton className="mt-2 h-72" /> : p.data?.calibrated && seg && h ? (<>
            <CalibrationStatus p={p.data} />
            <div className="mt-3"><SkeletonViewer3D segments={seg as unknown as Segments} heightCm={h} /></div>
            <p className="mt-2 text-xs text-mute">Blue: your measured proportions. Grey: average proportions at your height.</p>
            <dl className="mt-2 grid grid-cols-2 gap-x-4 text-sm">{Object.entries(seg).map(([k, v]) => <div key={k} className="flex justify-between border-t border-line py-1"><dt className="text-mute">{titleCase(k)}</dt><dd>{v} cm</dd></div>)}</dl></>)
            : <>{p.data?.calibrated && <CalibrationStatus p={p.data} />}<p className="mt-2 text-mute">{p.data?.calibrated ? "Add your height when calibrating to see estimated sizes." : "Not calibrated yet. Use the wizard below."}</p></>}</Card>
      </div>
      <div className="mt-4"><CalibrationWizard initialHeight={h ?? null} /></div>
      <Card className="mt-4"><h2 className="text-xl font-bold">Your baselines</h2>
        {p.data && Object.keys(p.data.baselines).length ? <ul className="mt-2 divide-y divide-line text-sm">{Object.entries(p.data.baselines).map(([k, b]) => <li key={k} className="flex justify-between py-2"><span>{titleCase(k)}</span><span className="text-mute">{b.n} reps · usual range {Math.round(b.mean.peak_progress * 100)}% · {b.mean.duration.toFixed(1)}s/rep</span></li>)}</ul> : <p className="mt-2 text-mute">Baselines build as you train.</p>}</Card>
      <Card className="mt-4"><h3 className="font-bold">Privacy</h3><p className="mt-1 text-sm text-mute">Video is analysed in your browser and never stored or uploaded; only pose landmarks and set summaries are sent. You can erase everything at any time.</p>
        <Button variant="danger" className="mt-3" onClick={() => confirm("This permanently deletes your account, workouts and history.") && del.mutate()}>Delete my account and data</Button></Card>
    </Shell>);
}

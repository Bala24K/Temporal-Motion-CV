"use client";
import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Exercise } from "@/lib/types";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge, Card, Input, Notice, Skeleton } from "@/components/ui/primitives";
import { cn, titleCase } from "@/lib/utils";
import { toastError } from "@/components/toast";
import { CameraDiagram } from "@/components/camera-diagram";
import { ErrorState } from "@/components/ui/states";

interface Resolve { supported: boolean; message?: string; exercise?: Exercise; similar?: Exercise[] }

export default function Exercises() {
  const { data, isLoading, error, refetch } = useQuery({ queryKey: ["exercises"], queryFn: () => api<{ exercises: Exercise[] }>("/api/exercises") });
  const [q, setQ] = useState(""), [muscle, setMuscle] = useState<string | null>(null), [open, setOpen] = useState<string | null>(null), [name, setName] = useState("");
  const muscles = useMemo(() => [...new Set((data?.exercises ?? []).flatMap((e) => e.target_muscles))].sort(), [data]);
  const list = (data?.exercises ?? []).filter((e) => (!q || e.name.toLowerCase().includes(q.toLowerCase())) && (!muscle || e.target_muscles.includes(muscle)));
  const detail = useQuery({ queryKey: ["exercise", open], queryFn: () => api<Exercise>(`/api/exercises/${open}`), enabled: !!open });
  const check = useMutation({ mutationFn: () => api<Resolve>("/api/exercises/resolve", { body: { name } }), onError: toastError });
  return (
    <Shell>
      <h1 className="font-display text-3xl font-extrabold">Library</h1>
      <p className="mt-1 text-mute">Every exercise lists where the camera goes - counting is only as good as the view.</p>
      {error && <div className="mt-4"><ErrorState what="the exercise library" onRetry={() => refetch()} /></div>}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Input aria-label="Search exercises" placeholder="Search…" className="max-w-xs" value={q} onChange={(e) => setQ(e.target.value)} />
        {["All", ...muscles].map((m) => <button key={m} aria-pressed={(m === "All" && !muscle) || muscle === m} onClick={() => setMuscle(m === "All" ? null : m)}
          className={cn("rounded-full border px-3 py-1 text-sm font-semibold", (m === "All" && !muscle) || muscle === m ? "border-ink bg-ink text-bg" : "border-line text-mute hover:text-ink")}>{titleCase(m)}</button>)}
      </div>
      <div className="mt-4 grid items-start gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {isLoading && [0, 1, 2].map((i) => <Skeleton key={i} className="h-28" />)}
        {list.map((e) => (
          <Card key={e.id} className="text-left"><button className="w-full text-left" onClick={() => setOpen(open === e.id ? null : e.id)} aria-expanded={open === e.id}>
            <h3 className="text-lg font-bold">{e.name}</h3><p className="text-sm text-mute">{e.target_muscles.map(titleCase).join(", ")}</p>
            <CameraDiagram className="mt-3" view={e.camera.preferred} fullBody={e.camera.full_body} />
            <div className="mt-3 flex flex-wrap gap-1.5">{e.equipment.map((x) => <Badge key={x}>{x}</Badge>)}<Badge tone="brand">Counts reps</Badge></div></button>
            {open === e.id && detail.data && <div className="mt-3 border-t border-line pt-3 text-sm"><p className="font-semibold">What we check</p><ul className="mt-1 list-disc pl-5 text-ink-2">{detail.data.form_checks?.map((c) => <li key={c}>{c}</li>)}</ul>
              {!!detail.data.view_limited_checks?.length && <p className="mt-2 text-mute">{detail.data.view_limited_checks.map((v) => `${v.check} (only from ${v.views.join("/")})`).join("; ")}</p>}
              {e.camera.notes && <p className="mt-2 text-ink-2">{e.camera.notes}</p>}<p className="mt-2 text-mute">Camera views that work: {e.camera.supported_views.map((v) => v === "45deg" ? "45°" : v).join(", ")}</p>
              {!!detail.data.limitations?.length && <><p className="mt-3 font-semibold">What we can&apos;t tell</p><ul className="mt-1 list-disc pl-5 text-mute">{detail.data.limitations.map((l) => <li key={l}>{l}</li>)}</ul></>}</div>}
          </Card>))}
        {!isLoading && list.length === 0 && <p className="text-mute">Nothing matches.</p>}
      </div>
      <Card className="mt-8"><h2 className="text-lg font-bold">Don&apos;t see yours?</h2>
        <p className="text-sm text-mute">Check whether AI analysis is available. If not, you can still track it manually — and we log the request so it can be added.</p>
        <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (name.length > 1) check.mutate(); }}>
          <Input aria-label="Exercise name" placeholder="e.g. Cable Y Raise" className="max-w-xs" value={name} onChange={(e) => setName(e.target.value)} /><Button type="submit">Check</Button></form>
        {check.data && <div className="mt-3">{check.data.supported ? <p><Badge tone="good">Supported</Badge> matched <b>{check.data.exercise?.name}</b></p>
          : <Notice><b>Exercise not fully supported yet.</b><p className="mt-1">{check.data.message}</p>{check.data.similar?.length ? <p className="mt-1">Similar: {check.data.similar.map((s) => s.name).join(", ")}</p> : null}</Notice>}</div>}
      </Card>
    </Shell>);
}

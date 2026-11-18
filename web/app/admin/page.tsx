"use client";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, useAuth } from "@/lib/api";
import type { ModelVersion } from "@/lib/types";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge, Card, Notice, Select, Stat, Textarea, Input, Label } from "@/components/ui/primitives";
import { toast, toastError } from "@/components/toast";

interface Req { id: number; name: string; requests: number; distinct_users: number; candidate_pattern: string | null; closest_supported: string[]; status: string }
const STATUSES = ["new", "triaged", "planned", "collecting", "shipped", "declined"];

export default function Admin() {
  const qc = useQueryClient(), user = useAuth((s) => s.user);
  const ok = !!user?.is_admin;
  const ov = useQuery({ queryKey: ["a-ov"], queryFn: () => api<any>("/api/admin/overview"), enabled: ok });
  const rq = useQuery({ queryKey: ["a-rq"], queryFn: () => api<Req[]>("/api/admin/requests"), enabled: ok });
  const ms = useQuery({ queryKey: ["a-ms"], queryFn: () => api<ModelVersion[]>("/api/admin/models"), enabled: ok });
  const ds = useQuery({ queryKey: ["a-ds"], queryFn: () => api<any>("/api/admin/dataset/report"), enabled: ok, retry: false });
  const [blocked, setBlocked] = useState<string | null>(null), [json, setJson] = useState(""), [why, setWhy] = useState<any>(null), [res, setRes] = useState("");
  const setStatus = useMutation({ mutationFn: (v: { id: number; status: string }) => api(`/api/admin/requests/${v.id}`, { method: "PATCH", body: { status: v.status } }), onSuccess: () => toast("Updated"), onError: toastError });
  const promote = useMutation({ mutationFn: (id: number) => api(`/api/admin/models/${id}/promote`, { body: {} }), onSuccess: () => { setBlocked(null); toast("Promoted"); qc.invalidateQueries({ queryKey: ["a-ms"] }); }, onError: (e) => setBlocked(e instanceof Error ? e.message : String(e)) });
  const parse = () => { try { return JSON.parse(json); } catch { toast("Not valid JSON"); return null; } };
  const validate = useMutation({ mutationFn: () => api<any>("/api/admin/exercises/validate", { body: { definition: parse() } }), onSuccess: (r) => setRes(r.valid ? `Valid. ${r.notes.join(" ")}` : r.errors.join("\n")), onError: toastError });
  const add = useMutation({ mutationFn: () => api<any>("/api/admin/exercises", { body: { definition: parse() } }), onSuccess: (r) => { setRes(`Added ${r.id}`); qc.invalidateQueries({ queryKey: ["exercises"] }); }, onError: toastError });
  const ingest = useMutation({ mutationFn: (f: FormData) => api<any>("/api/admin/dataset/ingest", { form: f }), onSuccess: (r) => { toast(`Ingest started (job ${r.job_id})`); setTimeout(() => qc.invalidateQueries({ queryKey: ["a-ds"] }), 3000); }, onError: toastError });
  if (!ok) return <Shell><Notice tone="bad">Console is for administrators.</Notice></Shell>;
  const m = ov.data;
  return (
    <Shell>
      <h1 className="font-display text-3xl font-extrabold">Console</h1>
      {m && <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3">{[["users", m.users], ["sessions", m.sessions], ["AI sets", m.ai_sets], ["supported exercises", m.supported_exercises], ["open requests", m.open_requests], ["unsupported rate", `${Math.round(m.metrics.unsupported_exercise_rate * 100)}%`]].map(([k, v]) => <Stat key={String(k)} value={v as any} label={String(k)} />)}</div>}
      <Card className="mt-4 overflow-x-auto"><h2 className="text-lg font-bold">Most-requested unsupported exercises</h2>
        <table className="mt-2 w-full text-sm"><thead><tr className="text-left text-mute"><th>Exercise</th><th>Requests</th><th>Users</th><th>Pattern</th><th>Closest</th><th>Status</th></tr></thead><tbody>
          {rq.data?.map((r) => <tr key={r.id} className="border-t border-line"><td className="py-2">{r.name}</td><td>{r.requests}</td><td>{r.distinct_users}</td><td>{r.candidate_pattern ?? "?"}</td><td className="text-mute">{r.closest_supported.join(", ")}</td>
            <td><Select aria-label={`Status for ${r.name}`} className="w-32" defaultValue={r.status} onChange={(e) => setStatus.mutate({ id: r.id, status: e.target.value })}>{STATUSES.map((s) => <option key={s}>{s}</option>)}</Select></td></tr>)}
          {rq.data?.length === 0 && <tr><td colSpan={6} className="py-2 text-mute">No requests yet.</td></tr>}</tbody></table></Card>
      <Card className="mt-4 overflow-x-auto"><h2 className="text-lg font-bold">Model registry</h2>
        <table className="mt-2 w-full text-sm"><tbody>{ms.data?.map((v) => <tr key={v.id} className="border-t border-line align-top"><td className="py-2"><b>{v.name}</b> v{v.version}<div className="text-xs text-mute">{v.kind} · {v.dataset_version}</div></td>
          <td><Badge tone={v.status === "production" ? "good" : "neutral"}>{v.status}</Badge></td>
          <td className="text-xs text-mute">{Object.entries(v.metrics).slice(0, 6).map(([k, x]) => <div key={k}>{k}: {typeof x === "number" ? +x.toFixed(3) : String(x)}</div>)}<div className="mt-1">{v.limitations}</div></td>
          <td className="space-y-1"><Button size="sm" onClick={async () => setWhy(await api(`/api/admin/models/${v.id}/why`))}>Why?</Button>{v.status === "candidate" && <Button size="sm" onClick={() => promote.mutate(v.id)}>Promote</Button>}</td></tr>)}</tbody></table>
        {blocked && <div className="mt-3"><Notice tone="bad"><b>Promotion blocked:</b> {blocked}</Notice></div>}
        {why && <div className="mt-3"><Notice tone={why.would_pass_gate_now ? "good" : "warn"}><b>{why.model.name} v{why.model.version}</b> — {why.reproducible ? "reproducible" : "not reproducible (missing lineage)"}; {why.would_pass_gate_now ? "passes the gate" : `blocked by: ${why.blocked_by.join("; ")}`}</Notice></div>}</Card>
      <Card className="mt-4"><h2 className="text-lg font-bold">Dataset</h2>
        {ds.data ? <div className="mt-1 text-sm"><p>{ds.data.clips} clips · {ds.data.subjects} subjects · trusted labels {Math.round(ds.data.trusted_fraction * 100)}% · <code>{ds.data.version}</code></p>
          {ds.data.warnings.map((w: string) => <p key={w} className="text-warn">⚠ {w}</p>)}</div> : <p className="mt-1 text-sm text-mute">No dataset yet.</p>}
        <form className="mt-3 flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); ingest.mutate(new FormData(e.currentTarget)); }}>
          <div><Label htmlFor="file">Video</Label><input id="file" name="file" type="file" accept="video/*" required className="text-sm" /></div>
          <div><Label htmlFor="subject_id">Subject ID</Label><Input id="subject_id" name="subject_id" required className="w-32" /></div>
          <div><Label htmlFor="exercise_id">Exercise (optional)</Label><Input id="exercise_id" name="exercise_id" className="w-40" /></div>
          <div><Label htmlFor="view">View</Label><Select id="view" name="view" className="w-28"><option>unknown</option><option>front</option><option>side</option><option>45deg</option></Select></div>
          <Button type="submit" variant="primary" disabled={ingest.isPending}>Ingest</Button></form>
        <p className="mt-2 text-xs text-mute">Pose is extracted server-side and the raw video is deleted; only landmarks are kept.</p></Card>
      <Card className="mt-4"><h2 className="text-lg font-bold">Add an exercise (definition JSON)</h2><p className="text-sm text-mute">Exercises are data. Validate, then add — live immediately, no deploy.</p>
        <Textarea rows={9} spellCheck={false} aria-label="Exercise definition JSON" className="mt-2" value={json} onChange={(e) => setJson(e.target.value)} />
        <div className="mt-2 flex gap-2"><Button onClick={() => validate.mutate()}>Validate</Button><Button variant="primary" onClick={() => add.mutate()}>Add exercise</Button></div><pre className="mt-2 whitespace-pre-wrap text-sm">{res}</pre></Card>
    </Shell>);
}

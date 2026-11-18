"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useFieldArray, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { Exercise, Workout } from "@/lib/types";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge, Card, FieldError, Input, Label } from "@/components/ui/primitives";
import { toast, toastError } from "@/components/toast";

const schema = z.object({ name: z.string().min(1, "Name your workout").max(120),
  items: z.array(z.object({ exercise: z.string().min(2, "Pick an exercise"), sets: z.coerce.number().int().min(1).max(20), reps: z.coerce.number().int().min(1).max(200), weight_kg: z.coerce.number().min(0).max(1000), rest_seconds: z.coerce.number().int().min(0).max(900) })).min(1) });
type F = z.infer<typeof schema>;
const EMPTY: F = { name: "", items: [{ exercise: "", sets: 3, reps: 10, weight_kg: 0, rest_seconds: 60 }] };

export default function WorkoutPage() {
  const qc = useQueryClient(), router = useRouter();
  const ex = useQuery({ queryKey: ["exercises"], queryFn: () => api<{ exercises: Exercise[] }>("/api/exercises") });
  const ws = useQuery({ queryKey: ["workouts"], queryFn: () => api<Workout[]>("/api/workouts") });
  const { register, control, handleSubmit, reset, watch, formState: { errors } } = useForm<F>({ resolver: zodResolver(schema) as any, defaultValues: EMPTY });
  const [editing, setEditing] = useState<number | null>(null);
  const { fields, append, remove } = useFieldArray({ control, name: "items" });
  const names = new Set((ex.data?.exercises ?? []).map((e) => e.name.toLowerCase()));
  const save = useMutation({
    mutationFn: async (v: F) => { const w = await api<Workout>(editing ? `/api/workouts/${editing}` : "/api/workouts", { method: editing ? "PUT" : "POST", body: v }); for (const i of w.items.filter((i) => !i.ai_analysis)) await api("/api/exercises/resolve", { body: { name: i.exercise_name } }); return w; },
    onSuccess: (w) => { qc.invalidateQueries({ queryKey: ["workouts"] }); reset(EMPTY); setEditing(null); toast(w.unsupported_count ? `Saved. ${w.unsupported_count} exercise(s) will be tracked manually.` : "Workout saved."); }, onError: toastError });
  const edit = (w: Workout) => {
    setEditing(w.id);
    reset({ name: w.name, items: w.items.map((i) => ({ exercise: i.exercise_name, sets: i.sets, reps: i.reps, weight_kg: i.weight_kg, rest_seconds: i.rest_seconds })) });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const del = useMutation({ mutationFn: (id: number) => api(`/api/workouts/${id}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["workouts"] }), onError: toastError });
  return (
    <Shell>
      <h1 className="font-display text-3xl font-extrabold">Workouts</h1>
      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <div className="space-y-4">
          <Button onClick={() => router.push("/live/?quick=1")}>Quick set (auto-detect)</Button>
          {ws.data?.map((w) => (
            <Card key={w.id}><div className="flex items-center justify-between"><h2 className="text-xl font-bold">{w.name}</h2><Button variant="primary" size="sm" onClick={() => router.push(`/live/?workout=${w.id}`)}>Start</Button></div>
              <table className="mt-2 w-full text-sm"><tbody>{w.items.map((i) => <tr key={i.id} className="border-t border-line"><td className="py-2">{i.exercise_name} {!i.ai_analysis && <Badge tone="warn">manual</Badge>}</td><td className="text-right text-mute">{i.sets}×{i.reps}{i.weight_kg ? ` · ${i.weight_kg}kg` : ""}</td></tr>)}</tbody></table>
              <div className="mt-2 flex gap-1"><Button variant="ghost" size="sm" onClick={() => edit(w)}>Edit</Button>
                <Button variant="ghost" size="sm" onClick={() => confirm("Delete this workout?") && del.mutate(w.id)}>Delete</Button></div></Card>))}
          {ws.data?.length === 0 && <p className="text-mute">No workouts yet.</p>}
        </div>
        <Card><h2 className="text-xl font-bold">{editing ? "Edit workout" : "New workout"}</h2>
          <form onSubmit={handleSubmit((v) => save.mutate(v))} noValidate>
            <Label htmlFor="wn">Name</Label><Input id="wn" placeholder="Push day" {...register("name")} /><FieldError msg={errors.name?.message} />
            <datalist id="exlist">{ex.data?.exercises.map((e) => <option key={e.id} value={e.name} />)}</datalist>
            <div className="mt-3 space-y-2">{fields.map((f, idx) => {
              const typed = watch(`items.${idx}.exercise`);
              return (<div key={f.id} className="flex flex-wrap items-center gap-2">
                <Input list="exlist" aria-label={`Exercise ${idx + 1}`} placeholder="Exercise" className="min-w-40 flex-[2]" {...register(`items.${idx}.exercise`)} />
                <Input type="number" aria-label="Sets" className="w-16" {...register(`items.${idx}.sets`)} /><Input type="number" aria-label="Reps" className="w-16" {...register(`items.${idx}.reps`)} />
                <Input type="number" step="0.5" aria-label="Weight kg" className="w-20" {...register(`items.${idx}.weight_kg`)} />
                <Input type="number" step="15" aria-label="Rest seconds" title="Rest between sets (seconds)" className="w-20" {...register(`items.${idx}.rest_seconds`)} />
                {typed && !names.has(typed.toLowerCase()) && <Badge tone="warn">manual</Badge>}
                <Button type="button" variant="ghost" size="sm" aria-label="Remove exercise" onClick={() => fields.length > 1 && remove(idx)}><Trash2 className="h-4 w-4" /></Button></div>);
            })}</div>
            <FieldError msg={errors.items?.root?.message ?? errors.items?.[0]?.exercise?.message} />
            <div className="mt-4 flex gap-2"><Button type="button" onClick={() => append({ ...EMPTY.items[0] })}><Plus className="h-4 w-4" />Add exercise</Button>{editing && <Button type="button" variant="ghost" className="ml-auto" onClick={() => { reset(EMPTY); setEditing(null); }}>Cancel</Button>}
              <Button variant="primary" type="submit" disabled={save.isPending} className={editing ? "" : "ml-auto"}>{editing ? "Save changes" : "Save workout"}</Button></div>
          </form></Card>
      </div>
    </Shell>);
}

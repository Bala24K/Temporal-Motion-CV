"use client";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { DaySummary, Targets } from "@/lib/types";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Card, FieldError, Input, Label, Meter, Notice, Select } from "@/components/ui/primitives";
import { toast, toastError } from "@/components/toast";
import { fmt } from "@/lib/utils";

const prof = z.object({ sex: z.enum(["male", "female", "unspecified"]), birth_year: z.coerce.number().int().min(1920).max(new Date().getFullYear() - 14), height_cm: z.coerce.number().min(100).max(250),
  weight_kg: z.coerce.number().min(30).max(400), activity_level: z.enum(["sedentary", "light", "moderate", "active", "very_active"]) });
type P = z.infer<typeof prof>;
const today = () => new Date().toISOString().slice(0, 10);

export default function Nutrition() {
  const qc = useQueryClient();
  const [day, setDay] = useState(today()), [food, setFood] = useState(""), [grams, setGrams] = useState(100), [custom, setCustom] = useState({ name: "", kcal: 0, protein_g: 0 }), [mode, setMode] = useState<"db" | "custom">("db");
  const foods = useQuery({ queryKey: ["foods", food], queryFn: () => api<{ key: string; name: string; kcal: number }[]>(`/api/nutrition/foods?q=${encodeURIComponent(food)}`) });
  const d = useQuery({ queryKey: ["day", day], queryFn: () => api<DaySummary>(`/api/nutrition/day?day=${day}`) });
  const tg = useQuery({ queryKey: ["targets"], queryFn: () => api<Targets>("/api/nutrition/targets"), retry: false });
  const profile = useQuery({ queryKey: ["nprofile"], queryFn: () => api<Partial<P>>("/api/nutrition/profile") });
  const { register, handleSubmit, reset, formState: { errors } } = useForm<P>({ resolver: zodResolver(prof) as any });
  useEffect(() => { if (profile.data) reset({ sex: (profile.data.sex as P["sex"]) ?? "unspecified", birth_year: profile.data.birth_year ?? undefined, height_cm: profile.data.height_cm ?? undefined, weight_kg: profile.data.weight_kg ?? undefined, activity_level: (profile.data.activity_level as P["activity_level"]) ?? "moderate" }); }, [profile.data, reset]);
  const save = useMutation({ mutationFn: (v: P) => api("/api/nutrition/profile", { method: "PATCH", body: v }), onSuccess: () => { toast("Saved"); qc.invalidateQueries(); }, onError: toastError });
  const add = useMutation({ mutationFn: () => api("/api/nutrition/log", { body: mode === "db" ? { day, food_key: foods.data?.find((f) => f.name === food || f.key === food)?.key, grams } : { day, name: custom.name, grams, kcal: custom.kcal, protein_g: custom.protein_g } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["day"] }); setFood(""); }, onError: toastError });
  const del = useMutation({ mutationFn: (id: number) => api(`/api/nutrition/log/${id}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["day"] }) });
  const t = d.data?.totals, T = tg.data, ex = d.data?.exercise_expenditure_estimate;
  const bar = (label: string, v: number, goal?: number, unit = "g") => (<div key={label}><div className="flex justify-between text-sm"><span>{label}</span><span className="text-mute">{fmt(v)}{goal ? ` / ${goal}` : ""} {unit}</span></div><Meter value={goal ? (v / goal) * 100 : 0} /></div>);
  return (
    <Shell>
      <h1 className="font-display text-3xl font-extrabold">Nutrition</h1>
      <p className="mt-1 text-sm text-mute">Estimates from standard formulas — general guidance, not medical advice.</p>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Card><h2 className="text-lg font-bold">Your day</h2><Input type="date" aria-label="Day" className="mt-2 w-44" value={day} onChange={(e) => setDay(e.target.value)} />
          <div className="mt-3 space-y-3">{t && (<><div className="font-display text-4xl font-extrabold">{fmt(t.kcal)}<span className="text-base font-medium text-mute"> kcal{d.data?.remaining_kcal != null ? ` · ${fmt(d.data.remaining_kcal)} left` : ""}</span></div>
            {bar("Protein", t.protein_g, T?.protein_g)}{bar("Carbs", t.carbs_g, T?.carbs_g)}{bar("Fat", t.fat_g, T?.fat_g)}</>)}</div>
          {ex && ex.kcal_mid > 0 && <div className="mt-4"><Notice tone="good">Training burned roughly <b>{ex.kcal_low}–{ex.kcal_high} kcal</b> (≈{ex.kcal_mid}){ex.weight_assumed ? ", assuming 70 kg — add your weight for a better estimate" : ""}. An estimate, not a measurement.</Notice></div>}
        </Card>
        <Card><h2 className="text-lg font-bold">Log food</h2>
          <div className="mt-2 flex gap-2"><Button size="sm" variant={mode === "db" ? "primary" : "outline"} onClick={() => setMode("db")}>Search</Button><Button size="sm" variant={mode === "custom" ? "primary" : "outline"} onClick={() => setMode("custom")}>Custom</Button></div>
          <form className="mt-3 space-y-2" onSubmit={(e) => { e.preventDefault(); add.mutate(); }}>
            {mode === "db" ? (<><Input list="foodlist" aria-label="Food" placeholder="Search foods…" value={food} onChange={(e) => setFood(e.target.value)} /><datalist id="foodlist">{foods.data?.map((f) => <option key={f.key} value={f.name}>{f.kcal} kcal/100g</option>)}</datalist></>)
              : (<><Input aria-label="Name" placeholder="Name" value={custom.name} onChange={(e) => setCustom({ ...custom, name: e.target.value })} /><div className="flex gap-2"><Input type="number" aria-label="kcal" placeholder="kcal" onChange={(e) => setCustom({ ...custom, kcal: +e.target.value })} /><Input type="number" aria-label="Protein g" placeholder="protein g" onChange={(e) => setCustom({ ...custom, protein_g: +e.target.value })} /></div></>)}
            <div className="flex items-center gap-2"><Input type="number" aria-label="Grams" className="w-28" value={grams} onChange={(e) => setGrams(+e.target.value)} /><span className="text-sm text-mute">g</span><Button type="submit" variant="primary" className="ml-auto" disabled={add.isPending}>Add</Button></div></form>
          <ul className="mt-3 divide-y divide-line text-sm">{d.data?.entries.map((e) => <li key={e.id} className="flex items-center justify-between py-2"><span>{e.name} <span className="text-mute">{e.grams}g</span></span><span className="flex items-center gap-2">{fmt(e.kcal)} kcal<Button variant="ghost" size="sm" aria-label={`Remove ${e.name}`} onClick={() => del.mutate(e.id)}><Trash2 className="h-4 w-4" /></Button></span></li>)}</ul></Card>
      </div>
      <Card className="mt-4"><h2 className="text-lg font-bold">Targets</h2>
        {T ? <p className="mt-1">Maintenance ≈ {T.maintenance_kcal} kcal · target <b>{T.kcal_range[0]}–{T.kcal_range[1]} kcal</b> · {T.protein_g}g protein / {T.carbs_g}g carbs / {T.fat_g}g fat <span className="text-mute">(BMR {T.bmr})</span></p> : <p className="mt-1 text-mute">Fill in your details to get targets.</p>}
        <form className="mt-3 grid gap-3 sm:grid-cols-3" onSubmit={handleSubmit((v) => save.mutate(v))} noValidate>
          <div><Label htmlFor="sex">Sex</Label><Select id="sex" {...register("sex")}><option value="male">Male</option><option value="female">Female</option><option value="unspecified">Prefer not to say</option></Select></div>
          <div><Label htmlFor="by">Birth year</Label><Input id="by" type="number" {...register("birth_year")} /><FieldError msg={errors.birth_year?.message} /></div>
          <div><Label htmlFor="ac">Activity</Label><Select id="ac" {...register("activity_level")}><option value="sedentary">Sedentary</option><option value="light">Light</option><option value="moderate">Moderate</option><option value="active">Active</option><option value="very_active">Very active</option></Select></div>
          <div><Label htmlFor="ht">Height (cm)</Label><Input id="ht" type="number" {...register("height_cm")} /><FieldError msg={errors.height_cm?.message} /></div>
          <div><Label htmlFor="wt">Weight (kg)</Label><Input id="wt" type="number" step="0.1" {...register("weight_kg")} /><FieldError msg={errors.weight_kg?.message} /></div>
          <div className="flex items-end"><Button variant="primary" type="submit">Save</Button></div></form></Card>
    </Shell>);
}

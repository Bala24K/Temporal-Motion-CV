"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api, useAuth } from "@/lib/api";
import type { GoalProgress, Progress, Workout } from "@/lib/types";
import { Shell } from "@/components/shell";
import { BarsChart } from "@/components/charts";
import { Button } from "@/components/ui/button";
import { Card, Skeleton, Stat } from "@/components/ui/primitives";
import { fmt } from "@/lib/utils";
import { ErrorState } from "@/components/ui/states";

export default function Home() {
  const router = useRouter();
  const user = useAuth((s) => s.user);
  const week = useQuery({ queryKey: ["progress", "week"], queryFn: () => api<Progress>("/api/progress?range=week") });
  const goal = useQuery({ queryKey: ["goal"], queryFn: () => api<GoalProgress>("/api/progress/goal") });
  const ws = useQuery({ queryKey: ["workouts"], queryFn: () => api<Workout[]>("/api/workouts") });
  const h = new Date().getHours(), greet = h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening", next = ws.data?.[0];
  return (
    <Shell>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-4xl font-black md:text-5xl">{greet}{user?.name ? `, ${user.name}` : ""}.</h1>
        {goal.data && <p className="mt-1 text-mute">{goal.data.headline}</p>}
        {(week.error || ws.error) && <div className="mt-4"><ErrorState what="your dashboard" onRetry={() => { week.refetch(); ws.refetch(); }} /></div>}
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <Card>{ws.isLoading ? <Skeleton className="h-40" /> : next ? (<>
            <h2 className="text-2xl font-bold">{next.name}</h2>
            <p className="text-mute">{next.items.length} exercises · {next.muscles.join(", ") || "manual tracking"}</p>
            <ul className="mt-3 space-y-1">{next.items.slice(0, 5).map((i) => <li key={i.id}>{i.exercise_name} <span className="text-mute">{i.sets}×{i.reps}{i.weight_kg ? ` @ ${i.weight_kg}kg` : ""}</span></li>)}</ul>
            <div className="mt-4 flex gap-2"><Button variant="primary" onClick={() => router.push(`/live/?workout=${next.id}`)}>Start workout</Button><Button variant="ghost" onClick={() => router.push("/live/?quick=1")}>Quick set</Button></div></>)
            : (<><h2 className="text-2xl font-bold">No workout yet</h2><p className="text-mute">Build one, or jump straight into a quick set with auto-detect.</p>
              <div className="mt-4 flex gap-2"><Link href="/workout/"><Button variant="primary">Build a workout</Button></Link><Button variant="ghost" onClick={() => router.push("/live/?quick=1")}>Quick set</Button></div></>)}
          </Card>
          <Card><h2 className="text-lg font-bold">This week</h2>
            {week.data ? (<><div className="mt-2 grid grid-cols-3 gap-3"><div><div className="font-display text-3xl font-extrabold">{week.data.workouts}<span className="text-base text-mute">/{week.data.weekly_target}</span></div><div className="text-sm text-mute">workouts</div></div>
              <div><div className="font-display text-3xl font-extrabold">{fmt(week.data.volume_kg)}</div><div className="text-sm text-mute">kg volume</div></div>
              <div><div className="font-display text-3xl font-extrabold">{fmt(week.data.avg_form_score)}</div><div className="text-sm text-mute">form</div></div></div>
              <div className="mt-3"><BarsChart data={week.data.series} x="date" y="volume_kg" label="Training volume by day this week" height={110} /></div></>) : <Skeleton className="mt-3 h-40" />}</Card>
        </div>
        {goal.data && goal.data.streak_weeks > 0 && <Stat className="mt-4" value={`${goal.data.streak_weeks} week${goal.data.streak_weeks > 1 ? "s" : ""}`} label={`streak of hitting ${goal.data.weekly_target} sessions`} />}
      </motion.div>
    </Shell>);
}

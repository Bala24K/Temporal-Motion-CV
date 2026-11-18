"use client";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const axis = { stroke: "var(--mute)", fontSize: 12 };
const tip = { contentStyle: { background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12, color: "var(--ink)" } };
const shortDate = (d: string) => d.slice(5);

export function BarsChart({ data, x, y, label, height = 180 }: { data: Record<string, any>[]; x: string; y: string; label: string; height?: number }) {
  return (
    <div role="img" aria-label={label} style={{ height }}>
      <ResponsiveContainer><BarChart data={data}><CartesianGrid vertical={false} stroke="var(--line)" />
        <XAxis dataKey={x} tickFormatter={shortDate} {...axis} tickLine={false} axisLine={false} /><YAxis width={40} {...axis} tickLine={false} axisLine={false} />
        <Tooltip {...tip} cursor={{ fill: "var(--raise)" }} /><Bar dataKey={y} fill="var(--brand)" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer>
    </div>);
}

export function TrendChart({ data, x, y, label, height = 200, domain }: { data: Record<string, any>[]; x: string; y: string; label: string; height?: number; domain?: [number | string, number | string] }) {
  return (
    <div role="img" aria-label={label} style={{ height }}>
      <ResponsiveContainer><LineChart data={data}><CartesianGrid vertical={false} stroke="var(--line)" />
        <XAxis dataKey={x} tickFormatter={shortDate} {...axis} tickLine={false} axisLine={false} /><YAxis width={44} domain={domain ?? ["auto", "auto"]} {...axis} tickLine={false} axisLine={false} />
        <Tooltip {...tip} /><Line type="monotone" dataKey={y} stroke="var(--brand)" strokeWidth={3} dot={{ r: 3 }} connectNulls /></LineChart></ResponsiveContainer>
    </div>);
}

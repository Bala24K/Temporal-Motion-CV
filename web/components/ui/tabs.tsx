"use client";
import { cn } from "@/lib/utils";

export function Tabs<T extends string>({ value, onChange, items, label }: { value: T; onChange: (v: T) => void; items: { id: T; label: string }[]; label: string }) {
  return (
    <div role="tablist" aria-label={label} className="inline-flex gap-1 rounded-full border border-line bg-surface p-1">
      {items.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          onKeyDown={(e) => { const i = items.findIndex((x) => x.id === value); if (e.key === "ArrowRight") onChange(items[(i + 1) % items.length].id); if (e.key === "ArrowLeft") onChange(items[(i + items.length - 1) % items.length].id); }}
          className={cn("rounded-full px-4 py-1.5 text-sm font-semibold transition-colors", value === t.id ? "bg-ink text-bg" : "text-mute hover:text-ink")}>{t.label}</button>
      ))}
    </div>);
}

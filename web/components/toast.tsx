"use client";
import { create } from "zustand";
import { AnimatePresence, motion } from "framer-motion";

interface T { id: number; msg: string; tone: "info" | "error" }
export const useToasts = create<{ items: T[]; push: (msg: string, tone?: T["tone"]) => void }>((set) => ({
  items: [],
  push: (msg, tone = "info") => {
    const id = Date.now() + Math.random();
    set((s) => ({ items: [...s.items, { id, msg, tone }] }));
    setTimeout(() => set((s) => ({ items: s.items.filter((t) => t.id !== id) })), 4000);
  },
}));
export const toast = (m: string) => useToasts.getState().push(m);
export const toastError = (e: unknown) => useToasts.getState().push(e instanceof Error ? e.message : String(e), "error");

export function Toaster() {
  const items = useToasts((s) => s.items);
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex flex-col items-center gap-2 px-4" aria-live="polite">
      <AnimatePresence>
        {items.map((t) => (
          <motion.div key={t.id} role="status" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className={`pointer-events-auto rounded-full px-5 py-2.5 text-sm font-semibold shadow-lg ${t.tone === "error" ? "bg-bad text-white" : "bg-ink text-bg"}`}>{t.msg}</motion.div>
        ))}
      </AnimatePresence>
    </div>);
}

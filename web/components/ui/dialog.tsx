"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Button } from "./button";

/** Accessible confirmation built on native <dialog>: focus moves into it, Escape cancels, focus returns on close.
 *  Replaces window.confirm(), which is unstyled, blocks the page and can't be themed or tested reliably. */
export function ConfirmDialog({ open, title, body, confirmLabel = "Confirm", danger = false, onConfirm, onCancel }:
  { open: boolean; title: string; body?: ReactNode; confirmLabel?: string; danger?: boolean; onConfirm: () => void; onCancel: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current; if (!d) return;
    if (open && !d.open) d.showModal(); else if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} onCancel={(e) => { e.preventDefault(); onCancel(); }} aria-labelledby="confirm-title"
      className="w-[min(92vw,26rem)] rounded-2xl border border-line bg-surface p-0 text-ink shadow-2xl">
      <div className="p-5">
        <h2 id="confirm-title" className="text-lg font-bold">{title}</h2>
        {body && <div className="mt-1 text-sm text-ink-2">{body}</div>}
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={onCancel} autoFocus>Cancel</Button>
          <Button variant={danger ? "danger" : "primary"} onClick={onConfirm}>{confirmLabel}</Button>
        </div>
      </div>
    </dialog>);
}

/** Hook form: const ask = useConfirm(); ... onClick={() => ask({title, onConfirm})} ... {ask.dialog} */
export function useConfirm() {
  const [cfg, setCfg] = useState<null | { title: string; body?: ReactNode; confirmLabel?: string; danger?: boolean; onConfirm: () => void }>(null);
  const ask = (c: NonNullable<typeof cfg>) => setCfg(c);
  ask.dialog = cfg ? <ConfirmDialog open title={cfg.title} body={cfg.body} confirmLabel={cfg.confirmLabel} danger={cfg.danger}
    onCancel={() => setCfg(null)} onConfirm={() => { const f = cfg.onConfirm; setCfg(null); f(); }} /> : null;
  return ask;
}

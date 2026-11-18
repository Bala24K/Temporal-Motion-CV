import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "./button";

/** A failed query must say so (the audit found pages showing a loading skeleton forever on error). */
export function ErrorState({ what, onRetry }: { what: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-2xl border border-bad/40 bg-raise p-4 text-sm">
      <AlertTriangle className="h-5 w-5 text-bad" aria-hidden />
      <span className="flex-1">Couldn&apos;t load {what}.</span>
      {onRetry && <Button size="sm" onClick={onRetry}>Retry</Button>}
    </div>);
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-dashed border-line p-6 text-center">
      <p className="font-display text-lg font-bold">{title}</p>
      {children && <div className="mt-1 text-sm text-mute">{children}</div>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>);
}

import { forwardRef, type HTMLAttributes, type InputHTMLAttributes, type LabelHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const Card = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("rounded-2xl border border-line bg-surface p-5", className)} {...p} />);

const field = "w-full rounded-xl border border-line bg-raise px-3 py-2.5 text-ink placeholder:text-mute";
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn(field, className)} {...p} />));
Input.displayName = "Input";
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(({ className, ...p }, ref) => (
  <select ref={ref} className={cn(field, className)} {...p} />));
Select.displayName = "Select";
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea ref={ref} className={cn(field, "font-mono text-xs", className)} {...p} />));
Textarea.displayName = "Textarea";
export const Label = ({ className, ...p }: LabelHTMLAttributes<HTMLLabelElement>) => (
  <label className={cn("mb-1 mt-3 block text-sm font-semibold", className)} {...p} />);

const tones = { neutral: "border-line text-mute", good: "border-good text-good", warn: "border-warn text-warn", bad: "border-bad text-bad", brand: "border-brand text-brand" };
export const Badge = ({ tone = "neutral", className, ...p }: HTMLAttributes<HTMLSpanElement> & { tone?: keyof typeof tones }) => (
  <span className={cn("inline-block rounded-full border bg-raise px-2.5 py-0.5 text-xs font-semibold", tones[tone], className)} {...p} />);

export const Meter = ({ value, className, label = "Progress" }: { value: number; className?: string; label?: string }) => (
  <div className={cn("h-2 overflow-hidden rounded-full bg-raise", className)} role="progressbar" aria-label={label} aria-valuenow={Math.round(value)} aria-valuemin={0} aria-valuemax={100}>
    <div className="h-full rounded-full bg-brand transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
  </div>);

export const Stat = ({ value, label, className }: { value: ReactNode; label: string; className?: string }) => (
  <Card className={className}><div className="font-display text-3xl font-extrabold tracking-tight">{value}</div><div className="text-sm text-mute">{label}</div></Card>);

export const Notice = ({ children, tone = "warn" }: { children: ReactNode; tone?: "warn" | "bad" | "good" }) => (
  <div role="note" className={cn("rounded-xl border-l-4 bg-raise px-4 py-3 text-sm", tone === "warn" ? "border-warn" : tone === "bad" ? "border-bad" : "border-good")}>{children}</div>);

export const FieldError = ({ msg }: { msg?: string }) => (msg ? <p role="alert" className="mt-1 text-sm text-bad">{msg}</p> : null);
export const Skeleton = ({ className }: { className?: string }) => <div aria-hidden className={cn("animate-pulse rounded-2xl bg-raise", className)} />;

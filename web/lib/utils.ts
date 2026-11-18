import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...i: ClassValue[]) => twMerge(clsx(i));
export const fmt = (n: number | null | undefined, d = 0) => (n == null ? "–" : n.toLocaleString(undefined, { maximumFractionDigits: d }));
export const titleCase = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
export const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

import { create } from "zustand";
import type { User } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
export class ApiError extends Error { constructor(msg: string, public status: number, public data?: unknown) { super(msg); } }

interface AuthState { token: string | null; user: User | null; ready: boolean; setSession: (t: string, u: User) => void; clear: () => void; hydrate: () => void }
export const useAuth = create<AuthState>((set) => ({
  token: null, user: null, ready: false,
  setSession: (token, user) => { localStorage.setItem("kinetic.token", token); set({ token, user, ready: true }); },
  clear: () => { localStorage.removeItem("kinetic.token"); set({ token: null, user: null, ready: true }); },
  hydrate: () => set({ token: localStorage.getItem("kinetic.token"), ready: true }),
}));

function detail(d: any, fallback: string): string {
  const x = d?.detail;
  if (typeof x === "string") return x;
  if (Array.isArray(x)) return x.map((e) => e.msg ?? JSON.stringify(e)).join("; ");
  if (x?.blocked_by) return x.blocked_by.join("; ");
  if (x?.errors) return x.errors.join("; ");
  return fallback;
}

export async function api<T = unknown>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const token = useAuth.getState().token;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(BASE + path, { method: opts.method ?? (opts.body || opts.form ? "POST" : "GET"), headers,
    body: opts.form ?? (opts.body !== undefined ? JSON.stringify(opts.body) : undefined) });
  if (r.status === 204) return null as T;
  let data: unknown = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    if (r.status === 401 && token) useAuth.getState().clear();
    throw new ApiError(detail(data, r.statusText), r.status, data);
  }
  return data as T;
}

export function wsUrl(path: string): string {
  const base = BASE || (typeof location !== "undefined" ? location.origin : "");
  return base.replace(/^http/, "ws") + path;
}

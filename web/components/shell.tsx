"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Apple, BookOpen, ClipboardList, History, LogOut, Moon, Shield, Sun, TrendingUp, User as UserIcon, type LucideIcon } from "lucide-react";
import { api, useAuth } from "@/lib/api";
import type { User } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Skeleton } from "./ui/primitives";

type NavItem = { href: string; label: string; icon: LucideIcon; mobile?: boolean };
/** Primary sections. Mobile shows the five marked `mobile` as a bottom tab bar (History lives under Progress there). */
const PRIMARY: NavItem[] = [
  { href: "/", label: "Train", icon: Activity, mobile: true },
  { href: "/workout/", label: "Workouts", icon: ClipboardList, mobile: true },
  { href: "/exercises/", label: "Library", icon: BookOpen, mobile: true },
  { href: "/history/", label: "History", icon: History },
  { href: "/progress/", label: "Progress", icon: TrendingUp, mobile: true },
  { href: "/profile/", label: "You", icon: UserIcon, mobile: true },
];
/** Secondary: outside the core training loop (nutrition is deferred; the console is for admins). */
const SECONDARY: NavItem[] = [{ href: "/nutrition/", label: "Nutrition", icon: Apple }];

const isOn = (href: string, path: string) => (href === "/" ? path === "/" : path.startsWith(href.slice(0, -1)));

export function Shell({ children }: { children: ReactNode }) {
  const { token, ready, user, setSession, clear } = useAuth();
  const router = useRouter();
  const path = usePathname();
  const [dark, setDark] = useState(true);
  useEffect(() => { setDark(document.documentElement.dataset.theme !== "light"); }, []);
  useEffect(() => { if (ready && !token) router.replace("/login/"); }, [ready, token, router]);
  const me = useQuery({ queryKey: ["me", token], queryFn: () => api<User>("/api/auth/me"), enabled: !!token });
  useEffect(() => { if (me.data && token) setSession(token, me.data); }, [me.data, token, setSession]);
  if (!ready || !token || !user) return <main className="mx-auto max-w-5xl p-6"><Skeleton className="h-10 w-48" /><Skeleton className="mt-6 h-64" /></main>;
  const secondary = user.is_admin ? [...SECONDARY, { href: "/admin/", label: "Console", icon: Shield }] : SECONDARY;
  const toggle = () => {
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next; setDark(!dark);
    try { localStorage.setItem("kinetic.theme", next); } catch { /* storage blocked */ }
  };
  const railLink = (n: NavItem) => {
    const on = isOn(n.href, path);
    return (
      <Link key={n.href} href={n.href} aria-current={on ? "page" : undefined}
        className={cn("flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition-colors",
          on ? "bg-brand-dim text-ink" : "text-mute hover:bg-raise hover:text-ink")}>
        <n.icon className={cn("h-[18px] w-[18px]", on && "text-brand")} aria-hidden />{n.label}</Link>);
  };
  return (
    <div className="min-h-screen md:grid md:grid-cols-[232px_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-brand focus:px-3 focus:py-2 focus:text-brand-ink">Skip to content</a>
      {/* desktop rail */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-line bg-surface px-3 py-5 md:flex">
        <span className="px-3 font-display text-2xl font-black tracking-tight" style={{ fontStretch: "120%" }}>KINETIC</span>
        <nav aria-label="Main" className="mt-6 flex flex-col gap-1">{PRIMARY.map(railLink)}</nav>
        <nav aria-label="More" className="mt-6 flex flex-col gap-1 border-t border-line pt-4">{secondary.map(railLink)}</nav>
        <div className="mt-auto flex items-center gap-1 px-1">
          <button aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggle} className="rounded-lg p-2 text-mute hover:bg-raise hover:text-ink">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
          <button onClick={() => { clear(); router.replace("/login/"); }} className="flex items-center gap-2 rounded-lg px-2 py-2 text-sm text-mute hover:bg-raise hover:text-ink"><LogOut className="h-4 w-4" aria-hidden />Sign out</button>
        </div>
      </aside>
      {/* mobile header */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-bg/90 px-4 py-3 backdrop-blur md:hidden">
        <span className="font-display text-xl font-black" style={{ fontStretch: "120%" }}>KINETIC</span>
        <button aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggle} className="rounded-lg p-2 text-mute">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
      </header>
      <main id="main" className="mx-auto w-full max-w-6xl px-4 pb-28 pt-6 md:px-8 md:pb-12 md:pt-8">{children}</main>
      {/* mobile bottom tabs */}
      <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-5 border-t border-line bg-surface/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden">
        {PRIMARY.filter((n) => n.mobile).map((n) => {
          const on = isOn(n.href, path) || (n.href === "/progress/" && path.startsWith("/history"));
          return (
            <Link key={n.href} href={n.href} aria-current={on ? "page" : undefined} className={cn("flex flex-col items-center gap-0.5 py-2 text-[11px] font-semibold", on ? "text-brand" : "text-mute")}>
              <n.icon className="h-5 w-5" aria-hidden />{n.label}</Link>);
        })}
      </nav>
    </div>);
}

"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { api, useAuth } from "@/lib/api";
import type { User } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Skeleton } from "./ui/primitives";

const LINKS = [["/", "Home"], ["/workout/", "Workout"], ["/exercises/", "Exercises"], ["/history/", "History"], ["/progress/", "Progress"], ["/nutrition/", "Nutrition"], ["/profile/", "Profile"]];

export function Shell({ children }: { children: ReactNode }) {
  const { token, ready, user, setSession, clear } = useAuth();
  const router = useRouter();
  const path = usePathname();
  useEffect(() => { if (ready && !token) router.replace("/login/"); }, [ready, token, router]);
  const [dark, setDark] = useState(false);
  useEffect(() => { const el = document.documentElement; setDark(el.dataset.theme ? el.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches); }, []);
  const me = useQuery({ queryKey: ["me", token], queryFn: () => api<User>("/api/auth/me"), enabled: !!token });
  useEffect(() => { if (me.data && token) setSession(token, me.data); }, [me.data, token, setSession]);
  if (!ready || !token || !user) return <main className="mx-auto max-w-5xl p-6"><Skeleton className="h-10 w-48" /><Skeleton className="mt-6 h-64" /></main>;
  const links = user.is_admin ? [...LINKS, ["/admin/", "Console"]] : LINKS;
  const toggle = () => { const el = document.documentElement; const dark = el.dataset.theme ? el.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; el.dataset.theme = dark ? "light" : "dark"; setDark(!dark); localStorage.setItem("kinetic.theme", el.dataset.theme); };
  return (
    <>
      <header className="sticky top-0 z-30 border-b border-line bg-bg/90 backdrop-blur">
        <nav aria-label="Main" className="mx-auto flex max-w-5xl items-center gap-1 overflow-x-auto px-4 py-2">
          <span className="mr-auto pr-4 font-display text-xl font-extrabold">Kinetic</span>
          {links.map(([href, label]) => {
            const on = href === "/" ? path === "/" : path.startsWith(href.slice(0, -1));
            return <Link key={href} href={href} aria-current={on ? "page" : undefined}
              className={cn("whitespace-nowrap rounded-full px-3.5 py-1.5 text-sm font-semibold", on ? "bg-ink text-bg" : "text-mute hover:text-ink")}>{label}</Link>;
          })}
          <button aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggle} className="ml-2 rounded-full p-2 text-mute hover:text-ink">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
          <button onClick={() => { clear(); router.replace("/login/"); }} className="rounded-full px-3 py-1.5 text-sm text-mute hover:text-ink">Sign out</button>
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 pb-24 pt-6">{children}</main>
    </>);
}

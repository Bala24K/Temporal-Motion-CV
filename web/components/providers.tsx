"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { useAuth } from "@/lib/api";
import { Toaster } from "./toast";

export function Providers({ children }: { children: ReactNode }) {
  const [qc] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 15_000, refetchOnWindowFocus: false } } }));
  const hydrate = useAuth((s) => s.hydrate);
  useEffect(() => { hydrate(); const t = localStorage.getItem("kinetic.theme"); if (t) document.documentElement.dataset.theme = t; }, [hydrate]);
  return <QueryClientProvider client={qc}>{children}<Toaster /></QueryClientProvider>;
}

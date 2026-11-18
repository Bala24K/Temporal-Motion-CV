"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { api, useAuth } from "@/lib/api";
import type { User } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, FieldError, Input, Label } from "@/components/ui/primitives";
import { toastError } from "@/components/toast";

const schema = z.object({ email: z.string().email("Enter a valid email"), password: z.string().min(8, "At least 8 characters") });
type F = z.infer<typeof schema>;

export default function Login() {
  const router = useRouter();
  const { token, ready, setSession } = useAuth();
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<F>({ resolver: zodResolver(schema) });
  const [mode, setMode] = useState<"login" | "register">("login");
  useEffect(() => { if (ready && token) router.replace("/"); }, [ready, token, router]);
  const submit = handleSubmit(async (v) => {
    try { const r = await api<{ token: string; user: User }>(`/api/auth/${mode}`, { method: "POST", body: v }); setSession(r.token, r.user); router.replace("/"); } catch (e) { toastError(e); }
  });
  return (
    <main className="mx-auto max-w-md px-4 pt-[10vh]">
      <h1 className="font-display text-4xl font-extrabold leading-tight">Train with a coach that watches.</h1>
      <p className="mt-2 text-mute">Counts your reps, checks your form, remembers your range. Your video never leaves your device.</p>
      <Card className="mt-6"><form onSubmit={submit} noValidate>
        <Label htmlFor="email">Email</Label><Input id="email" type="email" autoComplete="email" aria-invalid={!!errors.email} {...register("email")} /><FieldError msg={errors.email?.message} />
        <Label htmlFor="password">Password</Label><Input id="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} aria-invalid={!!errors.password} {...register("password")} /><FieldError msg={errors.password?.message} />
        <div className="mt-5 flex gap-2"><Button variant="primary" type="submit" disabled={isSubmitting}>{mode === "login" ? "Sign in" : "Create account"}</Button>
          <Button type="button" variant="ghost" onClick={() => setMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "New here? Create account" : "Have an account? Sign in"}</Button></div>
      </form></Card>
    </main>);
}

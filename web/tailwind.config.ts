import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./hooks/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)", surface: "var(--surface)", raise: "var(--raise)", "raise-2": "var(--raise-2)", ink: "var(--ink)",
        "ink-2": "var(--ink-2)", mute: "var(--mute)", line: "var(--line)", brand: "var(--brand)", "brand-ink": "var(--brand-ink)",
        "brand-dim": "var(--brand-dim)", good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)", info: "var(--info)",
      },
      fontFamily: {
        display: ["Archivo", "Inter", "system-ui", "sans-serif"],
        sans: ["Inter", "system-ui", "Segoe UI", "sans-serif"],
      },
      borderRadius: { xl2: "1.25rem" },
    },
  },
  plugins: [],
};
export default config;

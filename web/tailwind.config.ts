import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./hooks/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)", surface: "var(--surface)", raise: "var(--raise)", ink: "var(--ink)", mute: "var(--mute)",
        line: "var(--line)", brand: "var(--brand)", "brand-ink": "var(--brand-ink)",
        good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)",
      },
      fontFamily: {
        display: ["Bricolage Grotesque", "Avenir Next", "Segoe UI", "system-ui", "sans-serif"],
        sans: ["Figtree", "Segoe UI", "system-ui", "sans-serif"],
      },
      borderRadius: { xl2: "1.25rem" },
    },
  },
  plugins: [],
};
export default config;

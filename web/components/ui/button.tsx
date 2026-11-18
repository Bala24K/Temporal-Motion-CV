import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-colors disabled:opacity-50 disabled:pointer-events-none whitespace-nowrap",
  { variants: {
      variant: { primary: "bg-brand text-brand-ink hover:opacity-90", outline: "border border-line bg-surface hover:bg-raise",
                 ghost: "text-mute hover:bg-raise hover:text-ink", danger: "border border-bad text-bad hover:bg-raise" },
      size: { sm: "px-3 py-1.5 text-sm", md: "px-5 py-2.5", lg: "px-7 py-3.5 text-lg" } },
    defaultVariants: { variant: "outline", size: "md" } });

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof button> {}
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...p }, ref) => (
  <button ref={ref} className={cn(button({ variant, size }), className)} {...p} />));
Button.displayName = "Button";

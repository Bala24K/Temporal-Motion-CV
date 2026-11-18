/** Top-down camera placement: the person (head + shoulders) in the middle, the camera where it should stand. */
const POS: Record<string, { x: number; y: number; label: string }> = {
  front: { x: 60, y: 74, label: "In front" },
  side: { x: 112, y: 40, label: "To your side" },
  "45deg": { x: 100, y: 70, label: "45° to the front" },
};

export function CameraDiagram({ view, fullBody, className = "" }: { view: string; fullBody?: boolean; className?: string }) {
  const cam = POS[view] ?? POS.front;
  const angle = Math.atan2(40 - cam.y, 60 - cam.x) * (180 / Math.PI);
  return (
    <figure className={`flex items-center gap-3 ${className}`}>
      <svg viewBox="0 0 130 86" className="h-16 w-24 shrink-0" role="img" aria-label={`Camera ${cam.label.toLowerCase()}${fullBody ? ", whole body in view" : ""}`}>
        <line x1={cam.x} y1={cam.y} x2="60" y2="40" stroke="var(--brand)" strokeWidth="1.5" strokeDasharray="3 3" />
        <ellipse cx="60" cy="40" rx="17" ry="7" fill="var(--raise-2)" stroke="var(--mute)" />
        <circle cx="60" cy="40" r="6" fill="var(--ink-2)" />
        <g transform={`translate(${cam.x} ${cam.y}) rotate(${angle})`}>
          <rect x="-7" y="-5" width="12" height="10" rx="2" fill="var(--brand)" />
          <path d="M5 -3 L10 -6 L10 6 L5 3 Z" fill="var(--brand)" />
        </g>
      </svg>
      <figcaption className="text-sm"><span className="font-semibold">{cam.label}</span>{fullBody ? <span className="block text-mute">Whole body in view</span> : <span className="block text-mute">Waist up is enough</span>}</figcaption>
    </figure>);
}

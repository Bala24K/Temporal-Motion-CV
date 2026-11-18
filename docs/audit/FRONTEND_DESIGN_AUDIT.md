# Frontend and Design Audit

Scope: `RT/web` (Next.js 15 static export, the current product UI). V1's OpenCV overlay and RT's legacy SPA (`RT/frontend/index.html`, served at `/legacy`) are superseded and not covered in detail. Aliases and evidence tags are defined in `PROJECT_AUDIT.md`. Every finding below is [Inspected] unless marked otherwise. The UI was not run in a browser during this audit.

---

## 1. Inventory

| Route / component | Purpose | Data | Notes |
| --- | --- | --- | --- |
| `/login` | Login/register | `/api/auth/{login,register}` | Clean form (RHF + Zod). Marketing headline. |
| `/` Home | Greeting, next workout, week stats, streak | `/api/progress`, `/api/progress/goal`, `/api/workouts` | "Next workout" is just the newest workout (`ws.data?.[0]`) |
| `/workout` | List + builder | `/api/workouts`, `/api/exercises`, `/api/exercises/resolve` | No edit, no rest field, no reorder |
| `/live` | Session runner → `LiveWorkout` overlay | `/api/sessions`, WebSocket | Renders a `Shell` + skeleton *behind* a fixed full-screen dialog |
| `/exercises` | Library grid, muscle filter, detail accordion, "Don't see yours?" | `/api/exercises` | No imagery, no demo, no camera-setup illustration |
| `/progress` | Range tabs, 6 stat cards, 2 bar charts, muscle split, PBs, per-exercise trends, goal bars | `/api/progress*` | Dense; AI metrics shown without caveat |
| `/nutrition` | Profile form, targets, food log | `/api/nutrition/*` | Out of the core product's scope |
| `/profile` | Goal, 3D body model, calibration wizard, baselines, privacy/delete | `/api/profile`, `/api/auth/me` | Calibration is buried at the bottom of Profile |
| `/admin` | Requests, models, dataset | `/api/admin/*` | Shown in the main nav for admins |
| `components/shell.tsx` | Sticky top nav, theme toggle, sign-out | `/api/auth/me` | 6–7 pill links in a horizontally scrolling bar |
| `components/live-workout.tsx` | Set screen, rest screen, progress ring | WebSocket | Hard-coded exercise list; duplicated selects |
| `components/calibration.tsx` | 3-stage wizard | `/api/profile/calibrate/*` | 4:3 box, text instruction, progress meter |
| `components/skeleton-3d.tsx` | R3F T-pose viewer | — | Static figure; spins |
| `components/charts.tsx` | Recharts bar/line wrappers | — | Basic |
| `components/ui/*` | Button (cva), Card, Input, Select, Badge, Meter, Stat, Notice, Skeleton, Tabs, Toast | — | Small, consistent, reusable |

---

## 2. Assessment

| Area | Finding | Severity |
| --- | --- | --- |
| **Visual hierarchy** | Every page is a stack of identical white rounded cards with bold headings. Nothing is primary. On Home, "Start workout" carries the same weight as "Quick set" and the stats card. | High |
| **Typography** | Bricolage Grotesque (display) + Figtree (body) via Google Fonts is a decent pairing. No defined type scale (sizes are ad hoc: `text-4xl`, `text-3xl`, `text-xl`, `text-lg`). Numbers aren't tabular (`tabular-nums` absent), so stat and rep digits shift width as they change. | Medium |
| **Spacing** | Tailwind defaults used ad hoc (`mt-2/3/4/6/8`). No vertical rhythm or section spacing tokens. | Medium |
| **Layout consistency** | `max-w-5xl` everywhere, including dashboards that could use the width. The live screen uses hard-coded colours (`#0a1016`, `#15202b`, `#93a3b2`, `#6f92ff`) that bypass the tokens. | Medium |
| **Responsive behaviour** | Mobile: 7-item pill nav scrolls horizontally (hidden items, no bottom bar). The live footer packs up to 7 controls (two exercise selects, source, fault, Start, kg, End set) that wrap into a tall block over the camera. Live rep count moves to `top-[26%]` on mobile, over the user's head and torso. | High |
| **Component reuse** | Good primitives. But exercise-selection logic is duplicated (header and footer selects with identical handlers, `live-workout.tsx:92-105`, `139-151`), and instructions are duplicated between client and server. | Medium |
| **Loading states** | Skeletons exist. Several pages show a skeleton **forever** on error (e.g. `progress/page.tsx`: `!d ? <Skeleton/>`; Home stats). | High |
| **Error states** | Mutation errors → toast. Query errors are mostly ignored. The WebSocket close isn't surfaced (P9). Camera permission denial is handled with a helpful message (good). | High |
| **Navigation** | Flat nav of equal peers (Home, Workout, Exercises, Progress, Nutrition, Profile, Console). The most important action (start training) has no persistent entry point. Calibration lives under Profile. | High |
| **Accessibility** | Positives: `aria-label`s, `aria-live` on feedback, focus-visible ring, reduced-motion CSS, `role="img"` labels on charts. Gaps: the live dialog has no focus trap or `aria-modal`; native `confirm()` for destructive actions; **the rep count isn't announced** (no live region on the counter, no audio cue); the rest countdown has `aria-live="off"`; charts have no table alternative; theme icon state is broken (`dark:` uses the media strategy, not `data-theme`). | High |
| **Live camera layout** | Full-screen dark overlay with a mirrored video at 80% opacity, rep count on the left, progress ring bottom-right, feedback chips bottom-left. The concept is good. Problems: no pre-set setup/readiness phase; the quality warning is one pill; no indication of which joints are weak; the `confirm` state is a message with no action; nothing tells the user what the current rep needs ("go deeper"). | High |
| **Charts** | Recharts defaults with brand colour. Bars by day; e1RM/ROM lines. No targets, annotations, comparison periods or per-rep views. ROM is shown as a percentage of the *textbook* range. | Medium |
| **Skeleton visualization** | 4 px blue lines plus white dots, a single colour, threshold vis > 0.3, no smoothing, nose floating, no midline, no emphasis on the joints the exercise uses, no rule-violation highlighting, no angle readouts. The 3D viewer shows a static T-pose of proportions, not the live skeleton. | High |
| **API integration** | Typed `api()` wrapper with error detail parsing; TanStack Query keys. The WebSocket protocol is untyped JSON with no schema shared with the server. The token travels in the URL. | Medium |
| **State synchronization** | Exercise switch: the UI optimistically resets reps to 0 and shows "Switched to…" only *if* the server responds, but the server may never have received the message (latest-wins race). `rec` is cleared on every `state` message. Profile `calibrated` (UI) and engine `calibrated` disagree (RC-6). | High |
| **Placeholder / mock data** | Hard-coded `SUPPORTED_EXERCISES`; demo clip and demo body selectable in production UI; `Auto-detect (experimental)` offered although it can't succeed for 2 of 7 groups. | Medium |
| **Buttons that don't do what they say** | "Quick set (auto-detect)" starts a bicep-curl set. The exercise selector during a running set may not switch (race), and when it does it discards counted reps. "Finish (n/3)" in calibration shows "Calibrated" even when the stored profile is not calibrated. | High |
| **Build hygiene** | `web/out` (2026-10-03) is committed and served by FastAPI, and is older than the source (2026-11-06). Users of `uvicorn backend.main:app` run old UI code. | High |

---

## 3. Design direction: "performance lab"

**Positioning:** a sports-performance instrument, not a dashboard. Dark-first, camera-first, numbers you can read from 3 m away, one confident accent, and every on-screen claim traceable to a measurement.

### 3.1 Principles

1. **Camera first in training mode.** During a set the UI is the user's body, the count and one cue.
2. **Readable at distance.** Live numerals at least 12 vh, tabular, high contrast. Cues of at most 4 words.
3. **Every judgement shows its evidence.** "Depth 82% of your range (target 85%)", not "Bad form".
4. **Confidence is visible.** Tracking quality, calibration quality and "not evaluated" are first-class states.
5. **Calm by default, kinetic on events.** Motion marks a rep completing, a PR or a warning, and nothing else.

### 3.2 Tokens

```css
:root {                                    /* dark is the default theme */
  /* surfaces */
  --bg: #07090c; --surface-1: #0e1217; --surface-2: #151b22; --surface-3: #1d252e; --line: #26303b;
  /* text */
  --ink: #f2f5f8; --ink-2: #b4bfca; --ink-3: #7c8896;
  /* accent: electric lime for "go" / reps; one accent only */
  --accent: #c6f432; --accent-ink: #0a0d05; --accent-dim: #c6f43233;
  /* semantic */
  --good: #34d399; --warn: #fbbf24; --bad: #f87171; --info: #60a5fa;
  /* data viz (ordered, colour-blind-safe series) */
  --viz-1: #c6f432; --viz-2: #60a5fa; --viz-3: #f472b6; --viz-4: #fbbf24; --viz-grid: #1f2831;
  /* skeleton */
  --bone: #e6edf3; --bone-low: #e6edf366; --joint-active: var(--accent); --joint-warn: var(--warn); --joint-lost: #64748b;
  /* radius / elevation */
  --r-sm: 8px; --r-md: 14px; --r-lg: 22px; --r-pill: 999px;
  --shadow-1: 0 1px 0 #ffffff0d inset, 0 8px 24px #00000066;
  /* motion */
  --ease-out: cubic-bezier(.2,.8,.2,1); --dur-1: 120ms; --dur-2: 220ms; --dur-3: 420ms;
}
:root[data-theme="light"] {
  --bg: #f4f6f8; --surface-1: #ffffff; --surface-2: #f0f3f6; --surface-3: #e6ebf0; --line: #d6dde4;
  --ink: #0b1016; --ink-2: #3f4a56; --ink-3: #6b7785; --accent: #5b7f00; --accent-ink: #ffffff;
}
```
Spacing: a 4 px base (`4, 8, 12, 16, 24, 32, 48, 64, 96`). Contrast: check every text/surface pair against WCAG AA (4.5:1 body, 3:1 large). Lime needs dark text.

**Typography** (Google Fonts are already in use, so no new dependency category):
- Display / numerals: **Archivo** (variable width and weight). Condensed-bold for headlines, expanded-black for live counts, `font-variant-numeric: tabular-nums`.
- UI / body: **Inter** (or keep Figtree).
- Scale (rem): 0.75 / 0.875 / 1 / 1.125 / 1.375 / 1.75 / 2.25 / 3 / 4.5; live counter `clamp(6rem, 22vh, 16rem)`.

### 3.3 Component architecture

```
web/
  app/(train)/today | live | setup           ← training mode routes (full-bleed, dark)
  app/(app)/library | history | progress | body | settings
  features/
    live/        LiveScreen, RepCounter, PhaseGauge, CueStack, TrackingBadge, SetupChecklist, SetSummarySheet
    calibration/ SetupWizard (framing → neutral → range capture), QualityMeter, FailureHint
    skeleton/    SkeletonOverlay (canvas), skeletonStyle.ts (tokens → styles), useSmoothedPose
    library/     ExerciseCard, ExerciseDetail, ViewRequirement, MuscleMap (SVG)
    history/     SessionList, SessionDetail, RepTimeline
    progress/    KpiStrip, TrendCard, CompareToggle
  lib/
    protocol.ts  ← single typed WS protocol (zod), generated from server JSON Schema
    pose/        worker.ts (MediaPipe in a Web Worker), contract.ts (isotropic conversion + metadata)
    api.ts, queryKeys.ts
  components/ui/ Button, IconButton, Card, Sheet, Dialog (native <dialog> + focus management),
                 SegmentedControl, Stat, Badge, Meter, Toast, EmptyState, ErrorState
```
State: TanStack Query for server data; one Zustand store for the live session's state machine (`idle → setup → ready → active → resting → summary`). Throttle WS-driven re-renders to animation frames. Stream skeleton drawing outside React.

### 3.4 Navigation structure

- **Mobile:** bottom tab bar with 4 tabs (**Train**, **Library**, **Progress**, **You**) and a centre "Start" action. Training mode hides the tab bar.
- **Desktop:** left rail with the same four sections plus Settings. Admin console on a separate `/console` route, not in the user nav. Nutrition hidden behind a feature flag (deferred).
- **Calibration** is a first-class flow ("Body & camera") reached from You, and offered automatically when the setup fingerprint doesn't match.

### 3.5 Live-workout experience (built only on real engine outputs)

```
┌──────────────────────────────────────────────────────────────┐
│ ✕   Squat · side view        Set 2/3        ● Tracking good   │  ← TrackingBadge (quality.ok, weak joints)
│                                                              │
│                      [ camera + skeleton ]                    │  ← joints the exercise uses glow accent;
│   ┃                                                          │    failing-rule joints pulse warn
│   ┃ PhaseGauge: personal range                               │
│   ┃ ▲ rest ─ current ─ target depth line                     │  ← progress vs personal range (M3)
│   ┃                                                          │
│                                              7               │  ← RepCounter (tabular, huge), "of 10"
│  ⚠ Go a little deeper                       of 10            │  ← CueStack: max 1 warning + 1 positive
│                                                              │
│ [ Pause ]                              [ End set ]           │  ← two controls only during a set
└──────────────────────────────────────────────────────────────┘
```
- **Pre-set SetupChecklist** (from the calibration protocol): in frame ✓, distance ✓, view ✓ (side), light ✓, calibration ✓/stale. "Start" unlocks only when every item passes. Otherwise a specific hint with an arrow.
- **Range capture** (first time): "Do 2 slow reps through your full range". The gauge shows the captured rest and peak.
- **During the set:** rep counter, phase gauge, one cue. Audio "beep + number" per rep (optional). A partial rep is shown as a hollow tick.
- **Tracking states:** degraded → amber badge "Lost your knees"; lost → the set pauses automatically with "Step back into frame".
- **Exercise change** happens only between sets (a sheet). The current set is saved first.
- **Ambiguous groups:** a one-time chip picker ("Curl or hammer curl?"), remembered for the workout item.
- **SetSummarySheet:** count, partials, a per-rep bar strip (depth % of personal range, tempo), each failing rule with measured vs threshold, and "not evaluated" where landmarks were missing.

### 3.6 Exercise-library experience

- Cards with an SVG muscle map, equipment chips and **camera requirement** ("Side view · full body") as the most prominent attribute. Live analysis support shown as "Counts reps · Checks depth, chest" from the definition's `form_checks`.
- Detail: a setup illustration (top-down camera placement diagram), what is measured, what isn't ("can't tell curl from hammer curl"), and the user's personal range and history for that exercise.
- Unsupported search → honest "Track manually" with logging (the existing flow is good).

### 3.7 Analytics experience

- KPI strip: sessions this week vs target, volume, reps; tabular numerals; delta vs the previous period.
- Per-exercise page: e1RM trend (logged weight × reps, a reliable metric); **depth/ROM in degrees with view**, shown only from sets with calibration quality ≥ threshold; tempo; a rep timeline from stored per-rep evidence.
- Every AI-derived metric carries an "experimental" info affordance until the evaluation in `TEST_AND_EVALUATION_PLAN.md` exists.
- History list (F28) with session detail, which is the first missing piece.

### 3.8 Responsive layouts

| Breakpoint | Train mode | App mode |
| --- | --- | --- |
| < 640 px (phone, portrait) | Video fills the screen; counter bottom-right; gauge on the left edge; controls in a bottom sheet | Bottom tabs; single column; cards full-bleed |
| 640–1024 px (tablet/landscape phone) | Video left 2/3, HUD column right 1/3 | Two columns |
| ≥ 1024 px (laptop) | Video centre; side HUD; summary in a side panel | Left rail; 12-column grid; max width 1280 |

Recommend propping the phone in portrait for full-body exercises (taller frame, so the user can stand closer). This only works once the aspect bug is fixed.

### 3.9 Skeleton visualization design

- Draw on a canvas in a requestAnimationFrame loop, fed from the pose worker. Smooth with the same One Euro filter the engine uses, so what users see matches what is measured.
- Bones: `--bone`, 3 px (scaled by DPR and body size). Opacity mapped to min(visibility of both ends); below 0.5 the bone turns dashed in `--joint-lost`.
- Joints used by the active exercise: 7 px accent dots with a soft glow. Others: 4 px.
- Rule feedback: the joint and bone tied to a failing rule pulse `--joint-warn` for 600 ms when the rule fires.
- Optional angle label beside the primary joint (e.g. "94°"), off by default, shown in a "details" mode.
- Topology: include the midline (nose → shoulder centre → hip centre). It matches the engine graph.
- The 3D viewer is recast as a "Proportions" card: ratios vs typical ranges as bars. A static rotating T-pose doesn't explain anything to the user.

---

## 4. Frontend defects to fix (feeds Roadmap M0/M4/M5)

1. Rebuild or stop committing `web/out`; build it in CI or Docker only (`.gitignore` already lists `out/`, but the folder exists in the tree).
2. `live-workout.tsx:62`: respect `"auto"` (or remove auto from Quick set until M7).
3. Fetch the exercise list from `/api/exercises`; delete `SUPPORTED_EXERCISES`.
4. One exercise selector, between sets only.
5. Handle `confirm` with chips; handle WebSocket close with retry and a "set not saved" warning.
6. Error states for every query (`ErrorState` with retry).
7. Live region and optional audio for the rep count; focus trap and `aria-modal` on the live dialog; replace `confirm()` with an accessible dialog.
8. Fix the theme toggle (`darkMode: ["class", '[data-theme="dark"]']`).
9. Add a history page using the existing `GET /api/sessions` and `GET /api/sessions/{id}`.
10. Add workout editing using the existing `PUT /api/workouts/{id}`, plus a rest-seconds field.

import type { RecognitionOut, StateOut } from "./protocol";
/** Landmark frame (see motion_engine/skeleton.py IMAGE_KEY): joint -> [x/width, y/height, visibility], plus
 *  `_image: [width, height]` of the source image so the server can undo the per-axis normalisation. */
export type Landmarks = Record<string, number[]>;
export interface User { id: number; email: string; name: string; is_admin: boolean; goal: string | null; weekly_target: number }
export interface Exercise { id: string; name: string; family: string; difficulty: string; target_muscles: string[]; secondary_muscles: string[];
  equipment: string[]; movement_patterns: string[]; camera: { preferred: string; supported_views: string[]; notes: string }; form_checks?: string[] }
export interface WorkoutItem { id: number; exercise_id: string | null; exercise_name: string; sets: number; reps: number; weight_kg: number; rest_seconds: number; ai_analysis: boolean; target_muscles: string[] }
export interface Workout { id: number; name: string; notes: string; items: WorkoutItem[]; muscles: string[]; unsupported_count: number }
export interface SetOut { id: number; exercise_id?: string | null; exercise_name: string; set_number: number; reps: number; partial_reps: number; weight_kg: number; avg_form_score: number | null;
  avg_rom?: number | null; avg_rep_seconds?: number | null; source: string; created_at?: string | null; reps_detail?: RepDetail[] }
export interface SessionOut { id: number; name: string; started_at: string; finished_at: string | null; duration_minutes: number; sets: SetOut[]; total_reps: number; volume_kg: number; avg_form_score: number | null }
export interface Feedback { status: "good" | "warning" | "info"; message: string; rule: string }
export interface Progress { range: string; workouts: number; sets: number; reps: number; volume_kg: number; active_minutes: number; avg_form_score: number | null; avg_rom: number | null;
  days_trained: number; weekly_target: number; consistency: number; muscle_sets: Record<string, number>; muscle_share: Record<string, number>;
  personal_bests: Record<string, { max_weight_kg: number; max_reps: number }>; series: { date: string; volume_kg: number; reps: number; form: number | null }[]; unsupported_sets: number }
export interface Trend { direction: string; slope_pct_per_session?: number; points: number }
export interface ExerciseProgress { exercise: string; series: { date: string; best_e1rm: number | null; volume_kg: number; avg_rom: number | null; avg_form: number | null }[];
  personal_records: { date: string; e1rm: number }[]; trends: Record<string, Trend> }
export interface GoalProgress { goal: string | null; weekly_target: number; streak_weeks: number; volume_change_pct: number | null; headline: string;
  weeks: { week_start: string; sessions: number; target_met: boolean; volume_kg: number; avg_form: number | null }[]; form_trend: Trend }
export interface CalibrationSummary { status: "good" | "usable" | "retry"; quality: number; stability: number; confidence: number; coverage: number;
  plausibility: number; method: string; ratio_agreement_3d: number | null; warnings: string[]; setup: { label?: string; aspect?: number; image?: number[] } | null }
export interface CheckItem { id: string; ok: boolean; message: string | null }
export interface Assessment { ok: boolean; message: string | null; checks: CheckItem[]; metrics: Record<string, number | string>; stage: string; instruction: string }
export interface ProfileOut { calibrated: boolean; ratios: Record<string, number> | null; segments_cm: Record<string, number> | null; stages: string[];
  instructions: Record<string, string>; baselines: Record<string, { n: number; mean: { peak_progress: number; duration: number } }>; profile: { height_cm: number | null; quality: number };
  calibration: CalibrationSummary | null; missing_stages: string[] }
export interface FoodEntry { id: number; name: string; grams: number; kcal: number; protein_g: number; carbs_g: number; fat_g: number }
export interface Targets { bmr: number; maintenance_kcal: number; target_kcal: number; kcal_range: [number, number]; protein_g: number; fat_g: number; carbs_g: number; note: string }
export interface DaySummary { day: string; entries: FoodEntry[]; totals: { kcal: number; protein_g: number; carbs_g: number; fat_g: number }; targets: Targets | null; remaining_kcal: number | null;
  exercise_expenditure_estimate: { kcal_low: number; kcal_mid: number; kcal_high: number; weight_assumed: boolean } }
export interface ModelVersion { id: number; kind: string; name: string; version: number; dataset_version: string; status: string; metrics: Record<string, unknown>; limitations: string; lineage: Record<string, unknown> }

/** Live-set messages: generated from backend/protocol.py (see lib/protocol.ts). */
export type LiveState = StateOut;
export type Recognition = RecognitionOut;
export interface RepDetail { index: number; counted: boolean; peak_progress: number; duration: number; form_score: number | null;
  evidence: { rule_id: string; passed: boolean; severity: string; message: string; evaluated?: boolean; measured: number | null; threshold: number | null }[] }

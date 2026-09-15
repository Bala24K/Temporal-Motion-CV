"""
Demo: composes V1's rep boundaries, V2's rep/set analysis, and V3's
muscle-activation pipeline into the live-rep and post-set output the doc's
own mockups show — on synthetic data, the same "does this actually read
right" check that caught real bugs in the V1 overlay and the V2 set
summary.

Every muscle-related line is explicitly labeled "estimated" and carries a
confidence score and caveats — per V3 section 20/23, nothing here is
allowed to read like a measurement or a diagnosis.

Run: python demo_v3_muscle_estimate.py
"""
import numpy as np

from analysis.rep_analyzer import analyze_rep
from analysis.set_analyzer import summarize_set
from analysis.muscle_estimation import estimate_activation_sequence, summarize_rep_activation


def curl_rep(low, high, n=30, dt=0.05, t0=0.0):
    half = n // 2
    angles = list(np.linspace(high, low, half)) + list(np.linspace(low, high, n - half))
    times = [t0 + i * dt for i in range(n)]
    concentric = (half - 1) * dt
    eccentric = (n - half) * dt
    return times, angles, concentric, eccentric


def print_live_rep(rep_number, rep_metrics, muscle_summary, confidence_score, caveats):
    print(f"BICEP CURL — Rep {rep_number}")
    print(f"  ROM              {rep_metrics.rom:.0f} deg")
    print(f"  Tempo            {rep_metrics.tempo:.1f} s")
    print()
    print("  Estimated involvement")
    for name, s in muscle_summary.items():
        label = name.replace("_", " ").title()
        print(f"    {label:<18} {s['involvement']}")
    print()
    print(f"  Confidence       {confidence_score:.0f}%")
    for c in caveats[:1]:  # just the headline caveat inline, like the doc's mockup
        print(f"    ({c})")
    print()


def main():
    roms = [124, 122, 121, 119, 113, 108]  # 6-rep set, mild late-set decline
    body_mass_kg, body_height_m = 78.0, 1.78
    external_load_kg = 10.0  # a 10kg dumbbell curl

    reps = []
    all_frames = []
    t = 0.0
    for i, rom in enumerate(roms):
        times, angles, c, e = curl_rep(160 - rom, 160, t0=t)
        visibility = [0.93] * len(times)

        rep_metrics = analyze_rep(i + 1, times, angles, c, e)
        frames = estimate_activation_sequence(
            times, angles, visibility,
            body_mass_kg=body_mass_kg, body_height_m=body_height_m,
            external_load_kg=external_load_kg,
        )
        muscle_summary = summarize_rep_activation(frames)
        avg_confidence = sum(f.confidence.score for f in frames) / len(frames)
        caveats = frames[0].confidence.caveats

        print_live_rep(i + 1, rep_metrics, muscle_summary, avg_confidence, caveats)

        reps.append(rep_metrics)
        all_frames.extend(frames)
        t += 3.0

    # ---- post-set summary ----
    set_summary = summarize_set(reps)
    set_muscle_summary = summarize_rep_activation(all_frames)
    set_confidence = sum(f.confidence.score for f in all_frames) / len(all_frames)

    print("=" * 40)
    print("SET ANALYSIS")
    print()
    print(f"{set_summary.rep_count} repetitions")
    print(f"ROM consistency        {set_summary.rom_consistency:.0f}%")
    print(f"Tempo consistency      {set_summary.tempo_consistency:.0f}%")
    print()
    print("Estimated muscle involvement (peak across the set)")
    for name, s in set_muscle_summary.items():
        label = name.replace("_", " ").title()
        print(f"  {label:<18} {s['involvement']}   (peak activation ~{s['peak_activation']:.2f})")
    print()
    if set_summary.observations:
        print("Movement trend")
        for obs in set_summary.observations:
            print(f"  {obs}")
        print()
    print(f"Overall confidence     {set_confidence:.0f}%")
    print("  This is an estimate from a single camera view using generic")
    print("  (non-subject-specific) muscle and segment parameters — not a")
    print("  validated physiological measurement. Not medical advice.")
    print("=" * 40)


if __name__ == "__main__":
    main()

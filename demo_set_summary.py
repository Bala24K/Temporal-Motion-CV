"""
Demo: renders a set summary in the format the V2 doc's own mockup shows,
using synthetic reps (a 10-rep set with progressive ROM decline in the
back half, standing in for what a real recorded set would look like).

This isn't a test — it's here to eyeball whether the measurement layer
actually composes into something a person would find useful to read, the
same way the overlay_preview.png check caught a real bug in V1's UI.

Run: python demo_set_summary.py
"""
from analysis.rep_analyzer import analyze_rep
from analysis.set_analyzer import summarize_set


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def curl_rep(low, high, n=20, dt=0.05, t0=0.0, tempo_scale=1.0):
    half = n // 2
    angles = ramp(high, low, half) + ramp(low, high, n - half)
    times = [t0 + i * dt * tempo_scale for i in range(len(angles))]
    concentric = (half - 1) * dt * tempo_scale
    eccentric = (len(angles) - half) * dt * tempo_scale
    return times, angles, concentric, eccentric


def main():
    # ROM tapers off in the back half; tempo speeds up slightly (rushing) —
    # a plausible, if invented, "getting tired" pattern.
    roms = [124, 122, 121, 119, 113, 108, 104, 102, 100, 98]
    tempo_scales = [1.0, 1.0, 0.98, 0.97, 0.95, 0.9, 0.85, 0.83, 0.8, 0.78]

    reps = []
    t = 0.0
    for i, (rom, tscale) in enumerate(zip(roms, tempo_scales)):
        times, angles, c, e = curl_rep(160 - rom, 160, t0=t, tempo_scale=tscale)
        reps.append(analyze_rep(i + 1, times, angles, c, e))
        t += 3.0

    summary = summarize_set(reps)

    print("BICEP CURL — Set 1")
    print()
    print(f"{summary.rep_count} reps detected")
    print(f"ROM        Avg {summary.avg_rom:.0f}    Best {summary.best_rom:.0f}    Lowest {summary.lowest_rom:.0f}")
    print(f"ROM consistency:     {summary.rom_consistency:.0f}%")
    print(f"Tempo      Avg {summary.avg_tempo:.2f}s")
    print(f"Tempo consistency:   {summary.tempo_consistency:.0f}%")
    print()
    if summary.observations:
        print("Movement observation")
        for obs in summary.observations:
            print(f"  {obs}")
    else:
        print("No notable trends across this set.")


if __name__ == "__main__":
    main()

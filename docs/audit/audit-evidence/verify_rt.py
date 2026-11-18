"""Read-only reproduction of suspected defects in the RT (Kinetic) motion engine.
Run from the RT project root:  python <this file>
Does not modify any project file or database."""
import asyncio, math, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from motion_engine import synthetic as S
from motion_engine.calibration import check_stage_frame, calibrate, CalibrationError
from motion_engine.skeleton import frames_to_array
from motion_engine.primitives import compute_signal
from motion_engine.session import LiveSession
from motion_engine.recognition import recognize
from motion_engine.runtime import LatestSlot
from exercise_system.registry import ExerciseRegistry

reg = ExerciseRegistry.load()

def aspect(frame, k):
    """Isotropic coords -> MediaPipe normalized coords for a camera whose W/H = 1/k (x compressed by k)."""
    return {n: (0.5 + (v[0] - 0.5) * k, v[1], v[2]) for n, v in frame.items()}

print("== 1. Front calibration gate vs camera aspect (synthetic body, isotropic shoulder ratio)")
fr, _ = S.synth_standing(30)
for label, k in [("isotropic (tests)", 1.0), ("640x480 4:3", 0.75), ("1280x720 16:9", 9 / 16), ("480x640 portrait", 4 / 3)]:
    r = check_stage_frame("front", aspect(fr[0], k))
    print(f"  {label:20s} ok={r['ok']!s:5} shoulder_ratio={r['metrics'].get('shoulder_ratio')} msg={r['message']}")
    try:
        p, res = calibrate({"front": [aspect(f, k) for f in fr]})
        print(f"     calibrate(front only): quality={p.quality:.2f} plausibility={p.plausibility:.2f} profile.calibrated={p.calibrated}")
    except CalibrationError as e:
        print(f"     calibrate(front only): CalibrationError: {e}")

print("== 2. Staged calibration, all 3 stages, isotropic (what the demo body does)")
side = [S.yaw_project(f, 85) for f in fr]
up = [dict(S._pose(arm_elev=170, elbow_int=175)) for _ in range(30)]
p, res = calibrate({"front": fr, "side": side, "arms_up": up})
print(f"  quality={p.quality:.3f} plausibility={p.plausibility:.3f} calibrated={p.calibrated} stages={p.stages}")

print("== 3. Joint-angle distortion from un-corrected aspect (true angle -> measured)")
for true in (45, 90, 135, 170):
    # upper arm straight down, forearm rotated forward by (180-true) in the image plane
    a = math.radians(180 - true)
    sh, el = np.array([0.5, 0.3]), np.array([0.5, 0.45])
    wr = el + 0.12 * np.array([math.sin(a), math.cos(a)])
    out = []
    for k in (0.75, 9 / 16):
        s2, e2, w2 = [np.array([0.5 + (p_[0] - 0.5) * k, p_[1]]) for p_ in (sh, el, wr)]
        v1, v2 = s2 - e2, w2 - e2
        out.append(math.degrees(math.acos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))))
    print(f"  true {true:3d} deg -> 4:3 {out[0]:6.1f}  16:9 {out[1]:6.1f}")

print("== 4. Rep counting: clean synthetic vs realistic deviations (LiveSession, no profile)")
def run(ex, frames, ts):
    s = LiveSession(reg.get(ex))
    for f, t in zip(frames, ts):
        s.push(f, t)
    return s.counter.count, s.counter.partials

for ex in ("bicep_curl", "squat", "shoulder_press", "lateral_raise"):
    f, t = S.synth_sequence(ex, reps=5)
    print(f"  {ex:15s} clean 5 reps -> counted, partials = {run(ex, f, t)}")

orig = S.EX["bicep_curl"]
for rest in (160, 150, 140):
    S.EX["bicep_curl"] = lambda p, fault, r=rest: dict(elbow_int=S._lerp(r, 40, p), arm_elev=8, bend=-1, lean=3)
    f, t = S.synth_sequence("bicep_curl", reps=5)
    # lead-in still uses p=0 -> elbow at `rest`; user never fully extends
    print(f"  bicep_curl, arm only re-extends to {rest} deg between reps -> {run('bicep_curl', f, t)}")
S.EX["bicep_curl"] = orig
origp = S.EX["shoulder_press"]
for bottom in (100, 110):
    S.EX["shoulder_press"] = lambda p, fault, b=bottom: dict(elbow_int=S._lerp(b, 172, p), arm_elev=S._lerp(85, 168, p), bend=+1, lean=2)
    f, t = S.synth_sequence("shoulder_press", reps=5)
    print(f"  shoulder_press, bottom elbow {bottom} deg (not 90) -> {run('shoulder_press', f, t)}")
S.EX["shoulder_press"] = origp

# hold at peak longer than max_rep_seconds (12 s)
f, t = S.synth_sequence("squat", reps=1, rep_seconds=2.4)
mid = len(f) // 2
hold = [f[mid]] * int(13 * 30)
frames = f[:mid] + hold + f[mid:]
ts = [i / 30 for i in range(len(frames))]
print(f"  squat with a 13 s hold at the bottom -> {run('squat', frames, ts)}  (expected 1)")

# aspect-distorted curl at 16:9
f, t = S.synth_sequence("bicep_curl", reps=5)
print(f"  bicep_curl clean but 16:9 un-corrected -> {run('bicep_curl', [aspect(x, 9/16) for x in f], t)}")
f, t = S.synth_sequence("lateral_raise", reps=5)
print(f"  lateral_raise clean but 16:9 un-corrected -> {run('lateral_raise', [aspect(x, 9/16) for x in f], t)}")

print("== 5. Rule recognizer on each clean synthetic exercise (90-frame+ windows)")
for ex in S.EX:
    f, _ = S.synth_sequence(ex, reps=3)
    r = recognize(f[-150:], reg)
    print(f"  {ex:28s} status={r.status:11s} action={r.action:8s} top={r.top.members if r.top else None} conf={r.confidence:.2f}")
f, _ = S.synth_standing(150)
print(f"  {'standing still':28s} status={recognize(f, reg).status}")

print("== 6. LatestSlot: control message followed by a frame before the consumer runs")
async def slot_test():
    s = LatestSlot()
    s.put({"type": "select_exercise", "exercise": "squat"})
    s.put({"type": "frame"})
    got = await s.get()
    return got["type"], s.dropped
print("  consumer receives:", asyncio.run(slot_test()))

print("== 7. Equivalence groups (identical to the recognizer)")
groups = {}
for d in reg.recognisable():
    groups.setdefault(d.equivalence_key(), []).append(d.id)
print("  ", [g for g in groups.values() if len(g) > 1])

print("== 8. Rule recognizer on movements outside the vocabulary (unknown rejection)")
def seq(fn, reps=4, n=72):
    out = []
    for _ in range(reps):
        for i in range(n):
            p = S.smoothstep(1 - abs(2 * i / n - 1)); out.append(S._pose(**fn(p)))
    return out
novel = {
    "Y-raise (arms to 170)": lambda p: dict(arm_elev=S._lerp(8, 170, p), elbow_int=170),
    "thruster (squat+press)": lambda p: dict(knee_int=S._lerp(172, 95, p), elbow_int=S._lerp(170, 60, p)),
    "upright-row-like": lambda p: dict(elbow_int=S._lerp(170, 70, p), arm_elev=S._lerp(8, 80, p), bend=+1),
    "side bend / sway": lambda p: dict(lean=S._lerp(0, 30, p)),
    "half curl (to 100)": lambda p: dict(elbow_int=S._lerp(170, 100, p)),
    "arm swing small": lambda p: dict(arm_elev=S._lerp(8, 45, p), elbow_int=165),
}
for k, fn in novel.items():
    r = recognize(seq(fn), reg)
    print(f"  {k:24s} {r.status:11s} {r.action:8s} top={r.top.members if r.top else None} conf={r.confidence:.2f}")

print("== 9. Form evidence under aspect distortion (clean synthetic sets)")
for ex in ("bicep_curl", "lateral_raise", "shoulder_press", "squat"):
    for lab, k in (("iso", 1.0), ("4:3", .75), ("16:9", 9 / 16)):
        f, t = S.synth_sequence(ex, reps=4)
        s = LiveSession(reg.get(ex))
        for a, b in zip(f, t):
            s.push(aspect(a, k), b)
        sm = s.summary()
        fails = sorted({e['rule_id'] for r in sm['reps_detail'] for e in r['evidence'] if not e['passed']})
        print(f"  {ex:15s} {lab:5s} reps={sm['reps']} form={sm['avg_form_score']} failing={fails}")

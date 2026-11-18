# Pose Estimation and Skeleton Calibration Audit

Aliases (`RT/`, `V1/`, `MU/`) and evidence tags are defined in `PROJECT_AUDIT.md`.

---

## 1. Pipeline trace: camera frame to displayed skeleton (RT)

| Step | Code | What happens | Problems |
| --- | --- | --- | --- |
| 1. Acquire | `RT/web/lib/pose.ts:52` | `getUserMedia({video:{width:640,height:480,facingMode:"user"}})` | These are *ideal* constraints. Real devices return 640×480, 1280×720, 640×360 or a portrait 480×640 on phones. The resulting resolution and aspect are never read or sent. No frame-rate constraint. |
| 2. Display | `live-workout.tsx:110-112` | `<video>` with `object-contain`, mirrored with CSS `scaleX(-1)` together with the canvas | Fine. The overlay and the video share one transform. |
| 3. Pose model | `pose.ts:55-58` | `@mediapipe/tasks-vision@0.10.14` from jsDelivr; `pose_landmarker_lite/float16/latest`; GPU delegate; `runningMode: VIDEO`; `numPoses: 1`; default confidences (0.5) | Lowest-accuracy model; the `latest` model URL is unpinned; no CPU fallback if the GPU delegate fails; needs network access to the CDN; runs on the main thread. |
| 4. Inference loop | `pose.ts:62-71` | Runs `detectForVideo(video, performance.now())` once per new `currentTime` via `requestVideoFrameCallback` | The timestamp is processing time, not capture time. |
| 5. Conversion | `pose.ts:60` | `toLm`: 13 of 33 landmarks → `[x, y, visibility ?? 1]` | **`x` is a fraction of image width and `y` a fraction of image height; nothing records W/H.** `z`, `presence` and `worldLandmarks` (3D, hip-centred, in metres) are discarded. If `visibility` is missing it defaults to 1 (fully trusted). |
| 6. Draw | `pose.ts:14-25` | Letterboxed mapping `ox + x·rw, oy + y·rh`, threshold vis > 0.3, single colour | Geometry is correct for display (it multiplies by the rendered width and height). No confidence encoding, no smoothing, nose not connected, no midline. |
| 7. Transport | `useLiveSet.ts:47-49` | JSON frame `{t, landmarks}` per frame over WebSocket | No resolution, aspect, model, mirroring or frame index. |
| 8. Server array | `RT/motion_engine/skeleton.py:32-47` | `(15,3)` with NaN for vis < 0.3; synthetic `hip_center`/`shoulder_center` midpoints | A midpoint becomes NaN when one side is missing. In a side view this often removes the whole torso. |
| 9. Quality gate | `RT/motion_engine/quality.py:11-28`, `QualityTracker` | Every required joint needs vis ≥ 0.5 and a position inside a 2% margin; 15-frame window, bad if ≥ 40% failing | Two different visibility thresholds (0.3 for geometry, 0.5 for gating). Margins are in anisotropic units. |
| 10. Normalize | `skeleton.py:167-185` | Hip-centred; divided by the **calibration-time torso length** (if the profile is calibrated) or the per-frame torso; x divided by cos(yaw) for frontal exercises | Anisotropy is never removed. The fixed-scale branch ties results to the camera distance at calibration time (no current signal depends on it, but any future distance or offset signal would). |

### 1.1 2D versus 3D semantics

- RT uses **only 2D image landmarks**. Every angle is the angle of a 2D projection. Joint movement toward or away from the camera (curl forearm from the front, squat knee from the front, front raise from the front) shrinks or distorts the projected angle.
- MediaPipe also returns **world landmarks** (approximate 3D, metres, hip origin, estimated by the model). The app discards them. They could reduce front-view foreshortening, but their depth is a model estimate with its own error. Using them needs an evaluation before any accuracy claim (see `TEST_AND_EVALUATION_PLAN.md`).
- Yaw compensation (`skeleton.py:157-164`) divides x by `(shoulder_width/torso) / calibrated_ratio`, clamped to [0.5, 1]. That is exact only for orthographic projection of frontal-plane motion with an upright torso. It also treats shoulder shrugging, arm-raise shoulder elevation and protraction as "yaw".

### 1.2 Missing, occluded and unstable landmarks

- Below vis 0.3 a joint becomes NaN. `fill_gaps` (linear, at most 6 frames) is used only in recognition and offline analysis. The live `LiveSession` uses single-frame arrays, so any NaN joint makes the signal NaN for that frame, and `RepCounter.update` skips the frame (`RT/motion_engine/repetition.py:72`).
- **No temporal smoothing** of landmarks beyond MediaPipe's internal VIDEO-mode tracking. The only filter is an EMA with α = 0.5 on the rep signal. That filter depends on frame rate: at 15 fps it smooths over twice the real time it covers at 30 fps.
- In side views MediaPipe still reports the far-side limbs, often with plausible visibility values, but their positions are guesses. `side: "mean"` primitives average the visible limb with the guessed one [Hypothesis: confirm with recorded side-view footage].
- Of QEVD frames with a detection, only **43.8%** had all 13 landmarks at vis ≥ 0.5 inside the 2% margin [Measured]. Full-body framing at 640×480 is hard to meet.

### 1.3 Anatomical relationships and skeleton connectivity

- `BONES` (`pose.ts:6-8`) draws 12 bones. The nose floats unconnected and the midline is not drawn. The server graph (`RT/ml/seq.py:19`) uses 14 edges including the midline. The display and the model therefore use different topologies (cosmetic).
- No bone-length consistency check runs at runtime. A limb that suddenly doubles in length (a mis-tracked wrist, say) passes straight into angle computation.
- Shoulder and hip landmarks are MediaPipe joint estimates, not anatomical landmarks (acromion, ASIS). Segment "lengths" are distances between joint-centre estimates.

---

## 2. Calibration implementation (RT)

### 2.1 Procedure as built

`RT/web/components/calibration.tsx` → `POST /api/profile/calibrate/stage-check` per frame → `POST /api/profile/calibrate/staged` → `RT/motion_engine/calibration.py:calibrate`.

1. Height (optional) is typed in.
2. For each stage (front, side, arms_up): start the camera; every 300 ms `grab()` a frame and POST it to `stage-check`; keep it if `ok`; collect 20 frames or give up after 40 s; need at least 8.
3. `check_stage_frame` (`calibration.py:42-64`):
   - all 13 landmarks at vis ≥ 0.5 and inside a 2% edge margin (`check_frame`);
   - front/arms_up: **shoulder_ratio = |LS−RS| / |shoulder_center−hip_center| ≥ 0.55**, else "Turn to face the camera.";
   - side: ratio ≤ 0.40;
   - front: both wrists below shoulders; arms_up: both wrists above shoulders.
4. `calibrate`: `measure_body` on the front frames (median torso; `quality = 1 − 4·MAD/median`). Segment ratios use the **90th percentile** across all stages. Shoulder and hip ratios come from the front stage. `cm_per_torso` uses height × 0.92 (nose fraction) / nose-to-ankle pixels. Plausibility compares the ratios with Drillis–Contini averages (±40%). Final quality is `quality × (0.5 + 0.5·plausibility) × coverage{1:0.6, 2:0.8, 3:1.0}`.
5. The profile is saved as JSON in `body_profiles`. The response always says `"calibrated": true` (`RT/backend/routers/profile.py:79`).

### 2.2 What the profile is used for

| Consumer | Uses | Effect |
| --- | --- | --- |
| `normalize()` | `torso_len` as fixed scale if `profile.calibrated` | **None in practice.** All definition signals are angles, which don't change with scale. |
| `normalize(compensate_yaw)` | `shoulder_ratio` | Only for `lateral_raise` and `shoulder_press` (`motion_plane: frontal`) |
| `recognize()` | — (profile not passed) | None |
| `LearnedRecognizer` | — | None |
| `segments_cm` → 3D viewer | `cm_per_torso`, segment lengths | Display only |
| Rep thresholds | — | **None.** `rest`/`peak` are textbook constants; `RepCounter.set_range` is never called. |

---

## 3. Root causes: why the skeleton cannot calibrate reliably to a real person

Ordered by impact.

### RC-1. Coordinate-system contract bug (anisotropic normalized coordinates) — Critical [Reproduced]

MediaPipe returns `x = px/W` and `y = py/H`. In a landscape frame one unit of x covers more pixels than one unit of y, but every distance, ratio and angle in RT (`skeleton.py`, `primitives.py`, `calibration.py`) assumes isotropic units. The synthetic generator produces isotropic coordinates, so all 163 tests miss it.

**Real-footage evidence** (`docs/audit-evidence/pose_study.py`, 40 QEVD clips, MediaPipe lite, frames filtered to full body visible and both wrists below shoulders):

| Camera aspect | Frames | Median shoulder/torso, app's normalized coords | Median, pixel-corrected | Pass front gate (≥ 0.55), normalized | Pass, pixel-corrected |
| --- | --- | --- | --- | --- | --- |
| 4:3 (640×480) | 179 | 0.515 | 0.686 | **0%** | 97% |
| 16:9 (640×360) | 552 | 0.394 | 0.694 | **25%** | 94% |

(QEVD frames are mid-exercise and facing is not controlled, so treat these as indicative.) The two aspect ratios agree once corrected (0.686 vs 0.694), which is what you'd expect if the distortion is purely the coordinate artefact.

**Synthetic evidence** (`verify_rt.py §1`): the synthetic body passes at isotropic (ratio 0.592) and portrait 3:4 (0.79), and fails at 4:3 (0.444) and 16:9 (0.333) with "Turn to face the camera."

**Consequences beyond the gate**:
- Joint angles are distorted. A true 45° elbow reads 36.9° at 4:3 and 29.4° at 16:9; a true 135° reads 143.1° and 150.6° (`verify_rt.py §3`).
- Plausibility and the reported segment centimetres are distorted (shoulder width shrinks).
- Yaw compensation measures part of the aspect distortion as yaw.
- A calibration made on one device or aspect (for example a portrait phone) is silently wrong on another.

### RC-2. The gate is a fixed, population-level threshold used as a pose detector — High

Even after aspect correction, the 0.55 front threshold and 0.40 side threshold are untested constants. Shoulder/torso varies between people, with clothing, posture and shoulder elevation. Real pixel-corrected ratios had a median near 0.69 with some frames under 0.55 (3–6% failing). A narrow-shouldered or long-torsoed user, or one wearing loose clothing, will fail the "face the camera" check while facing the camera. Facing direction should come from geometry that does not depend on the user's own proportions (shoulder z-difference from world landmarks, nose/ear symmetry, hip-vs-shoulder consistency), with the ratio *measured*, not *gated*.

### RC-3. All-or-nothing visibility requirement — High

`check_frame` needs all 13 landmarks at vis ≥ 0.5 inside a 2% margin for every stage.
- In the **side** stage the far arm is occluded by design, so this requirement conflicts with the purpose of the stage. [Hypothesis: MediaPipe's visibility for occluded far-side joints is often below 0.5, which would mean the side stage is rarely passable.]
- At 640×480 the user has to be roughly 2.5–3.5 m away for head-to-ankle framing with margins. The lite model at that distance gives smaller, noisier landmarks. QEVD frames met the requirement only 43.8% of the time.
- The user gets one generic message ("Step back…", "Make sure your arms…") instead of which joint failed and in which direction to move.

### RC-4. No temporal stability or stillness check — Medium

Calibration keeps any frame that passes the per-frame checks, sampled every 300 ms over up to 40 s. There's no check that the user stood still, that frames are consecutive, or that the camera didn't move between stages. `measure_body`'s MAD-based quality is computed only over the front frames that passed, which are already filtered.

### RC-5. The profile is not tied to a camera setup — High

The stored profile has no resolution, aspect, device id, camera height or tilt, user distance (torso pixel fraction) or timestamp. Nothing detects a moved camera, a different device or a different distance. Because the fixed scale is measured in image units, it depends on distance and resolution by construction (`skeleton.py:177-178`). It is harmless today only because no live signal uses distance.

### RC-6. Inconsistent "calibrated" state — Medium-High [Reproduced]

`calibrate_staged` always returns `"calibrated": true`. But `BodyProfile.calibrated` needs `quality ≥ 0.5`, and the final quality is multiplied by coverage. A front-only calibration of the clean synthetic body yields quality 0.50 and `calibrated=False` (`verify_rt.py §1`). The UI shows "Calibrated", while `/api/profile` reports `calibrated: false` and `normalize()` ignores the profile scale. The profile page's "Your body model" card also needs a height to render anything.

### RC-7. Calibration doesn't calibrate the thing users experience — Critical (product)

Users expect calibration to make counting and feedback fit *their* body and range. The profile changes no rep threshold, no form threshold and no recognition behaviour (§2.2). Even a perfect calibration would leave rep-counting accuracy unchanged. The missing piece is **per-exercise personal range calibration**: capture the user's own rest and peak over 2–3 controlled reps.

### RC-8. Metric claims without a metric basis — Medium

`cm_per_torso` uses `height × 0.92 / (ankle_y − nose_y)` in normalized y. It assumes a fronto-parallel upright person, no lens distortion, no camera tilt and a correct ankle landmark, and ignores aspect for x-segments. The UI then shows segment lengths to 0.1 cm next to an "average body" (`profile/page.tsx`). Those are **relative proportions scaled by a self-reported height**, not measurements. Their error is unknown and probably several centimetres.

### RC-9. Camera path never exercised — Process

`RT/README.md` limit #2 and `STACK.md` say `pose.ts` was "written, not exercised". The browser visibility semantics (`p[i].visibility ?? 1`), GPU delegate fallback, CDN availability and the real camera resolution have never been observed. The local DB holds only the demo-body profile, which is consistent with that.

---

## 4. V1 calibration (for completeness)

`V1/ui/dashboard.py` has a "camera setup / calibration screen" with a body-positioning guide, and `V1/biomechanics/anthropometrics.py` estimates segments from height using population tables. V1 captures 1280×720 (16:9) and also has no aspect correction [Inspected]. V1's setup check is a visibility guide, not a stored personal calibration. Nothing in V1 calibration should be ported except the setup-screen idea.

---

## 5. Three kinds of measurement

Every screen and API field must say which of these it reports.

| Kind | Definition | Valid claims | Invalid claims |
| --- | --- | --- | --- |
| **Normalized image coordinates** | MediaPipe x/W, y/H in [0,1] | Where a joint is in the frame; framing checks *after* converting to pixels | Distances or angles (anisotropic) |
| **Isotropic image geometry** | Pixels, or normalized by one dimension (`x·W/H`, y) | 2D projected angles; relative proportions (ratios) **for a fixed view** | True 3D angles for out-of-plane motion; absolute size |
| **Relative body proportions** | Dimensionless ratios (e.g. upper_arm/torso) from controlled poses | Personal normalisation; plausibility; changes in proportions *under the same protocol* | Anatomical segment lengths |
| **Metric measurements** | cm, m | Only with a known scale reference (self-reported height ± stated error, a known-size object, or a calibrated depth/stereo camera) | — without one of those |

MediaPipe world landmarks are in metres, but the model estimates them up to its own scale assumptions. Treat them as approximate 3D for angles, not as a ruler.

---

## 6. Recommended calibration protocol

### 6.1 Coordinate contract (prerequisite for everything else)

Each frame message should carry:
```json
{"t_capture_ms": 12345.6, "frame_idx": 812, "image": {"w": 1280, "h": 720, "mirrored": true},
 "model": {"name": "pose_landmarker_full", "version": "0.10.x"},
 "lm2d": {"left_shoulder": [x_px_norm, y, vis, presence], "...": []},
 "lm3d": {"left_shoulder": [X, Y, Z, vis], "...": []}}
```
- Convert to isotropic coordinates **once, at the boundary**: `x_iso = x · W/H`, `y_iso = y`, so units are image heights. Every distance and angle downstream assumes isotropy. Framing checks use the original [0,1] values.
- Use `t_capture_ms` from `requestVideoFrameCallback` metadata.
- Keep `presence` and 3D world landmarks so angle sources can be chosen and evaluated per exercise.

### 6.2 Guided user positioning (setup phase, before calibration)

1. Live framing overlay: a target silhouette box. Per-joint status dots coloured by visibility.
2. Distance guidance: the body's vertical extent (nose→ankle) as a fraction of frame height, target 0.65–0.85; say "step back / closer" with an arrow.
3. Horizontal centring: hip centre x within ±0.15 of the centre.
4. Camera height and tilt: ankle line and shoulder line roughly horizontal (|roll| < 5°). Hip y within the middle 40% of the frame (camera at about hip height).
5. Lighting check: mean visibility over the last 30 frames ≥ 0.8 and per-joint jitter below a threshold (step 6.5). Otherwise "Improve lighting / reduce backlight".
6. The view required by the chosen exercise (front / side / 45°) is shown as an illustration. The check comes from geometry (step 6.3), not the shoulder ratio alone.

### 6.3 Neutral pose detection

Neutral = standing, facing the camera, arms relaxed by the sides. Detect it with tests that don't depend on the user's proportions:
- facing: world-landmark |Z_ls − Z_rs| < 0.06 m and |Z_lh − Z_rh| < 0.06 m (thresholds to be tuned on recorded data); 2D nose x between the shoulders;
- upright: trunk lean < 10°; knees > 165°; hips level;
- arms down: wrist y below hip y, elbow angle > 150°;
- still: next section.

### 6.4 Landmark confidence requirements

| Joint group | Minimum visibility | Minimum presence | Notes |
| --- | --- | --- | --- |
| Joints required by this calibration stage | 0.7 (median over the window) | 0.9 | Per-joint, not all-or-nothing across unrelated joints |
| Joints not required in this stage (e.g. far arm in side stage) | — | — | Excluded from the gate and from measurement |
| Per frame | Rejected if any required joint < 0.5 | | |

### 6.5 Temporal stability checks

- Collect a contiguous window of N = 45 frames (1.5 s at 30 fps), not sparse samples.
- Per joint: robust jitter = median |second difference| / torso length < 0.01. Per-segment length CV < 3%.
- Body bounding box drift < 2% of frame height across the window (detects walking or a moving camera).
- The window restarts automatically if a check fails, with a specific message.

### 6.6 Body-proportion estimation

- From the front neutral window: shoulder_width/torso, hip_width/torso, upper_arm/torso, forearm/torso, thigh/torso, shin/torso. Take medians over the window, in isotropic units, preferring 3D world-landmark lengths when they are available and agree with 2D within a tolerance.
- From a T-pose or arms-forward window (replacing arms_up): arm segment lengths are measured while the arms lie close to the image plane. "Arms up" puts the arms parallel to the torso axis, which is also fine for length, but the overhead wrist is often at the frame edge.
- Do not use the side stage for lengths (far side is occluded). Use it only for sagittal exercises' reference posture, if at all.
- Store the ratios only. Absolute lengths are shown only when height is given, with an explicit ± range.

### 6.7 Camera-position validation and invalidation

Store a **setup fingerprint** with every calibration:
`{device_label, w, h, aspect, mirrored, body_height_frac, hip_center_xy, ankle_line_roll, shoulder_line_roll, view: "front"|"side"|"45", created_at}`.

At the start of each set, compare a 1-second neutral window with the fingerprint:
- `body_height_frac` changed by > 15%: distance changed. Re-validate distance but keep the ratios (they don't depend on scale).
- Aspect or resolution or device changed: re-run the front window (≈5 s) and re-check ratios against stored ones (each within 8%). If they agree, keep the profile and add the new fingerprint.
- Ratios disagree by > 15%: different person, or a bad calibration. Ask to recalibrate.

### 6.8 Calibration quality scoring

`Q = min(stability, confidence, coverage) × plausibility_weight`, with each component in [0,1] and reported separately:
- stability = 1 − (jitter / limit), clipped;
- confidence = median required-joint visibility normalized to [0.7, 1];
- coverage = stages captured / stages needed for the user's exercise set;
- plausibility: deviation of ratios from population ranges, used **only to warn** ("unusual proportions — check camera angle") and never to block.

Thresholds: Q ≥ 0.8 "good", 0.6–0.8 "usable with warning", < 0.6 "retry". The engine and UI read the **same** `calibration.status` field (fixes RC-6).

### 6.9 Failure detection and recovery

| Failure | Detection | User message / recovery |
| --- | --- | --- |
| Joint out of frame | Normalized coordinate outside the margin | "Your left foot is out of frame — step back" plus an arrow |
| Low visibility | Required joint median vis < 0.7 | "We can't see your right wrist clearly — improve lighting or remove baggy sleeves" |
| Not facing | 3D shoulder depth difference | "Turn slightly to your left" (signed) |
| Moving | Bounding-box drift / jitter | "Hold still for 2 seconds" plus a countdown ring |
| Timeout (> 30 s on one stage) | Timer | Show the single most frequent failure reason; offer the alternative view or "skip, use defaults" |
| GPU delegate failure | Exception in `createFromOptions` | Retry with the CPU delegate; report it in telemetry |
| CDN unavailable | Fetch error | Serve the model and wasm from the app origin (self-host) |

### 6.10 Persistent personalized calibration data

```
body_calibration(user_id, version, ratios{...}, quality{stability, confidence, coverage, plausibility, Q},
                 height_cm?, method: "2d"|"3d", created_at)
camera_setup(user_id, fingerprint{...}, calibration_version, last_validated_at)
exercise_range(user_id, exercise_id, view, rest_value, peak_value, signal_spec_hash,
               n_reps, captured_at, setup_fingerprint_id)        ← personal ROM; drives rep thresholds
```
`exercise_range` is captured with a 2–3 rep "show me your range" step the first time the user does an exercise, or after the setup changes. The rep counter uses `rest_value`/`peak_value`, clamped to within ±25% of the definition's population range (so a calibration made while barely moving can't make the counter trivially easy). See `REP_COUNTING_AUDIT.md §4`.

---

## 7. Pose-model recommendations (evidence first)

1. **Keep MediaPipe.** Pin the exact model asset and wasm version and self-host them.
2. Benchmark **lite vs full** (and heavy on desktop) on recorded user footage before choosing. Lite measured 14.2 ms p50 on CPU in Python here [Measured]. Browser and GPU numbers are unknown.
3. Move inference into a **Web Worker** (`OffscreenCanvas`/`ImageBitmap`) to free the main thread.
4. Add a per-landmark **One Euro filter** (low lag at speed, smooth at rest), gated by visibility, before any signal computation. Tune its parameters on recorded data, and measure the lag it adds.
5. Don't adopt RTMPose/YOLO-Pose until the benchmark harness (`RT/motion_engine/pose.py:benchmark`) has been run on real labelled footage.

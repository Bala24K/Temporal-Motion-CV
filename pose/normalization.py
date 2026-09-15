"""
Body-relative pose normalization (V2.1 / V2.3).

Two people of different heights, standing at different distances from the
camera, performing the same movement should produce near-identical
normalized features. Without this, an ML model trained on one body size
just learns "pixel coordinates of this particular person" — the doc's
own warning in section 3.

Normalization pipeline, applied per frame:
  1. TRANSLATE  — move the origin to the pelvis midpoint (hip center).
  2. SCALE      — divide all coordinates by torso length (shoulder-center
                  to hip-center distance), so a "1.0 unit" arm swing means
                  the same thing regardless of how close someone stands to
                  the camera or how tall they are.
  3. ROTATE     — rotate around the pelvis so the shoulder line is
                  horizontal, removing incidental camera-relative tilt
                  (e.g. the person standing slightly angled to the lens).

This does NOT fix true camera-angle problems (frontal vs. 30 degrees vs.
side-on) — that's a different, harder issue noted in the V1 README. This
only removes translation/scale/in-plane-rotation nuisance variation for a
GIVEN camera angle.
"""
from dataclasses import replace
from typing import Optional
import numpy as np

from pose.landmarks import PoseFrame, Landmark


def _midpoint(a: Landmark, b: Landmark):
    return np.array([(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2])


def normalize_frame(frame: PoseFrame, min_visibility: float = 0.4) -> Optional[PoseFrame]:
    """
    Returns a new PoseFrame with all landmark coordinates normalized as
    described above. Returns None if the landmarks needed to compute the
    normalization basis (both shoulders, both hips) aren't visible —
    callers should treat that like any other dropped reading, not silently
    fall back to un-normalized coordinates (which would defeat the point).
    """
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not frame.all_visible(required, min_visibility):
        return None

    ls, rs = frame.get("left_shoulder"), frame.get("right_shoulder")
    lh, rh = frame.get("left_hip"), frame.get("right_hip")

    pelvis = _midpoint(lh, rh)
    shoulder_center = _midpoint(ls, rs)

    torso_vec = shoulder_center - pelvis
    torso_len = float(np.linalg.norm(torso_vec[:2]))  # use x,y only — z is noisier
    if torso_len < 1e-6:
        return None

    # Rotation to make the shoulder line horizontal (2D, in the x-y plane).
    shoulder_vec = np.array([rs.x - ls.x, rs.y - ls.y])
    shoulder_len = np.linalg.norm(shoulder_vec)
    if shoulder_len < 1e-6:
        angle = 0.0
    else:
        angle = -np.arctan2(shoulder_vec[1], shoulder_vec[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    new_frame = PoseFrame(timestamp=frame.timestamp)
    for name, lm in frame.landmarks.items():
        p = np.array([lm.x, lm.y, lm.z]) - pelvis
        p_xy = rot @ p[:2]
        p = np.array([p_xy[0], p_xy[1], p[2]]) / torso_len
        new_frame.landmarks[name] = Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                              visibility=lm.visibility)
    return new_frame

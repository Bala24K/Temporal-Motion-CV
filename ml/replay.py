"""Replay landmark recordings through the live engine, deterministically and without a camera.

    python -m ml.replay recording.jsonl [--exercise squat] [--expect-reps 10] [--payloads out.jsonl]

Input: the JSON Lines file the live screen's "Record landmarks" option downloads - an optional
`{"type": "meta", "exercise": ...}` header, then `{"type": "frame", "t": seconds, "landmarks": {...}}` per frame
(exactly what the WebSocket received, incl. `_image`). No video is involved.

Every frame goes through the same contract check and `LiveSession.push` the WebSocket handler uses, so a recording
reproduces a live set's per-frame output exactly (except that the live server may drop frames under load; the
replay processes every recorded frame). This is the basis of the golden-set regression suite
(docs/TEST_AND_EVALUATION_PLAN.md section 2.3).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from exercise_system.registry import ExerciseRegistry
from motion_engine.contract import frame_error
from motion_engine.session import LiveSession
from motion_engine.skeleton import BodyProfile


def read_recording(path: str | Path) -> Tuple[dict, List[Tuple[float, dict]]]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    meta, frames = {}, []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "meta":
                meta = row
            elif row.get("type", "frame") == "frame":
                frames.append((float(row["t"]), row["landmarks"]))
    return meta, frames


def replay(frames: Iterable[Tuple[float, dict]], definition, profile: Optional[BodyProfile] = None,
           target_reps: Optional[int] = None) -> Dict:
    sess = LiveSession(definition, profile=profile, target_reps=target_reps)
    payloads, rejected = [], 0
    for t, lm in frames:
        if frame_error(lm):
            rejected += 1
            continue
        payloads.append(sess.push(lm, t))
    return {"summary": sess.summary(), "payloads": payloads, "frames": len(payloads), "rejected_frames": rejected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("--exercise", help="override the exercise recorded in the meta header")
    ap.add_argument("--expect-reps", type=int, help="exit non-zero if the counted reps differ")
    ap.add_argument("--payloads", help="write per-frame payloads as JSON Lines")
    a = ap.parse_args(argv)
    meta, frames = read_recording(a.recording)
    ex = a.exercise or meta.get("exercise")
    reg = ExerciseRegistry.load()
    defn = reg.get(ex) if ex else None
    if not defn:
        print(f"unknown exercise {ex!r}; pass --exercise (one of {sorted(d.id for d in reg.all())})", file=sys.stderr)
        return 2
    res = replay(frames, defn)
    s = res["summary"]
    print(json.dumps({"exercise": defn.id, "frames": res["frames"], "rejected_frames": res["rejected_frames"],
                      "reps": s["reps"], "partial_reps": s["partial_reps"], "avg_form_score": s["avg_form_score"]}))
    if a.payloads:
        Path(a.payloads).write_text("".join(json.dumps(p) + "\n" for p in res["payloads"]))
    if a.expect_reps is not None and s["reps"] != a.expect_reps:
        print(f"expected {a.expect_reps} reps, counted {s['reps']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

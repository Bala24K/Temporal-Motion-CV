"""Label Studio bridge: engine-assisted pre-annotation out, human annotations back in.

Result formats follow Label Studio's documented schema for <TimelineLabels>/<Choices>; verify against a real
export the first time you use it (the parser ignores result types it does not recognise).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from exercise_system.registry import ExerciseRegistry
from motion_engine.repetition import counter_for
from motion_engine.primitives import compute_signal
from motion_engine.skeleton import frames_to_array, normalize, fill_gaps
from .schema import ClipAnnotation, FORM_ISSUES, RepAnnotation
from .store import DatasetStore


def label_config(registry: ExerciseRegistry) -> str:
    ex = "".join(f'<Choice value="{d.id}"/>' for d in registry.all()) + '<Choice value="other_unsupported"/>'
    issues = "".join(f'<Choice value="{i}"/>' for i in FORM_ISSUES)
    return f"""<View>
  <Video name="video" value="$video" framerate="$fps"/>
  <TimelineLabels name="reps" toName="video">
    <Label value="good_rep" background="#14895c"/><Label value="partial_rep" background="#c77610"/><Label value="bad_rep" background="#c93c3c"/>
  </TimelineLabels>
  <Choices name="exercise" toName="video" choice="single" showInline="true">{ex}</Choices>
  <Choices name="view" toName="video" choice="single" showInline="true"><Choice value="front"/><Choice value="side"/><Choice value="45deg"/></Choices>
  <Choices name="issues" toName="video" choice="multiple" showInline="true">{issues}</Choices>
  <Choices name="camera_ok" toName="video" choice="single"><Choice value="yes"/><Choice value="no"/></Choices>
  <TextArea name="notes" toName="video" rows="2"/>
</View>"""


def prelabel_reps(frames, exercise_id: str, registry: ExerciseRegistry, fps: float) -> List[RepAnnotation]:
    """Run the engine over the clip to propose rep boundaries; annotators correct instead of drawing from scratch."""
    d = registry.get(exercise_id)
    if not d:
        return []
    arr = normalize(fill_gaps(frames_to_array(frames)))
    sig = compute_signal(arr, d.rep.signal)
    rc, reps, start, started = counter_for(d.rep), [], 0, False
    for i, v in enumerate(sig):
        ev = rc.update(i / fps, {"rep": float(v)})
        if ev:
            reps.append(RepAnnotation(start_frame=int(round(ev.t_start * fps)), end_frame=int(round(ev.t_end * fps)),
                                      peak_frame=int(round(ev.t_peak * fps)), quality="good" if ev.counted else "partial"))
    return reps


def to_tasks(store: DatasetStore, registry: ExerciseRegistry, video_url_for=lambda cid: f"/data/upload/{cid}.mp4") -> List[Dict]:
    tasks = []
    for m in store.effective():
        frames, _ = store.load_clip(m.clip_id)
        result = []
        if m.exercise_id:
            result.append({"from_name": "exercise", "to_name": "video", "type": "choices", "value": {"choices": [m.exercise_id]}})
            for r in prelabel_reps(frames, m.exercise_id, registry, m.fps):
                result.append({"from_name": "reps", "to_name": "video", "type": "timelinelabels",
                               "value": {"ranges": [{"start": r.start_frame, "end": r.end_frame}],
                                         "timelinelabels": [f"{r.quality}_rep"]}})
        tasks.append({"data": {"video": video_url_for(m.clip_id), "fps": m.fps, "clip_id": m.clip_id,
                               "subject_id": m.subject_id},
                      "predictions": [{"model_version": "engine-prelabel", "result": result}] if result else []})
    return tasks


def from_export(export: List[Dict], annotator_default: str = "") -> List[ClipAnnotation]:
    """Label Studio JSON export -> ClipAnnotation list (only submitted, non-cancelled annotations)."""
    out = []
    for task in export:
        cid = (task.get("data") or {}).get("clip_id")
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not cid or not anns:
            continue
        a = anns[-1]
        ca = ClipAnnotation(clip_id=cid, annotator=str((a.get("completed_by") or annotator_default) or ""), label_source="human_verified")
        for r in a.get("result", []):
            v, t, name = r.get("value", {}), r.get("type"), r.get("from_name")
            if t == "choices" and name == "exercise":
                c = v["choices"][0]
                ca.exercise_id = None if c == "other_unsupported" else c
            elif t == "choices" and name == "view":
                ca.view = v["choices"][0]
            elif t == "choices" and name == "camera_ok":
                ca.camera_ok = v["choices"][0] == "yes"
            elif t == "choices" and name == "issues":
                issues = [i for i in v["choices"] if i in FORM_ISSUES]
                if ca.reps:
                    ca.reps[-1].issues = issues
                else:
                    ca.notes += "issues:" + ",".join(issues)
            elif t == "timelinelabels":
                for rng in v.get("ranges", []):
                    q = (v.get("timelinelabels") or ["good_rep"])[0].replace("_rep", "")
                    ca.reps.append(RepAnnotation(start_frame=int(rng["start"]), end_frame=int(rng["end"]),
                                                 quality=q if q in ("good", "partial", "bad") else "good"))
            elif t == "textarea":
                ca.notes += " ".join(v.get("text", []))
        out.append(ca)
    return out

import json
import os

import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from ml.data.extract import IngestError, ingest_video
from ml.data.labelstudio import from_export, label_config, prelabel_reps, to_tasks
from ml.data.report import report
from ml.data.schema import ClipAnnotation, ClipMeta, RepAnnotation
from ml.data.split import LeakageError, assert_no_leakage, subject_disjoint_split
from ml.data.store import DatasetStore, frames_to_landmarks, landmarks_to_frames
from ml.tracking import Run
from motion_engine.pose import (BenchResult, PoseEstimator, benchmark, from_coco17, from_mediapipe33, make_estimator, PoseUnavailable)
from motion_engine.synthetic import synth_sequence

REG = ExerciseRegistry.load()


def make_store(tmp_path, n_subjects=6, exercises=("squat", "bicep_curl"), trusted=True):
    st = DatasetStore(tmp_path / "ds")
    k = 0
    for s in range(n_subjects):
        for ex in exercises:
            fr, ts = synth_sequence(ex, reps=3, seed=k, scale=0.8 + 0.05 * s)
            st.add_clip(fr, ts, ClipMeta(clip_id=f"c{k}", subject_id=f"s{s}", exercise_id=ex, view="side" if k % 2 else "front",
                                        label_source="expert" if trusted else "rule", pose_model="synthetic"))
            k += 1
    return st


# ---------------- store / versioning ----------------
def test_store_roundtrip_and_immutability(tmp_path):
    st = make_store(tmp_path, 1)
    fr, ts = st.load_clip("c0")
    orig, _ = synth_sequence("squat", reps=3, seed=0, scale=0.8)
    assert len(fr) == len(orig) and fr[5]["left_knee"][0] == pytest.approx(orig[5]["left_knee"][0], abs=1e-5)
    with pytest.raises(ValueError, match="already exists"):
        st.add_clip(fr, ts, ClipMeta(clip_id="c0", subject_id="s0"))
    assert landmarks_to_frames(frames_to_landmarks([{"nose": [0.1, 0.2, 0.9]}]))[0]["nose"][0] == pytest.approx(0.1)


def test_consent_is_enforced_for_user_clips(tmp_path):
    st = DatasetStore(tmp_path)
    fr, ts = synth_sequence("squat", reps=1)
    with pytest.raises(ValueError, match="consent"):
        st.add_clip(fr, ts, ClipMeta(clip_id="u1", subject_id="x", source="consented_user"))
    st.add_clip(fr, ts, ClipMeta(clip_id="u2", subject_id="x", source="consented_user", consent=True))


def test_version_id_tracks_data_and_labels_and_frozen_versions_are_immutable(tmp_path):
    st = make_store(tmp_path, 2)
    v1 = st.version_id()
    assert st.version_id() == v1                              # deterministic
    st.annotate(ClipAnnotation(clip_id="c0", exercise_id="squat", label_source="human_verified", reps=[RepAnnotation(start_frame=0, end_frame=60)]))
    v2 = st.version_id()
    assert v2 != v1                                           # a relabel is a new dataset version
    rec = st.freeze("v1")
    assert rec["id"] == v2 and len(rec["clips"]) == 4
    with pytest.raises(FileExistsError):
        st.freeze("v1")
    with pytest.raises(KeyError):
        st.annotate(ClipAnnotation(clip_id="nope"))


def test_annotation_upgrades_label_provenance(tmp_path):
    st = make_store(tmp_path, 1, trusted=False)
    assert {m.label_source for m in st.effective()} == {"rule"}
    st.annotate(ClipAnnotation(clip_id="c0", exercise_id="squat", label_source="expert"))
    assert {m.clip_id: m.label_source for m in st.effective()}["c0"] == "expert"


# ---------------- subject-disjoint splits ----------------
def test_split_is_subject_disjoint_and_covers_everything(tmp_path):
    metas = make_store(tmp_path, 10).effective()
    sp = subject_disjoint_split(metas, seed=1)
    assert sorted(sum(sp.values(), [])) == sorted(m.clip_id for m in metas)
    subj = {m.clip_id: m.subject_id for m in metas}
    sets = {k: {subj[c] for c in v} for k, v in sp.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert len(sp["train"]) > len(sp["val"]) and len(sp["test"]) > 0
    ex = {m.clip_id: m.exercise_id for m in metas}
    assert {ex[c] for c in sp["test"]} == {"squat", "bicep_curl"}      # stratified: no class missing from test


def test_leakage_assertion_catches_shared_subjects(tmp_path):
    metas = make_store(tmp_path, 3).effective()
    bad = {"train": ["c0", "c2", "c3", "c4", "c5"], "val": ["c1"], "test": []}     # c0 and c1 are both subject s0
    with pytest.raises(LeakageError, match="s0"):
        assert_no_leakage(metas, bad)
    with pytest.raises(LeakageError, match="every clip"):
        assert_no_leakage(metas, {"train": ["c0", "c1"], "val": [], "test": []})


def test_split_is_reproducible(tmp_path):
    metas = make_store(tmp_path, 8).effective()
    assert subject_disjoint_split(metas, seed=3) == subject_disjoint_split(metas, seed=3)


# ---------------- report ----------------
def test_report_warns_about_what_makes_a_dataset_untrustworthy(tmp_path):
    r = report(make_store(tmp_path, 3, trusted=False).effective())
    w = " ".join(r["warnings"])
    assert not r["gate_ready"] and "not human-verified" in w and "subjects" in w and "only 3 clips" in w
    assert r["per_exercise"] == {"squat": 3, "bicep_curl": 3} and r["trusted_fraction"] == 0.0 and r["pose_models"] == ["synthetic"]


def test_report_gate_ready_only_when_big_trusted_diverse(tmp_path):
    st = DatasetStore(tmp_path / "big")
    k = 0
    for s in range(9):
        for rep in range(4):
            fr, ts = synth_sequence("squat", reps=1, seed=k)
            st.add_clip(fr, ts, ClipMeta(clip_id=f"b{k}", subject_id=f"s{s}", exercise_id="squat", view=["front", "side", "45deg"][k % 3], label_source="expert"))
            k += 1
    r = report(st.effective())
    assert r["warnings"] == [] or all("clips" in x for x in r["warnings"])        # only the 30-clip floor may remain
    assert r["subjects_per_exercise"]["squat"] == 9 and r["trusted_fraction"] == 1.0


# ---------------- pose estimators ----------------
def test_coco17_and_mediapipe_conversion():
    k = np.zeros((17, 2)); k[5] = (50, 20); k[16] = (10, 90)
    f = from_coco17(k, np.full(17, 0.9), width=100, height=100)
    assert f["left_shoulder"] == [0.5, 0.2, 0.9] and f["right_ankle"][1] == 0.9 and len(f) == 14
    assert f["_image"] == [100, 100]                     # frame contract: source image size travels with the frame
    assert from_coco17(k / 100, np.full(17, 0.9), image_size=(640, 480))["_image"] == [640, 480]

    class P:
        def __init__(s, x, y): s.x, s.y, s.visibility = x, y, 0.8
    lm = [P(i / 40, i / 40) for i in range(33)]
    f = from_mediapipe33(lm)
    assert f["left_shoulder"][0] == pytest.approx(11 / 40) and f["right_ankle"][0] == pytest.approx(28 / 40)
    assert "_image" not in f and from_mediapipe33(lm, (1280, 720))["_image"] == [1280, 720]


def test_unavailable_estimators_fail_loudly_not_silently():
    for name in ("mediapipe", "yolo-pose", "rtmpose"):
        try:
            make_estimator(name)
        except PoseUnavailable as e:
            assert str(e)
    with pytest.raises(KeyError):
        make_estimator("nope")


class Fake(PoseEstimator):
    def __init__(self, name, truth, noise=0.0, drop=0.0, seed=0):
        self.name, self.truth, self.noise, self.drop = name, truth, noise, drop
        self.rng = np.random.default_rng(seed)

    def detect(self, img, t_ms):
        if self.rng.random() < self.drop:
            return None
        i = int(img[0, 0, 0])
        return {k: [v[0] + self.rng.normal(0, self.noise), v[1] + self.rng.normal(0, self.noise), v[2]] for k, v in self.truth[i].items()}


def test_benchmark_ranks_estimators_on_jitter_accuracy_and_dropouts():
    truth, _ = synth_sequence("squat", reps=3, noise=0.0)
    samples = lambda: ((np.full((4, 4, 3), i % 256, np.uint8), truth[i]) for i in range(len(truth)))
    good = benchmark(Fake("good", truth, noise=0.001), samples())
    noisy = benchmark(Fake("noisy", truth, noise=0.02), samples())
    flaky = benchmark(Fake("flaky", truth, noise=0.001, drop=0.3), samples())
    assert isinstance(good, BenchResult) and good.detection_rate == 1.0 and good.pck > 0.95
    assert noisy.jitter > 5 * good.jitter and noisy.error > 5 * good.error and noisy.pck < good.pck
    assert 0.55 < flaky.detection_rate < 0.85
    assert good.fps > 0 and good.latency_ms_p95 >= good.latency_ms_p50


# ---------------- video ingestion (real mp4 through cv2) ----------------
def write_video(path, n=45, fps=30):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 encoder in this OpenCV build")
    for i in range(n):
        w.write(np.full((48, 64, 3), 10, np.uint8))
    w.release()


class VideoFake(PoseEstimator):
    name, version = "fake", "1"

    def __init__(self, fail_every=0):
        self.i, self.fail_every = 0, fail_every
        self.truth, _ = synth_sequence("squat", reps=3, noise=0.0)

    def detect(self, img, t_ms):
        self.i += 1
        if self.fail_every and self.i % self.fail_every == 0:
            return None
        return self.truth[(self.i - 1) % len(self.truth)]


def test_ingest_video_extracts_pose_and_deletes_raw_video(tmp_path):
    v = tmp_path / "in.mp4"; write_video(v)
    st = DatasetStore(tmp_path / "ds")
    m = ingest_video(st, str(v), ClipMeta(clip_id="v1", subject_id="s1", exercise_id="squat"), VideoFake(fail_every=5))
    assert not v.exists()                                     # privacy: raw video gone, landmarks kept
    assert m.n_frames >= 40 and m.pose_model == "fake:1" and 0.7 < m.quality < 0.9
    assert len(st.load_clip("v1")[0]) == m.n_frames


def test_ingest_rejects_bad_clips_but_still_deletes_the_video(tmp_path):
    v = tmp_path / "bad.mp4"; write_video(v)
    st = DatasetStore(tmp_path / "ds")
    with pytest.raises(IngestError, match="pose found"):
        ingest_video(st, str(v), ClipMeta(clip_id="x", subject_id="s"), VideoFake(fail_every=1))
    assert not v.exists() and st.meta_index() == {}
    v2 = tmp_path / "short.mp4"; write_video(v2, n=10)
    with pytest.raises(IngestError, match="shorter"):
        ingest_video(st, str(v2), ClipMeta(clip_id="y", subject_id="s"), VideoFake())
    with pytest.raises(IngestError, match="open"):
        ingest_video(st, str(tmp_path / "missing.mp4"), ClipMeta(clip_id="z", subject_id="s"), VideoFake())


# ---------------- Label Studio ----------------
def test_prelabel_finds_the_reps_the_engine_sees():
    fr, _ = synth_sequence("bicep_curl", reps=4)
    reps = prelabel_reps(fr, "bicep_curl", REG, 30.0)
    assert len(reps) == 4 and all(r.start_frame < r.peak_frame < r.end_frame and r.quality == "good" for r in reps)
    assert prelabel_reps(fr, "not_an_exercise", REG, 30.0) == []


def test_label_config_lists_every_exercise_plus_unsupported():
    cfg = label_config(REG)
    assert all(f'value="{d.id}"' in cfg for d in REG.all()) and "other_unsupported" in cfg and "TimelineLabels" in cfg


def test_label_studio_roundtrip(tmp_path):
    st = make_store(tmp_path, 1)
    tasks = to_tasks(st, REG)
    assert len(tasks) == 2 and tasks[0]["data"]["clip_id"] == "c0"
    pre = tasks[0]["predictions"][0]["result"]
    assert pre[0]["value"]["choices"] == ["squat"] and sum(r["type"] == "timelinelabels" for r in pre) == 3
    export = [{"data": {"clip_id": "c0"}, "annotations": [
        {"was_cancelled": True, "result": []},
        {"completed_by": 7, "result": [
            {"type": "choices", "from_name": "exercise", "value": {"choices": ["squat"]}},
            {"type": "choices", "from_name": "view", "value": {"choices": ["side"]}},
            {"type": "timelinelabels", "from_name": "reps", "value": {"ranges": [{"start": 10, "end": 80}], "timelinelabels": ["partial_rep"]}},
            {"type": "timelinelabels", "from_name": "reps", "value": {"ranges": [{"start": 90, "end": 160}], "timelinelabels": ["good_rep"]}},
            {"type": "choices", "from_name": "issues", "value": {"choices": ["shallow_depth", "not_a_real_issue"]}},
            {"type": "choices", "from_name": "camera_ok", "value": {"choices": ["no"]}},
            {"type": "something_new", "value": {}}]}]},
        {"data": {"clip_id": "c1"}, "annotations": []}, {"data": {}, "annotations": [{"result": []}]}]
    anns = from_export(export)
    assert len(anns) == 1
    a = anns[0]
    assert a.exercise_id == "squat" and a.view == "side" and not a.camera_ok and a.label_source == "human_verified"
    assert [(r.quality, r.start_frame) for r in a.reps] == [("partial", 10), ("good", 90)] and a.reps[-1].issues == ["shallow_depth"]
    st.annotate(a)
    assert st.annotations()["c0"].reps[0].quality == "partial"
    unk = from_export([{"data": {"clip_id": "c1"}, "annotations": [{"result": [{"type": "choices", "from_name": "exercise", "value": {"choices": ["other_unsupported"]}}]}]}])[0]
    assert unk.exercise_id is None


# ---------------- tracking ----------------
def test_run_log_records_lineage(tmp_path):
    r = Run("t", {"lr": 0.1}, root=str(tmp_path), dataset_version="ds-abc")
    r.log(1, loss=0.5)
    rec = r.finish({"acc": 0.9}, {"model": "m.onnx"})
    saved = json.loads(open(r.path).read())
    assert saved["dataset_version"] == "ds-abc" and saved["params"] == {"lr": 0.1} and saved["history"][0]["loss"] == 0.5
    assert "code_commit" in saved and saved["metrics"]["acc"] == 0.9


def test_dvc_and_train_config_are_valid():
    import yaml
    d = yaml.safe_load(open("dvc.yaml"))
    assert set(d["stages"]) == {"report", "encoder"} and yaml.safe_load(open("train.yaml"))["window"] == 90

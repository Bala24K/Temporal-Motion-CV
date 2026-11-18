"""python -m ml.data.cli <command> ...
  ingest   --store DIR --video F --subject S [--exercise E --view V --consent --pose mediapipe]
  report   --store DIR
  split    --store DIR [--seed N]
  freeze   --store DIR --name v1
  ls-config / ls-export --store DIR / ls-import --store DIR --file export.json
"""
import argparse
import json
import sys
import uuid

from exercise_system.registry import ExerciseRegistry
from motion_engine.pose import make_estimator
from .extract import ingest_video
from .labelstudio import from_export, label_config, to_tasks
from .report import report
from .schema import ClipMeta
from .split import subject_disjoint_split
from .store import DatasetStore


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ml.data.cli")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "report", "split", "freeze", "ls-config", "ls-export", "ls-import"):
        p = sp.add_parser(name)
        p.add_argument("--store", default="data/dataset")
        if name == "ingest":
            p.add_argument("--video", required=True); p.add_argument("--subject", required=True)
            p.add_argument("--exercise"); p.add_argument("--view", default="unknown"); p.add_argument("--consent", action="store_true")
            p.add_argument("--source", default="internal"); p.add_argument("--pose", default="mediapipe"); p.add_argument("--keep-video", action="store_true")
        if name == "split":
            p.add_argument("--seed", type=int, default=0)
        if name == "freeze":
            p.add_argument("--name", required=True)
        if name == "ls-import":
            p.add_argument("--file", required=True)
    a = ap.parse_args(argv)
    store, reg = DatasetStore(a.store), ExerciseRegistry.load()
    if a.cmd == "ingest":
        with make_estimator(a.pose) as est:
            m = ingest_video(store, a.video, ClipMeta(clip_id=uuid.uuid4().hex[:12], subject_id=a.subject, exercise_id=a.exercise,
                                                      view=a.view, consent=a.consent, source=a.source,
                                                      label_source="user" if a.exercise else "model"), est, delete_video=not a.keep_video)
        print(m.model_dump_json())
    elif a.cmd == "report":
        print(json.dumps(report(store.effective()), indent=2))
    elif a.cmd == "split":
        print(json.dumps({k: len(v) for k, v in subject_disjoint_split(store.effective(), seed=a.seed).items()}))
    elif a.cmd == "freeze":
        print(json.dumps(store.freeze(a.name)))
    elif a.cmd == "ls-config":
        print(label_config(reg))
    elif a.cmd == "ls-export":
        print(json.dumps(to_tasks(store, reg), indent=2))
    elif a.cmd == "ls-import":
        anns = from_export(json.load(open(a.file)))
        for x in anns:
            store.annotate(x)
        print(f"imported {len(anns)} annotations; dataset version {store.version_id()}")


if __name__ == "__main__":
    main(sys.argv[1:])

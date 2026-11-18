"""Validate form rules against expert-labelled reps (docs/audit/TEST_AND_EVALUATION_PLAN.md section 6).

    python -m ml.form_eval DATA_DIR [--exercise shoulder_press] [--json report.json] [--apply --dataset NAME]

DATA_DIR holds landmark recordings (the live screen's "Record landmarks" JSON Lines, `*.jsonl` or `*.jsonl.gz`) and,
next to each, one label file per reviewer: `<stem>.labels.json` or `<stem>.labels.<reviewer>.json`:

    {"exercise": "shoulder_press", "reviewer": "coach-a",
     "reps": [{"t0": 1.2, "t1": 3.5, "rules": {"lockout": "fail", "torso_upright": "pass", "arm_sync": "na"}}, ...]}

Each rule label is "pass", "fail" or "na" (not judgeable from this view). With two or more reviewers, a rep-rule pair
is used only where every reviewer gave pass/fail and they agree; Cohen's kappa is reported per rule (first two
reviewers). Reps the engine counted are matched to labelled reps by time overlap.

Per rule: precision and recall of the warning, false warnings per 100 clean reps (a clean rep = every judged rule
labelled "pass"), and the share of judgeable reps the engine couldn't evaluate. Recommendation (targets from the plan):
  validated  precision >= 0.8 and <= 5 false warnings / 100 clean reps
  info_only  misses a target but precision >= 0.5 and <= 15 false / 100 clean  (shown as a tip, not scored)
  disabled   otherwise
  insufficient  fewer than --min-reps judged reps, --min-faulty faulty reps or --min-clean clean reps (no change)

--apply writes the recommendation and its numbers into each rule's `validation` block in
exercise_system/definitions/<exercise>.json (only rules with sufficient data).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from exercise_system.registry import ExerciseRegistry
from ml.replay import read_recording, replay

TARGET_PRECISION, TARGET_FALSE_PER_100 = 0.8, 5.0
INFO_PRECISION, INFO_FALSE_PER_100 = 0.5, 15.0
LABELS = ("pass", "fail", "na")
DEFS_DIR = Path(__file__).resolve().parent.parent / "exercise_system" / "definitions"


# ------------------------------------------------------------------------------------------- loading
def label_files(rec: Path) -> List[Path]:
    stem = rec.name[:-len(".jsonl.gz")] if rec.name.endswith(".jsonl.gz") else rec.stem
    return sorted(rec.parent.glob(f"{stem}.labels*.json"))


def load_labels(paths: List[Path]) -> Tuple[Optional[str], List[dict]]:
    """Reads every reviewer's file. Returns (exercise, reviewers) with reviewers = [{"reviewer", "reps"}]."""
    exercise, out = None, []
    for i, p in enumerate(paths):
        d = json.loads(p.read_text(encoding="utf-8"))
        exercise = exercise or d.get("exercise")
        reps = []
        for r in d.get("reps", []):
            rules = {k: str(v).lower() for k, v in (r.get("rules") or {}).items()}
            bad = [v for v in rules.values() if v not in LABELS]
            if bad:
                raise ValueError(f"{p.name}: rule labels must be one of {LABELS}, got {bad}")
            reps.append({"t0": float(r["t0"]), "t1": float(r["t1"]), "rules": rules})
        out.append({"reviewer": d.get("reviewer") or f"reviewer{i + 1}", "reps": sorted(reps, key=lambda x: x["t0"])})
    return exercise, out


def overlap(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match(engine: List[dict], labelled: List[dict], min_iou: float = 0.3) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching by temporal IoU (highest first)."""
    pairs = sorted(((overlap(e["t_start"], e["t_end"], l["t0"], l["t1"]), i, j)
                    for i, e in enumerate(engine) for j, l in enumerate(labelled)), reverse=True)
    used_e, used_l, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i not in used_e and j not in used_l:
            used_e.add(i); used_l.add(j); out.append((i, j))
    return out


def consensus(reviewers: List[dict]) -> Tuple[List[dict], Dict[str, List[Tuple[str, str]]]]:
    """Labelled reps on the first reviewer's timeline; other reviewers' reps are matched to them by overlap.
    Returns (reps with consensus rule labels, per-rule (label_a, label_b) pairs for kappa)."""
    base = reviewers[0]["reps"]
    as_engine = [{"t_start": r["t0"], "t_end": r["t1"]} for r in base]
    others = [{i: rv["reps"][j] for i, j in match(as_engine, rv["reps"])} for rv in reviewers[1:]]
    pairs_for_kappa: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    out = []
    for k, rep in enumerate(base):
        rules = {}
        for rule, lab in rep["rules"].items():
            labs = [lab] + [o[k]["rules"].get(rule, "na") if k in o else "na" for o in others]
            if len(labs) >= 2:
                pairs_for_kappa[rule].append((labs[0], labs[1]))
            rules[rule] = lab if lab != "na" and all(x == lab for x in labs) else "na"
        out.append({**rep, "rules": rules})
    return out, pairs_for_kappa


def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    pairs = [(a, b) for a, b in pairs if a != "na" and b != "na"]
    if len(pairs) < 5:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "fail" for a, _ in pairs) / n
    pb = sum(b == "fail" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1 else round((po - pe) / (1 - pe), 3)


def wilson_low(k: int, n: int, z: float = 1.96) -> Optional[float]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    return round((c - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d, 3)


# ------------------------------------------------------------------------------------------- scoring
def engine_verdicts(rep: dict) -> Dict[str, Optional[bool]]:
    """rule -> True (warned) / False (passed) / None (not evaluated). A partial rep counts as a rom warning."""
    out: Dict[str, Optional[bool]] = {}
    for e in rep.get("evidence", []):
        rid = "rom" if e["rule_id"] == "partial" else e["rule_id"]
        out[rid] = (not e["passed"]) if e.get("evaluated", True) else None
    return out


def score(observations: List[dict], rules: List[str], min_reps: int, min_faulty: int, min_clean: int) -> Dict[str, dict]:
    """observations: [{"labels": {rule: pass|fail|na}, "engine": {rule: True|False|None}}] for matched reps."""
    clean = [o for o in observations if any(v != "na" for v in o["labels"].values())
             and all(v in ("pass", "na") for v in o["labels"].values())]
    report = {}
    for rule in rules:
        tp = fp = fn = tn = not_eval = 0
        for o in observations:
            lab = o["labels"].get(rule, "na")
            if lab == "na":
                continue
            warned = o["engine"].get(rule)
            if warned is None:
                not_eval += 1
                continue
            if warned and lab == "fail": tp += 1
            elif warned: fp += 1
            elif lab == "fail": fn += 1
            else: tn += 1
        judged = tp + fp + fn + tn
        faulty = tp + fn
        false_on_clean = sum(1 for o in clean if o["engine"].get(rule) is True)
        clean_judged = sum(1 for o in clean if o["engine"].get(rule) is not None)
        precision = round(tp / (tp + fp), 3) if tp + fp else None
        recall = round(tp / faulty, 3) if faulty else None
        fp100 = round(100 * false_on_clean / clean_judged, 2) if clean_judged else None
        if judged < min_reps or faulty < min_faulty or clean_judged < min_clean:
            rec = "insufficient"
        elif (precision or 0) >= TARGET_PRECISION and fp100 is not None and fp100 <= TARGET_FALSE_PER_100:
            rec = "validated"
        elif (precision or 0) >= INFO_PRECISION and fp100 is not None and fp100 <= INFO_FALSE_PER_100:
            rec = "info_only"
        else:
            rec = "disabled"
        report[rule] = {"judged": judged, "faulty": faulty, "clean_reps": clean_judged, "tp": tp, "fp": fp, "fn": fn,
                        "tn": tn, "precision": precision, "precision_low95": wilson_low(tp, tp + fp), "recall": recall,
                        "false_per_100_clean": fp100, "not_evaluated": not_eval,
                        "not_evaluated_rate": round(not_eval / (judged + not_eval), 3) if judged + not_eval else None,
                        "recommendation": rec}
    return report


def evaluate(data_dir: Path, exercise: Optional[str] = None, min_reps: int = 30, min_faulty: int = 9,
             min_clean: int = 30, min_iou: float = 0.3) -> Dict[str, dict]:
    reg = ExerciseRegistry.load()
    recs = sorted([*data_dir.rglob("*.jsonl"), *data_dir.rglob("*.jsonl.gz")])
    per_ex: Dict[str, dict] = {}
    for rec in recs:
        lfs = label_files(rec)
        if not lfs:
            continue
        meta, frames = read_recording(rec)
        lab_ex, reviewers = load_labels(lfs)
        ex = meta.get("exercise") or lab_ex
        if exercise and ex != exercise:
            continue
        defn = reg.get(ex) if ex else None
        if defn is None:
            print(f"skip {rec.name}: unknown exercise {ex!r}", file=sys.stderr)
            continue
        res = replay(frames, defn)
        engine_reps = res["summary"]["reps_detail"]          # per-rep records with times and evidence
        labelled, kappa_pairs = consensus(reviewers)
        acc = per_ex.setdefault(ex, {"observations": [], "recordings": 0, "labelled_reps": 0, "engine_reps": 0,
                                     "matched": 0, "kappa_pairs": defaultdict(list), "reviewers": set()})
        acc["recordings"] += 1
        acc["labelled_reps"] += len(labelled)
        acc["engine_reps"] += len(engine_reps)
        acc["reviewers"].update(r["reviewer"] for r in reviewers)
        for rule, ps in kappa_pairs.items():
            acc["kappa_pairs"][rule].extend(ps)
        for i, j in match(engine_reps, labelled, min_iou):
            acc["matched"] += 1
            acc["observations"].append({"labels": labelled[j]["rules"], "engine": engine_verdicts(engine_reps[i])})
    out = {}
    for ex, acc in per_ex.items():
        defn = reg.get(ex)
        rules = [r.id for r in defn.form_rules]
        rep = score(acc["observations"], rules, min_reps, min_faulty, min_clean)
        for rule in rules:
            rep[rule]["kappa"] = cohen_kappa(acc["kappa_pairs"].get(rule, []))
        out[ex] = {"recordings": acc["recordings"], "reviewers": sorted(acc["reviewers"]),
                   "labelled_reps": acc["labelled_reps"], "engine_reps": acc["engine_reps"], "matched_reps": acc["matched"],
                   "missed_reps": acc["labelled_reps"] - acc["matched"], "extra_reps": acc["engine_reps"] - acc["matched"],
                   "rules": rep}
    return out


def apply(report: Dict[str, dict], dataset: str, defs_dir: Path = DEFS_DIR) -> List[str]:
    """Write recommendations into the definition files. Returns the changed 'exercise.rule' ids."""
    changed, today = [], _dt.date.today().isoformat()
    for ex, r in report.items():
        p = defs_dir / f"{ex}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for rule in d.get("form_rules", []):
            m = r["rules"].get(rule["id"])
            if not m or m["recommendation"] == "insufficient":
                continue
            rule["validation"] = {"status": m["recommendation"], "precision": m["precision"], "recall": m["recall"],
                                  "false_per_100_clean": m["false_per_100_clean"], "faulty_reps": m["faulty"],
                                  "clean_reps": m["clean_reps"], "dataset": dataset, "date": today}
            changed.append(f"{ex}.{rule['id']}")
        p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--exercise")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--apply", action="store_true", help="write recommendations into the exercise definitions")
    ap.add_argument("--dataset", help="dataset name recorded with --apply (required with --apply)")
    ap.add_argument("--min-reps", type=int, default=30)
    ap.add_argument("--min-faulty", type=int, default=9)
    ap.add_argument("--min-clean", type=int, default=30)
    a = ap.parse_args(argv)
    if a.apply and not a.dataset:
        ap.error("--apply needs --dataset NAME")
    report = evaluate(Path(a.data_dir), a.exercise, a.min_reps, a.min_faulty, a.min_clean)
    if not report:
        print("no labelled recordings found", file=sys.stderr)
        return 2
    for ex, r in report.items():
        print(f"{ex}: {r['recordings']} recordings, {r['labelled_reps']} labelled reps, {r['matched_reps']} matched, "
              f"{r['missed_reps']} missed, {r['extra_reps']} extra")
        for rule, m in r["rules"].items():
            print(f"  {rule:16s} {m['recommendation']:12s} precision={m['precision']} recall={m['recall']} "
                  f"false/100clean={m['false_per_100_clean']} judged={m['judged']} faulty={m['faulty']} kappa={m['kappa']}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if a.apply:
        print("updated:", ", ".join(apply(report, a.dataset)) or "nothing (insufficient data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

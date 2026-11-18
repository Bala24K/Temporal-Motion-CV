"""Recognizer selection. Modes (env RECOGNIZER_MODE):
  rules      - rule-based only (default)
  shadow     - serve rules; also run the learned ONNX model and record agreement (safe rollout)
  consensus  - commit to an exercise only when BOTH agree on the group; otherwise ask the user to confirm
  learned    - serve the learned model (a registered production model is expected; see promotion gate)
A broken/missing model never takes recognition down: we fall back to rules and count the error.
"""
from __future__ import annotations

import os
from typing import Optional

from motion_engine.learned import LearnedRecognizer
from motion_engine.recognition import Recognition, recognize


class RecognizerService:
    def __init__(self, registry, metrics, onnx_path: Optional[str] = None, mode: Optional[str] = None):
        self.registry, self.metrics = registry, metrics
        self.mode = mode or os.getenv("RECOGNIZER_MODE", "rules")
        self.learned: Optional[LearnedRecognizer] = None
        path = onnx_path or os.getenv("RECOGNIZER_ONNX")
        if path:
            try:
                self.learned = LearnedRecognizer(path)
            except Exception:
                self.metrics.inc("learned_recognizer_load_errors_total")
                self.mode = "rules"
        elif self.mode != "rules":
            self.mode = "rules"

    def recognize(self, frames) -> Recognition:
        rules = recognize(frames, self.registry)
        if self.learned is None or self.mode == "rules" or rules.action == "wait":
            return rules
        try:
            ml = self.learned.recognize(frames)
        except Exception:
            self.metrics.inc("learned_recognizer_errors_total")
            return rules
        agree = (ml.top is not None and rules.top is not None and set(ml.top.members) == set(rules.top.members)) \
            or (ml.top is None and rules.top is None)
        self.metrics.inc("recognizer_agree_total" if agree else "recognizer_disagree_total")
        if self.mode == "shadow":
            return rules
        if self.mode == "learned":
            return ml
        if agree:
            return rules
        # disagreement: never analyse on a coin flip
        cands = [c for c in (rules.top, ml.top) if c]
        if not cands:
            return Recognition("unsupported", "fallback", 0.0, None, message=rules.message or ml.message)
        return Recognition("ambiguous", "confirm", min(rules.confidence, ml.confidence), cands[0],
                           (rules.candidates + ml.candidates)[:3], "Not sure which exercise this is - please confirm.")

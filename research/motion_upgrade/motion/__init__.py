"""
motion — learned spatial-temporal movement recognition.

The recognition core of the next version: skeleton sequences in their
natural (T, J, C) shape, an ST-GCN/transformer model over the body's
connectivity graph, and per-frame movement phases. The engineered
kinematics in `biomechanics/`, `body_model/` and `exercises/` are not
replaced — they become the interpretable branch that explains what the
learned model recognises.

Import policy: this module deliberately exposes only the torch-free
surface (graph, tensors, dataset, phase, augment, calibration). The model,
trainer and inference adapter import torch and are imported directly:

    from motion.model import MotionModelConfig, build_model
    from motion.recognizer import NeuralExerciseRecognizer

That keeps `import motion` usable for data preparation and evaluation on
machines without torch installed.
"""
from motion.graph import (
    BONES,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    build_adjacency,
    mirror_permutation,
)
from motion.phase import NUM_PHASES, PHASE_NAMES, derive_phase_labels, segment_reps_from_phases
from motion.tensors import SkeletonWindow, build_streams, build_window

__all__ = [
    "BONES", "JOINT_INDEX", "JOINT_NAMES", "NUM_JOINTS",
    "build_adjacency", "mirror_permutation",
    "NUM_PHASES", "PHASE_NAMES", "derive_phase_labels", "segment_reps_from_phases",
    "SkeletonWindow", "build_streams", "build_window",
]

"""
ONNX export and inference benchmarking.

Deliberately placed after the model, the training loop and the async
runtime rather than before them. Export freezes an architecture: every
subsequent change to the model means re-exporting, re-verifying numerical
agreement and re-benchmarking. Doing that while the architecture is still
moving costs more than the speedup is worth, and the async pipeline
already removes the reason most people reach for ONNX early — inference
blocking the camera.

The order that pays off:

    1. profile the real pipeline (`AsyncMotionPipeline.profile`)
    2. find the actual bottleneck
    3. if it is the recognition model, export and benchmark here
    4. only then consider FP16 / INT8

Step 2 matters. On this pipeline MediaPipe pose estimation is the usual
bottleneck, and a model that is 3x faster after quantisation moves the
end-to-end number by very little.
"""
from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from motion.model import MotionModelConfig
from motion.tensors import STREAM_NAMES, stream_channel_count

DEFAULT_OPSET = 17


@dataclass
class ExportReport:
    """What the export produced and whether it can be trusted."""
    path: str
    opset: int
    input_shape: Sequence[int]
    max_absolute_difference: float
    outputs_agree: bool
    torch_ms: float
    onnx_ms: float
    size_mb: float

    @property
    def speedup(self) -> float:
        return self.torch_ms / self.onnx_ms if self.onnx_ms > 0 else 0.0

    def summary(self) -> str:
        verdict = "agree" if self.outputs_agree else "DISAGREE"
        return (
            f"{self.path} ({self.size_mb:.1f} MB, opset {self.opset})\n"
            f"  outputs {verdict} (max abs diff {self.max_absolute_difference:.2e})\n"
            f"  torch {self.torch_ms:.1f} ms  onnx {self.onnx_ms:.1f} ms  "
            f"speedup {self.speedup:.2f}x"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path, "opset": self.opset,
            "input_shape": list(self.input_shape),
            "max_absolute_difference": self.max_absolute_difference,
            "outputs_agree": self.outputs_agree,
            "torch_ms": self.torch_ms, "onnx_ms": self.onnx_ms,
            "speedup": self.speedup, "size_mb": self.size_mb,
        }


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    window_frames: Optional[int] = None,
    opset: int = DEFAULT_OPSET,
    dynamic_time_axis: bool = True,
    tolerance: float = 1e-4,
    benchmark_runs: int = 30,
) -> ExportReport:
    """
    Export a trained checkpoint to ONNX, verify it numerically, and
    benchmark both runtimes.

    Verification is not optional here. A silently wrong export produces a
    model that runs faster and predicts differently, and the difference
    shows up as "the demo got worse after we optimised it" days later.
    The tolerance is checked against the logits, not the argmax: matching
    labels on a handful of random inputs proves very little.

    `dynamic_time_axis` keeps T free so the same file serves a 48-frame
    window now and a different one later without re-export. Batch is
    always dynamic.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = MotionModelConfig.from_dict(ckpt["config"])

    from motion.model import build_model

    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    frames = int(window_frames or ckpt.get("window_frames", 48))
    channels = stream_channel_count(tuple(ckpt.get("streams", STREAM_NAMES)),
                                    cfg.include_validity_channel)
    dummy = torch.randn(1, channels, frames, cfg.num_joints)

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "skeleton": {0: "batch"},
        "exercise_logits": {0: "batch"},
        "phase_logits": {0: "batch"},
    }
    if dynamic_time_axis:
        dynamic_axes["skeleton"][2] = "time"
        dynamic_axes["phase_logits"][1] = "time"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # The model returns a dict; ONNX export wants positional outputs, so
    # wrap it rather than changing the model's own signature — the dict
    # is the right interface for every other caller.
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, skeleton):
            out = self.inner(skeleton)
            return out["exercise_logits"], out["phase_logits"]

    torch.onnx.export(
        _ExportWrapper(model),
        dummy,
        output_path,
        input_names=["skeleton"],
        output_names=["exercise_logits", "phase_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    with torch.no_grad():
        reference = model(dummy)
    torch_logits = reference["exercise_logits"].numpy()
    torch_phases = reference["phase_logits"].numpy()

    session = OnnxMotionModel(output_path)
    onnx_logits, onnx_phases = session.run(dummy.numpy())

    max_diff = float(max(
        np.abs(torch_logits - onnx_logits).max(),
        np.abs(torch_phases - onnx_phases).max(),
    ))

    torch_ms = _benchmark(lambda: model(dummy), benchmark_runs)
    onnx_ms = _benchmark(lambda: session.run(dummy.numpy()), benchmark_runs)

    return ExportReport(
        path=output_path,
        opset=opset,
        input_shape=(1, channels, frames, cfg.num_joints),
        max_absolute_difference=max_diff,
        outputs_agree=max_diff <= tolerance,
        torch_ms=torch_ms,
        onnx_ms=onnx_ms,
        size_mb=os.path.getsize(output_path) / (1024 * 1024),
    )


def _benchmark(fn, runs: int, warmup: int = 5) -> float:
    """Median wall time in ms. Median, not mean: the first few runs and
    the occasional scheduler hiccup are not what is being measured."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


class OnnxMotionModel:
    """
    ONNX Runtime wrapper with the same call shape as the torch model, so
    `NeuralExerciseRecognizer` can hold either one.

    Threading matters here: ORT defaults to using every core, which on a
    laptop means the recognition thread starves the pose thread and the
    end-to-end latency gets worse despite the model itself getting
    faster. One intra-op thread by default; raise it only after measuring
    the whole pipeline, not the model alone.
    """

    def __init__(self, path: str, providers: Optional[Sequence[str]] = None,
                 intra_op_threads: int = 1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            path, sess_options=options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, skeleton: np.ndarray):
        """(N, C, T, J) float32 -> (exercise_logits, phase_logits)."""
        outputs = self.session.run(
            self.output_names, {self.input_name: np.ascontiguousarray(skeleton, dtype=np.float32)}
        )
        return outputs[0], outputs[1]

    def __call__(self, skeleton) -> Dict[str, np.ndarray]:
        """Dict-shaped output matching `MotionRecognitionNet.forward`, so
        the recogniser needs no branch for which backend it holds."""
        array = skeleton.numpy() if hasattr(skeleton, "numpy") else np.asarray(skeleton)
        exercise_logits, phase_logits = self.run(array)
        return {"exercise_logits": exercise_logits, "phase_logits": phase_logits}


def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Export a trained motion checkpoint to ONNX")
    p.add_argument("checkpoint")
    p.add_argument("--out", default="models/motion_stgcn.onnx")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--window-frames", type=int, default=None, dest="window_frames")
    p.add_argument("--static-time-axis", action="store_true", dest="static_time_axis")
    p.add_argument("--tolerance", type=float, default=1e-4)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    report = export_onnx(
        args.checkpoint, args.out,
        window_frames=args.window_frames,
        opset=args.opset,
        dynamic_time_axis=not args.static_time_axis,
        tolerance=args.tolerance,
    )
    print(report.summary())
    if not report.outputs_agree:
        raise SystemExit(
            "ONNX outputs do not match the torch model within tolerance — "
            "do not ship this file."
        )

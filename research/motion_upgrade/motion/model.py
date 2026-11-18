"""
The learned movement model: a spatial-temporal graph network over
skeleton sequences, with an attention stage on top and two heads.

    (N, C, T, J)
        |
   ST-GCN backbone            graph convolution over the skeleton
        |                     topology, interleaved with temporal
        |                     convolution
        v
   spatial pooling            joints -> one token per timestep
        |
   temporal transformer       which moments of the movement matter
        |
      +-+--------------------+
      |                      |
  attention pool         per-frame
      |                  phase logits
  exercise logits

WHY BOTH ST-GCN AND ATTENTION, IN ONE MODEL
    The plan lists these as consecutive milestones. Building them as one
    configurable model instead costs very little — the transformer stage
    is `use_temporal_transformer=False` away from being absent — and
    means there is a single model to train, tune, export and integrate
    rather than three. The ablation is then a config flag, which is a
    better experiment than three separately-tuned codebases anyway.

WHY EARLY STREAM FUSION IS THE DEFAULT
    Joint, bone and motion streams are concatenated on the channel axis
    and share one backbone. The literature's usual multi-stream result
    trains a separate network per stream and ensembles the softmax
    outputs, which multiplies both training time and inference cost —
    the wrong trade for a model that has to run live alongside pose
    estimation. `fusion="late"` is implemented for the comparison, but
    early fusion is what the runtime should ship.

SHAPE CONVENTION
    (N, C, T, J) throughout — batch, channels, time, joints. Adjacency is
    (K, J, J). Kept identical to the reference ST-GCN layout so published
    implementations can be read against this one.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less environments
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

    class _Missing:
        def __getattr__(self, item):
            raise ImportError(
                "PyTorch is required for motion.model. Install it with:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "The rest of the motion package (graph, tensors, dataset, phase, "
                "calibration) works without torch."
            )

    nn = _Missing()  # type: ignore
    F = _Missing()  # type: ignore

from motion.graph import NUM_JOINTS, build_adjacency
from motion.phase import NUM_PHASES
from motion.tensors import STREAM_NAMES, NUM_COORDS, stream_channel_count


@dataclass
class MotionModelConfig:
    """Everything that defines the architecture, serialised into every checkpoint."""
    num_classes: int = 4
    num_phases: int = NUM_PHASES
    num_joints: int = NUM_JOINTS
    streams: Tuple[str, ...] = STREAM_NAMES
    include_validity_channel: bool = True

    adjacency: str = "spatial"          # 'spatial' | 'uniform'
    adaptive_adjacency: bool = True
    fusion: str = "early"               # 'early' | 'late'

    # (out_channels, temporal_stride) per ST-GCN block.
    blocks: Tuple[Tuple[int, int], ...] = ((64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1))
    temporal_kernel: int = 9
    dropout: float = 0.3

    use_temporal_transformer: bool = True
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_mult: int = 2

    kinematic_dim: int = 0              # 0 disables the kinematic side-channel
    kinematic_hidden: int = 64

    @property
    def in_channels(self) -> int:
        return stream_channel_count(self.streams, self.include_validity_channel)

    @property
    def channels_per_stream(self) -> int:
        return NUM_COORDS

    def to_dict(self) -> Dict:
        return {
            "num_classes": self.num_classes, "num_phases": self.num_phases,
            "num_joints": self.num_joints, "streams": list(self.streams),
            "include_validity_channel": self.include_validity_channel,
            "adjacency": self.adjacency, "adaptive_adjacency": self.adaptive_adjacency,
            "fusion": self.fusion, "blocks": [list(b) for b in self.blocks],
            "temporal_kernel": self.temporal_kernel, "dropout": self.dropout,
            "use_temporal_transformer": self.use_temporal_transformer,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_ff_mult": self.transformer_ff_mult,
            "kinematic_dim": self.kinematic_dim, "kinematic_hidden": self.kinematic_hidden,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MotionModelConfig":
        d = dict(d)
        d["streams"] = tuple(d.get("streams", STREAM_NAMES))
        d["blocks"] = tuple(tuple(b) for b in d.get("blocks", cls.blocks))
        return cls(**d)


if _TORCH_AVAILABLE:

    class GraphConv(nn.Module):
        """
        Spatial graph convolution: for each adjacency partition k,
        transform features then propagate along A[k], and sum.

            y[n,:,t,i] = sum_k sum_j A[k,i,j] * (W_k x)[n,:,t,j]

        Implemented as a single 1x1 convolution producing K * C_out
        channels, so all partitions are computed in one GEMM.
        """

        def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray,
                     adaptive: bool = True):
            super().__init__()
            self.K = int(adjacency.shape[0])
            self.out_channels = out_channels
            self.register_buffer("A", torch.tensor(adjacency, dtype=torch.float32))
            self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)

            if adaptive:
                # Learnable residual on the adjacency. Initialised at zero
                # so the model starts as an exact anatomical ST-GCN and
                # only deviates where the data pays for it — random init
                # here throws away the topology prior on step one.
                self.A_residual = nn.Parameter(torch.zeros_like(self.A))
            else:
                self.A_residual = None

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            n, _, t, v = x.shape
            y = self.conv(x)
            y = y.view(n, self.K, self.out_channels, t, v)
            adj = self.A if self.A_residual is None else self.A + self.A_residual
            out = torch.einsum("nkctv,kvw->nctw", y, adj)
            return out.contiguous()

    class STGCNBlock(nn.Module):
        """Graph conv over joints, then temporal conv over time, with a residual."""

        def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray,
                     stride: int = 1, temporal_kernel: int = 9, dropout: float = 0.3,
                     adaptive: bool = True, residual: bool = True):
            super().__init__()
            if temporal_kernel % 2 != 1:
                raise ValueError("temporal_kernel must be odd so padding keeps the sequence length")
            pad = (temporal_kernel - 1) // 2

            self.gcn = GraphConv(in_channels, out_channels, adjacency, adaptive=adaptive)
            self.gcn_bn = nn.BatchNorm2d(out_channels)
            self.tcn = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1),
                          stride=(stride, 1), padding=(pad, 0)),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout),
            )
            self.relu = nn.ReLU(inplace=True)

            if not residual:
                self.residual = None
            elif in_channels == out_channels and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            res = 0.0 if self.residual is None else self.residual(x)
            y = self.relu(self.gcn_bn(self.gcn(x)))
            y = self.tcn(y)
            return self.relu(y + res)

    class STGCNBackbone(nn.Module):
        """Input normalisation plus a stack of ST-GCN blocks."""

        def __init__(self, in_channels: int, cfg: MotionModelConfig, adjacency: np.ndarray):
            super().__init__()
            # BatchNorm over (channels x joints): each joint's channel
            # distribution is standardised separately, because an ankle's
            # coordinate range genuinely differs from a nose's and a
            # single shared statistic washes out the smaller one.
            self.input_bn = nn.BatchNorm1d(in_channels * cfg.num_joints)

            layers: List[nn.Module] = []
            c_prev = in_channels
            for i, (c_out, stride) in enumerate(cfg.blocks):
                layers.append(
                    STGCNBlock(
                        c_prev, c_out, adjacency, stride=stride,
                        temporal_kernel=cfg.temporal_kernel, dropout=cfg.dropout,
                        adaptive=cfg.adaptive_adjacency,
                        residual=(i > 0),  # no residual on the first block: the
                                           # channel counts and the semantics of
                                           # raw coordinates vs features differ
                    )
                )
                c_prev = c_out
            self.blocks = nn.ModuleList(layers)
            self.out_channels = c_prev

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            n, c, t, v = x.shape
            x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
            x = self.input_bn(x)
            x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
            for block in self.blocks:
                x = block(x)
            return x

    class PositionalEncoding(nn.Module):
        """Standard sinusoidal encoding, added to the time tokens."""

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.pe[:, : x.shape[1], : x.shape[2]]

    class AttentionPool(nn.Module):
        """
        Learned query attention over time.

        Mean pooling gives the empty frames before a rep starts the same
        weight as the bottom of the squat. A learned query lets the model
        concentrate on the frames that actually discriminate — and the
        weights are returned, which makes them directly usable as a
        "what did the model look at" overlay in the UI.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            weights = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (N, T)
            pooled = torch.einsum("ntd,nt->nd", x, weights)
            return pooled, weights

    class MotionRecognitionNet(nn.Module):
        """
        The full model.

        forward() returns a dict rather than a tuple: there are three
        outputs of different shapes and a positional convention would be
        misread the first time someone adds a fourth.

            exercise_logits   (N, num_classes)
            phase_logits      (N, T_in, num_phases)
            attention         (N, T_feat)
            embedding         (N, D)
        """

        def __init__(self, cfg: MotionModelConfig):
            super().__init__()
            self.cfg = cfg
            adjacency = build_adjacency(cfg.adjacency)
            if adjacency.shape[1] != cfg.num_joints:
                raise ValueError(
                    f"adjacency is built for {adjacency.shape[1]} joints but config says {cfg.num_joints}"
                )

            if cfg.fusion == "early":
                self.backbones = nn.ModuleList([STGCNBackbone(cfg.in_channels, cfg, adjacency)])
                self.stream_slices = [slice(0, cfg.in_channels)]
                feat_dim = self.backbones[0].out_channels
            elif cfg.fusion == "late":
                self.backbones = nn.ModuleList()
                self.stream_slices = []
                start = 0
                for _ in cfg.streams:
                    stop = start + cfg.channels_per_stream
                    self.stream_slices.append(slice(start, stop))
                    self.backbones.append(STGCNBackbone(cfg.channels_per_stream, cfg, adjacency))
                    start = stop
                if cfg.include_validity_channel:
                    # The validity mask rides along with every stream
                    # rather than becoming a stream of its own; on its own
                    # it carries no movement information.
                    self.stream_slices = [slice(s.start, s.stop) for s in self.stream_slices]
                feat_dim = self.backbones[0].out_channels * len(self.backbones)
            else:
                raise ValueError(f"unknown fusion mode {cfg.fusion!r} (expected 'early' or 'late')")

            self.feat_dim = feat_dim

            if cfg.use_temporal_transformer:
                if feat_dim % cfg.transformer_heads != 0:
                    raise ValueError(
                        f"feature dim {feat_dim} must be divisible by transformer_heads "
                        f"{cfg.transformer_heads}"
                    )
                self.pos_encoding = PositionalEncoding(feat_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=feat_dim,
                    nhead=cfg.transformer_heads,
                    dim_feedforward=feat_dim * cfg.transformer_ff_mult,
                    dropout=cfg.dropout,
                    batch_first=True,
                    norm_first=True,   # pre-norm: trains without a warmup schedule,
                                       # which matters for short runs on small data
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
            else:
                self.pos_encoding = None
                self.transformer = None

            self.pool = AttentionPool(feat_dim)

            if cfg.kinematic_dim > 0:
                self.kinematic_mlp = nn.Sequential(
                    nn.Linear(cfg.kinematic_dim, cfg.kinematic_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.kinematic_hidden, feat_dim),
                )
                # Gated residual, zero-initialised: the kinematic channel
                # starts contributing nothing and has to earn its way in.
                # Without the gate it tends to dominate early training,
                # and the graph branch never learns anything.
                self.kinematic_gate = nn.Parameter(torch.zeros(1))
            else:
                self.kinematic_mlp = None
                self.kinematic_gate = None

            self.dropout = nn.Dropout(cfg.dropout)
            self.exercise_head = nn.Linear(feat_dim, cfg.num_classes)
            self.phase_head = nn.Linear(feat_dim, cfg.num_phases)

            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor",
                    kinematics: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
            if x.dim() != 4:
                raise ValueError(f"expected input of shape (N, C, T, J), got {tuple(x.shape)}")
            t_in = x.shape[2]

            if self.cfg.fusion == "early":
                feats = [self.backbones[0](x)]
            else:
                mask = x[:, -1:] if self.cfg.include_validity_channel else None
                feats = []
                for backbone, sl in zip(self.backbones, self.stream_slices):
                    part = x[:, sl]
                    if mask is not None:
                        part = part * mask
                    feats.append(backbone(part))

            # (N, C, T', J) -> (N, T', C) by averaging over joints. The
            # graph stage has already mixed information across joints;
            # keeping them separate into the transformer would multiply
            # the token count by 15 for little gain.
            pooled = [f.mean(dim=3).permute(0, 2, 1) for f in feats]
            h = torch.cat(pooled, dim=2)

            if self.transformer is not None:
                h = self.transformer(self.pos_encoding(h))

            phase_logits = self.phase_head(h)                       # (N, T', P)
            if phase_logits.shape[1] != t_in:
                # Temporal strides shrank the sequence; put it back on the
                # input clock so the phase head can be supervised against
                # per-frame labels and read by the rep segmenter.
                phase_logits = F.interpolate(
                    phase_logits.permute(0, 2, 1), size=t_in, mode="linear", align_corners=False
                ).permute(0, 2, 1)

            embedding, attention = self.pool(h)

            if self.kinematic_mlp is not None and kinematics is not None:
                embedding = embedding + torch.tanh(self.kinematic_gate) * self.kinematic_mlp(kinematics)

            exercise_logits = self.exercise_head(self.dropout(embedding))

            return {
                "exercise_logits": exercise_logits,
                "phase_logits": phase_logits,
                "attention": attention,
                "embedding": embedding,
            }

    def build_model(cfg: MotionModelConfig) -> "MotionRecognitionNet":
        return MotionRecognitionNet(cfg)

    def count_parameters(model: "nn.Module") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def selfcheck(verbose: bool = True) -> bool:
        """
        Shape and gradient smoke test.

        Run this immediately after installing torch, before starting a
        training run: it catches adjacency/joint mismatches, head shape
        errors and dead gradients in about a second, none of which are
        obvious from a loss curve.
        """
        ok = True
        for fusion in ("early", "late"):
            for transformer in (True, False):
                cfg = MotionModelConfig(
                    num_classes=5, fusion=fusion, use_temporal_transformer=transformer,
                    kinematic_dim=12,
                )
                model = build_model(cfg)
                n, t = 2, 48
                x = torch.randn(n, cfg.in_channels, t, cfg.num_joints)
                kin = torch.randn(n, cfg.kinematic_dim)
                out = model(x, kin)

                checks = {
                    "exercise_logits": (n, cfg.num_classes),
                    "phase_logits": (n, t, cfg.num_phases),
                }
                for key, expected in checks.items():
                    got = tuple(out[key].shape)
                    if got != expected:
                        ok = False
                        print(f"  FAIL {fusion}/transformer={transformer} {key}: {got} != {expected}")

                loss = out["exercise_logits"].sum() + out["phase_logits"].sum()
                loss.backward()
                no_grad = [n_ for n_, p in model.named_parameters()
                           if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
                # The adaptive-adjacency residual and the kinematic gate
                # start at zero by design; a zero gradient elsewhere means
                # a disconnected subgraph.
                no_grad = [n_ for n_ in no_grad if "A_residual" not in n_ and "kinematic_gate" not in n_]
                if no_grad:
                    ok = False
                    print(f"  FAIL {fusion}/transformer={transformer}: no gradient for {no_grad[:5]}")

                if verbose:
                    print(f"  ok  fusion={fusion:<5} transformer={str(transformer):<5} "
                          f"params={count_parameters(model):,}")
        if verbose:
            print("selfcheck:", "PASS" if ok else "FAIL")
        return ok

else:  # pragma: no cover

    def build_model(cfg: MotionModelConfig):
        raise ImportError(
            "PyTorch is required to build the model. "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    def selfcheck(verbose: bool = True) -> bool:
        print("torch is not installed; cannot run the model selfcheck.")
        return False


if __name__ == "__main__":
    selfcheck()

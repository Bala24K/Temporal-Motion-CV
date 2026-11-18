"""Skeleton movement encoder (PyTorch): adaptive-adjacency ST-GCN blocks -> temporal transformer -> L2-normalised
embedding. The embedding (not an exercise label) is the product: prototypes, OOD and conformal sets sit on top of it
(motion_engine/uncertainty.py), so a new exercise = a prototype + a definition, not a new network.

NOT executed in the authoring sandbox (no torch there). tests/test_torch_encoder.py runs where torch is installed;
run it before trusting this file. Lives under ml/ (research) - production serves the exported ONNX only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq import V, adjacency


class GraphConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        A = torch.tensor(adjacency(), dtype=torch.float32)
        self.register_buffer("A", torch.stack([torch.eye(V), A - torch.eye(V)]))   # (K=2, V, V)
        self.B = nn.Parameter(torch.zeros(2, V, V))                                  # learnable (adaptive) part
        self.conv = nn.Conv2d(cin, cout * 2, 1)
        self.cout = cout

    def forward(self, x):                                   # (N, C, T, V)
        n, _, t, v = x.shape
        y = self.conv(x).view(n, 2, self.cout, t, v)
        return torch.einsum("nkctv,kvw->nctw", y, self.A + self.B)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, dropout=0.2):
        super().__init__()
        self.gc, self.bn1 = GraphConv(cin, cout), nn.BatchNorm2d(cout)
        self.tc = nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0))
        self.bn2, self.drop = nn.BatchNorm2d(cout), nn.Dropout(dropout)
        self.res = nn.Identity() if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, (stride, 1))

    def forward(self, x):
        y = F.relu(self.bn1(self.gc(x)))
        y = self.drop(self.bn2(self.tc(y)))
        return F.relu(y + self.res(x))


class MovementEncoder(nn.Module):
    def __init__(self, in_ch=4, embed_dim=64, width=64, max_t=128):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_ch * V)
        self.blocks = nn.Sequential(Block(in_ch, 32), Block(32, width, stride=2), Block(width, width))
        self.pos = nn.Parameter(torch.zeros(1, max_t, width))
        self.tr = nn.TransformerEncoderLayer(width, 4, 2 * width, 0.1, batch_first=True)
        self.proj = nn.Linear(width, embed_dim)

    def forward(self, x):                                   # (N, C, T, V) -> (N, D) unit vectors
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x).mean(3).permute(0, 2, 1)         # (N, T', width)
        x = self.tr(x + self.pos[:, : x.shape[1]]).mean(1)
        return F.normalize(self.proj(x), dim=-1)


class Classifier(nn.Module):
    """Encoder + linear head, used only to give the contrastive embedding a supervised anchor."""
    def __init__(self, n_classes, **kw):
        super().__init__()
        self.enc, self.head = MovementEncoder(**kw), None
        self.head = nn.Linear(self.enc.proj.out_features, n_classes)

    def forward(self, x):
        z = self.enc(x)
        return z, self.head(z * 10.0)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalised embeddings."""
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    cnt = pos.sum(1)
    ok = cnt > 0
    if not bool(ok.any()):
        return z.sum() * 0.0
    return (-(logp * pos).sum(1)[ok] / cnt[ok]).mean()


def export_onnx(encoder: MovementEncoder, path: str, window: int = 90, check: bool = True, atol: float = 1e-4) -> float:
    """Export the embedding network; returns max |torch - onnxruntime| on random input (parity check)."""
    encoder.eval()
    dummy = torch.randn(2, 4, window, V)
    kw = dict(input_names=["x"], output_names=["embedding"], dynamic_axes={"x": {0: "N"}, "embedding": {0: "N"}}, opset_version=17)
    try:
        torch.onnx.export(encoder, dummy, path, dynamo=False, **kw)
    except TypeError:                                       # older torch without the `dynamo` flag
        torch.onnx.export(encoder, dummy, path, **kw)
    if not check:
        return 0.0
    import onnxruntime as ort
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(["embedding"], {"x": dummy.numpy()})[0]
    err = float(np.abs(got - encoder(dummy).detach().numpy()).max())
    if err > atol:
        raise RuntimeError(f"ONNX export diverges from torch: max error {err:.2e} > {atol}")
    return err

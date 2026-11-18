"""
Skeleton graph topology — the structural prior the learned motion model
gets for free, instead of having to discover "the wrist is attached to
the elbow" from data.

This is the graph side of `body_model/skeleton.py`. That module answers
"what angle is this joint at right now"; this one answers "which joints
are adjacent, and in what direction relative to the body's root" — the
form an ST-GCN needs.

NODE SET (J = 15)
    The 13 anatomical landmarks the pose layer actually tracks
    (`recognition.feature_engineering.LANDMARK_ORDER`) plus the two
    synthetic midline points already used elsewhere in the codebase
    (`body_model.skeleton.SkeletonFrame`): hip_center and
    shoulder_center. The midline points matter for the graph
    specifically: without them the left and right halves of the body are
    two disconnected components, and no amount of graph convolution will
    move information between an arm and a leg.

EDGE SET
    A tree, rooted at hip_center. Every bone is stored child -> parent,
    which is what makes the bone stream (child_pos - parent_pos) and the
    centripetal/centrifugal partitioning below well defined.

PARTITIONING
    ST-GCN's "spatial configuration" partitioning splits each node's
    neighbourhood into three subsets — self, centripetal (neighbour is
    closer to the root), centrifugal (neighbour is further) — so a
    single graph-conv layer can learn a different transform for
    "information flowing inward toward the torso" than for "information
    flowing outward toward the extremities". That asymmetry is the whole
    point; a plain symmetric adjacency collapses the two and measurably
    underperforms.

Everything here is pure numpy and deterministic, so it is testable
without torch installed.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Node order. Index 0-12 deliberately matches
# recognition.feature_engineering.LANDMARK_ORDER so the two
# representations can be zipped without a lookup table.
JOINT_NAMES: List[str] = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "hip_center",        # 13 — synthetic, root of the tree
    "shoulder_center",   # 14 — synthetic
]

JOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}
NUM_JOINTS = len(JOINT_NAMES)
ROOT_JOINT = "hip_center"
ROOT_INDEX = JOINT_INDEX[ROOT_JOINT]

# The synthetic nodes are derived, not detected — callers that need to
# know which coordinates came from a real landmark check against this.
SYNTHETIC_JOINTS: Tuple[str, ...] = ("hip_center", "shoulder_center")

# Bones as (child, parent). This IS the topology; adjacency, the bone
# stream, and the partitioning are all derived from it below.
BONES: List[Tuple[str, str]] = [
    ("shoulder_center", "hip_center"),   # torso
    ("nose", "shoulder_center"),
    ("left_shoulder", "shoulder_center"),
    ("right_shoulder", "shoulder_center"),
    ("left_elbow", "left_shoulder"),
    ("right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow"),
    ("right_wrist", "right_elbow"),
    ("left_hip", "hip_center"),
    ("right_hip", "hip_center"),
    ("left_knee", "left_hip"),
    ("right_knee", "right_hip"),
    ("left_ankle", "left_knee"),
    ("right_ankle", "right_knee"),
]

# Left/right counterparts, used by the horizontal-flip augmentation. A
# mirror that moves the pixels but not the labels teaches the model that
# a left-arm curl and a right-arm curl are different exercises, which is
# the exact opposite of what the augmentation is for.
MIRROR_PAIRS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def bone_index_pairs() -> np.ndarray:
    """Bones as an integer array of shape (B, 2), columns (child, parent)."""
    return np.array([[JOINT_INDEX[c], JOINT_INDEX[p]] for c, p in BONES], dtype=np.int64)


def parent_array() -> np.ndarray:
    """
    parent[j] = index of j's parent in the tree; parent[root] = root.

    Shape (J,). Used to build the bone stream in one vectorised gather
    rather than a Python loop over bones.
    """
    parent = np.arange(NUM_JOINTS, dtype=np.int64)
    for child, par in BONES:
        parent[JOINT_INDEX[child]] = JOINT_INDEX[par]
    return parent


def mirror_permutation() -> np.ndarray:
    """
    perm[j] = index of the joint that j becomes under a left/right flip.

    Self-inverse: applying it twice is the identity. Midline joints
    (nose, hip_center, shoulder_center) map to themselves.
    """
    perm = np.arange(NUM_JOINTS, dtype=np.int64)
    for a, b in MIRROR_PAIRS:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        perm[ia], perm[ib] = ib, ia
    return perm


def hop_distances() -> np.ndarray:
    """
    Shortest-path hop count between every pair of joints, shape (J, J).

    Unreachable pairs would be np.inf, but the skeleton is a connected
    tree so every entry is finite. Kept as float so callers can compare
    against np.inf safely if the topology is ever edited into pieces.
    """
    inf = float("inf")
    dist = np.full((NUM_JOINTS, NUM_JOINTS), inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for child, par in BONES:
        i, j = JOINT_INDEX[child], JOINT_INDEX[par]
        dist[i, j] = 1.0
        dist[j, i] = 1.0
    # Floyd-Warshall; J = 15, so the cubic cost is irrelevant.
    for k in range(NUM_JOINTS):
        dist = np.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric degree normalisation, D^-1/2 (A) D^-1/2, applied per
    partition. Isolated rows (a partition where a node has no neighbour —
    the root has no centripetal neighbour, for instance) normalise to
    zero rather than dividing by zero.
    """
    degree = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    d = np.diag(inv_sqrt)
    return d @ adj @ d


def spatial_adjacency(normalize: bool = True) -> np.ndarray:
    """
    ST-GCN spatial-configuration adjacency, shape (3, J, J).

        A[0] — self connections (identity)
        A[1] — centripetal: neighbours nearer the root than the node
        A[2] — centrifugal: neighbours further from the root

    Row convention: A[k, i, j] is nonzero when node j contributes to
    node i, so the graph convolution is `sum_k A[k] @ (X W_k)`.
    """
    dist_to_root = hop_distances()[ROOT_INDEX]

    self_adj = np.eye(NUM_JOINTS, dtype=np.float64)
    centripetal = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float64)
    centrifugal = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float64)

    for child, par in BONES:
        i, j = JOINT_INDEX[child], JOINT_INDEX[par]
        # i -> j: j is the parent, so j is nearer the root.
        if dist_to_root[j] < dist_to_root[i]:
            centripetal[i, j] = 1.0
            centrifugal[j, i] = 1.0
        else:  # defensive: only reachable if BONES is edited inconsistently
            centrifugal[i, j] = 1.0
            centripetal[j, i] = 1.0

    stack = np.stack([self_adj, centripetal, centrifugal], axis=0)
    if normalize:
        stack = np.stack([_normalize_adjacency(a) for a in stack], axis=0)
    return stack.astype(np.float32)


def uniform_adjacency(max_hop: int = 1, normalize: bool = True) -> np.ndarray:
    """
    Single-partition adjacency over all pairs within `max_hop`, shape
    (1, J, J). Provided as an ablation baseline: it is what an ST-GCN
    degenerates to without the directional partitioning, and the
    comparison is worth reporting.
    """
    hops = hop_distances()
    adj = ((hops <= max_hop) & np.isfinite(hops)).astype(np.float64)
    if normalize:
        adj = _normalize_adjacency(adj)
    return adj[None, ...].astype(np.float32)


def build_adjacency(strategy: str = "spatial", max_hop: int = 1) -> np.ndarray:
    """Dispatch used by the model config so the strategy is a hyperparameter."""
    if strategy == "spatial":
        return spatial_adjacency()
    if strategy == "uniform":
        return uniform_adjacency(max_hop=max_hop)
    raise ValueError(f"unknown adjacency strategy: {strategy!r} (expected 'spatial' or 'uniform')")


def joint_indices(names: Sequence[str]) -> np.ndarray:
    """Map joint names to indices, raising on an unknown name rather than silently dropping it."""
    missing = [n for n in names if n not in JOINT_INDEX]
    if missing:
        raise KeyError(f"unknown joint name(s): {missing}")
    return np.array([JOINT_INDEX[n] for n in names], dtype=np.int64)

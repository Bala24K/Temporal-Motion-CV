"""
Graph topology tests.

The adjacency is a fixed, hand-specified structure that every downstream
layer trusts silently. A transposed partition or a disconnected limb
produces a model that trains without error and quietly performs worse, so
these properties are asserted rather than eyeballed.
"""
import numpy as np
import pytest

from motion.graph import (
    BONES,
    JOINT_INDEX,
    JOINT_NAMES,
    MIRROR_PAIRS,
    NUM_JOINTS,
    ROOT_INDEX,
    ROOT_JOINT,
    bone_index_pairs,
    build_adjacency,
    hop_distances,
    joint_indices,
    mirror_permutation,
    parent_array,
    spatial_adjacency,
    uniform_adjacency,
)
from recognition.feature_engineering import LANDMARK_ORDER


def test_first_13_joints_match_existing_landmark_order():
    # The two representations are zipped without a lookup table in
    # several places; if this ordering ever diverges, those become silent
    # joint swaps rather than errors.
    assert JOINT_NAMES[:len(LANDMARK_ORDER)] == list(LANDMARK_ORDER)


def test_node_and_bone_counts_form_a_tree():
    assert NUM_JOINTS == 15
    assert len(BONES) == NUM_JOINTS - 1  # a tree has exactly V-1 edges


def test_graph_is_connected():
    hops = hop_distances()
    assert np.isfinite(hops).all(), "every joint must be reachable from every other"


def test_left_and_right_halves_are_connected_through_the_midline():
    # Without the synthetic centre nodes the arms and legs are separate
    # components and no graph convolution can relate them.
    hops = hop_distances()
    left_wrist, right_wrist = JOINT_INDEX["left_wrist"], JOINT_INDEX["right_wrist"]
    assert np.isfinite(hops[left_wrist, right_wrist])
    left_wrist_to_ankle = hops[left_wrist, JOINT_INDEX["left_ankle"]]
    assert np.isfinite(left_wrist_to_ankle)


def test_parent_array_points_toward_the_root():
    parent = parent_array()
    dist = hop_distances()[ROOT_INDEX]
    assert parent[ROOT_INDEX] == ROOT_INDEX, "the root is its own parent"
    for j in range(NUM_JOINTS):
        if j == ROOT_INDEX:
            continue
        assert dist[parent[j]] == dist[j] - 1


def test_bone_index_pairs_match_bone_definitions():
    pairs = bone_index_pairs()
    assert pairs.shape == (len(BONES), 2)
    for (child, par), row in zip(BONES, pairs):
        assert row[0] == JOINT_INDEX[child]
        assert row[1] == JOINT_INDEX[par]


def test_mirror_permutation_is_an_involution():
    perm = mirror_permutation()
    assert np.array_equal(perm[perm], np.arange(NUM_JOINTS))


def test_mirror_permutation_swaps_sides_and_fixes_the_midline():
    perm = mirror_permutation()
    for left, right in MIRROR_PAIRS:
        assert perm[JOINT_INDEX[left]] == JOINT_INDEX[right]
        assert perm[JOINT_INDEX[right]] == JOINT_INDEX[left]
    for midline in ("nose", "hip_center", "shoulder_center"):
        assert perm[JOINT_INDEX[midline]] == JOINT_INDEX[midline]


def test_spatial_adjacency_shape_and_partitions():
    adj = spatial_adjacency(normalize=False)
    assert adj.shape == (3, NUM_JOINTS, NUM_JOINTS)
    self_adj, centripetal, centrifugal = adj
    assert np.array_equal(self_adj, np.eye(NUM_JOINTS, dtype=self_adj.dtype))
    # Centripetal and centrifugal are transposes of each other: every
    # inward edge has a matching outward edge.
    assert np.array_equal(centripetal, centrifugal.T)
    # And they are disjoint — no edge is both.
    assert not np.any((centripetal > 0) & (centrifugal > 0))


def test_root_has_no_centripetal_neighbour():
    _, centripetal, _ = spatial_adjacency(normalize=False)
    assert centripetal[ROOT_INDEX].sum() == 0, f"{ROOT_JOINT} is the root; nothing is nearer"


def test_centripetal_edges_point_from_child_to_parent():
    _, centripetal, _ = spatial_adjacency(normalize=False)
    parent = parent_array()
    for j in range(NUM_JOINTS):
        if j == ROOT_INDEX:
            continue
        assert centripetal[j, parent[j]] == 1.0
        assert centripetal[j].sum() == 1.0, "a tree node has exactly one parent"


def test_normalisation_keeps_rows_bounded_and_handles_empty_partitions():
    adj = spatial_adjacency(normalize=True)
    assert np.isfinite(adj).all(), "an isolated node must not produce a divide-by-zero"
    assert adj.min() >= 0.0
    assert adj.max() <= 1.0 + 1e-6


def test_uniform_adjacency_is_symmetric_and_single_partition():
    adj = uniform_adjacency(max_hop=1, normalize=False)
    assert adj.shape == (1, NUM_JOINTS, NUM_JOINTS)
    assert np.array_equal(adj[0], adj[0].T)


def test_build_adjacency_dispatch_and_error():
    assert build_adjacency("spatial").shape[0] == 3
    assert build_adjacency("uniform").shape[0] == 1
    with pytest.raises(ValueError):
        build_adjacency("nonsense")


def test_joint_indices_raises_on_unknown_name():
    assert list(joint_indices(["left_wrist", "hip_center"])) == [
        JOINT_INDEX["left_wrist"], JOINT_INDEX["hip_center"]
    ]
    with pytest.raises(KeyError):
        joint_indices(["left_eyebrow"])

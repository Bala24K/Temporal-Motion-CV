import numpy as np
import pytest

from ml.seq import EDGES, SWAP, V, adjacency, augment, channels, clip_positions, mirror, resample_time, windows
from motion_engine.skeleton import J, JOINTS
from motion_engine.synthetic import synth_sequence


def test_adjacency_is_symmetric_normalised_and_connected():
    A = adjacency()
    assert A.shape == (V, V) and np.allclose(A, A.T) and (np.diag(A) > 0).all()
    reach = np.linalg.matrix_power((A > 0).astype(int), V) > 0
    assert reach.all()                                              # one connected skeleton
    assert len(EDGES) == V - 1                                      # a tree


def test_mirror_swaps_sides_and_is_an_involution():
    xy = np.random.default_rng(0).normal(size=(10, V, 2)).astype(np.float32)
    m = mirror(xy)
    assert np.allclose(m[:, J["left_wrist"], 0], -xy[:, J["right_wrist"], 0])
    assert np.allclose(m[:, J["nose"], 0], -xy[:, J["nose"], 0]) and np.allclose(mirror(m), xy)
    assert sorted(SWAP) == list(range(V))


def test_channels_and_windows_shapes():
    fr, _ = synth_sequence("squat", reps=3)
    xy = clip_positions(fr)
    assert xy.shape == (len(fr), V, 2) and np.isfinite(xy).all()
    ws = windows(xy, 90, 45)
    assert len(ws) >= 3 and channels(ws[0]).shape == (4, 90, V)
    assert np.allclose(channels(ws[0])[2:, 0], 0) and windows(xy[:30], 90) == []


def test_clip_positions_are_scale_and_position_invariant():
    a = clip_positions(synth_sequence("bicep_curl", reps=2, scale=1.0, noise=0)[0])
    b = clip_positions(synth_sequence("bicep_curl", reps=2, scale=0.6, noise=0)[0])
    assert np.abs(a - b).max() < 0.05


def test_augment_preserves_shape_and_changes_data_deterministically():
    xy = clip_positions(synth_sequence("squat", reps=3)[0])
    a, b = augment(xy, 90, np.random.default_rng(1)), augment(xy, 90, np.random.default_rng(1))
    c = augment(xy, 90, np.random.default_rng(2))
    assert a.shape == (90, V, 2) and a.dtype == np.float32 and np.array_equal(a, b) and not np.array_equal(a, c)
    short = augment(xy[:50], 90, np.random.default_rng(3))
    assert short.shape == (90, V, 2)                                  # clips shorter than the window are stretched, not crashed


def test_resample_time_endpoints():
    xy = np.random.default_rng(0).normal(size=(40, V, 2)).astype(np.float32)
    r = resample_time(xy, 90)
    assert r.shape == (90, V, 2) and np.allclose(r[0], xy[0]) and np.allclose(r[-1], xy[-1], atol=1e-5)
